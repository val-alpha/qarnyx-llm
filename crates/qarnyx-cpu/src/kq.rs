//! AVX2 integer-dot kernels for GGML Q4_K, Q5_K, and Q6_K weights.
//!
//! F32 activations are quantized once into reusable Q8_K blocks. The kernels
//! then perform packed integer dot products and apply the per-block scales.
//! Q4_K/Q5_K minimum corrections and Q6_K's -32 bias use Q8_K partial sums.
//!
//! Callers must verify AVX2+FMA support before invoking the dot kernels.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

/// Elements per K-quant super-block.
pub(crate) const QK_K: usize = 256;
const Q4K_BYTES: usize = 144;
const Q5K_BYTES: usize = 176;
const Q6K_BYTES: usize = 210;

/// Q8_K is an intermediate activation format, not a model-storage format.
#[repr(C)]
pub(crate) struct BlockQ8K {
    pub(crate) d: f32,
    pub(crate) qs: [i8; QK_K],
    pub(crate) bsums: [i16; QK_K / 16],
}

impl Default for BlockQ8K {
    fn default() -> Self {
        Self {
            d: 0.0,
            qs: [0; QK_K],
            bsums: [0; QK_K / 16],
        }
    }
}

/// Quantizes F32 activations into Q8_K blocks.
///
/// Each block uses an absmax scale and stores signed int8 values plus sums of
/// each 16-value group. Returns false for non-finite input so the caller can
/// retain the unquantized fallback instead of silently changing its behavior.
pub(crate) fn quantize_activations(input: &[f32], output: &mut Vec<BlockQ8K>) -> bool {
    if input.len() % QK_K != 0 {
        return false;
    }

    output.resize_with(input.len() / QK_K, BlockQ8K::default);
    for (source, target) in input.chunks_exact(QK_K).zip(output.iter_mut()) {
        // Fold finite validation into each block's absmax pass. Q8_K uses an
        // independent scale for every 256 activation values.
        let mut absmax = 0.0_f32;
        for &value in source {
            if !value.is_finite() {
                return false;
            }
            absmax = absmax.max(value.abs());
        }

        if absmax == 0.0 {
            target.d = 0.0;
            target.qs.fill(0);
            target.bsums.fill(0);
            continue;
        }

        let d = absmax / 127.0;
        let inverse = 127.0 / absmax;
        target.d = d;

        // Quantize and accumulate 16-element partial sums in one pass.
        for (group, source_group) in source.chunks_exact(16).enumerate() {
            let start = group * 16;
            let mut sum = 0_i16;
            for (offset, value) in source_group.iter().copied().enumerate() {
                let quantized = (value * inverse).round_ties_even().clamp(-127.0, 127.0) as i8;
                target.qs[start + offset] = quantized;
                sum += i16::from(quantized);
            }
            target.bsums[group] = sum;
        }
    }
    true
}

#[inline]
fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = u32::from(bits & 0x03ff);
    let output = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let mut normalized = mantissa;
            let mut unbiased = -14_i32;
            while normalized & 0x0400 == 0 {
                normalized <<= 1;
                unbiased -= 1;
            }
            normalized &= 0x03ff;
            sign | (((unbiased + 127) as u32) << 23) | (normalized << 13)
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((u32::from(exponent) + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(output)
}

#[inline]
fn read_half(bytes: &[u8]) -> f32 {
    half_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]))
}

/// GGML get_scale_min_k4: 6-bit scale and minimum for group index 0..8.
#[inline]
fn scale_min(index: usize, packed: &[u8]) -> (u8, u8) {
    if index < 4 {
        (packed[index] & 0x3f, packed[index + 4] & 0x3f)
    } else {
        (
            (packed[index + 4] & 0x0f) | ((packed[index - 4] >> 6) << 4),
            (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4),
        )
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_i32(values: __m256i) -> i32 {
    unsafe {
        let mut lanes = [0_i32; 8];
        _mm256_storeu_si256(lanes.as_mut_ptr().cast(), values);
        lanes.into_iter().sum()
    }
}

/// Dot one 32-element group of unsigned 8-bit weights and signed Q8 activations.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn dot_u8_i8_32(weights: __m256i, activations: *const i8) -> i32 {
    unsafe {
        let activation_vector = _mm256_loadu_si256(activations.cast());
        // maddubs multiplies unsigned weights by signed activations, then sums
        // adjacent products into i16 lanes. Q4_K/Q5_K bounds fit in i16.
        let pair_products = _mm256_maddubs_epi16(weights, activation_vector);
        let quad_sums = _mm256_madd_epi16(pair_products, _mm256_set1_epi16(1));
        horizontal_sum_i32(quad_sums)
    }
}

/// Dot one 32-element Q6_K plane, preserving its two distinct 16-value scales.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn dot_u6_i8_32(weights: __m256i, activations: *const i8) -> (i32, i32) {
    unsafe {
        let activation_vector = _mm256_loadu_si256(activations.cast());
        let pair_products = _mm256_maddubs_epi16(weights, activation_vector);
        let quad_sums = _mm256_madd_epi16(pair_products, _mm256_set1_epi16(1));
        let mut lanes = [0_i32; 8];
        _mm256_storeu_si256(lanes.as_mut_ptr().cast(), quad_sums);
        (lanes[..4].iter().sum(), lanes[4..].iter().sum())
    }
}

/// Dot product of one Q4_K row against already-quantized Q8_K activations.
pub(crate) fn dot_q4_k_q8(encoded: &[u8], activations: &[BlockQ8K]) -> f32 {
    let blocks = encoded.len() / Q4K_BYTES;
    assert_eq!(encoded.len(), blocks * Q4K_BYTES, "partial Q4_K block");
    assert_eq!(activations.len(), blocks, "Q8_K block count mismatch");
    // SAFETY: lengths validated above; callers verified AVX2+FMA.
    unsafe { dot_q4_k_q8_impl(encoded, activations) }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q4_k_q8_impl(encoded: &[u8], activations: &[BlockQ8K]) -> f32 {
    unsafe {
        let nibble_mask = _mm256_set1_epi8(0x0f);
        let mut total = 0.0_f32;

        for (block, activation) in encoded.chunks_exact(Q4K_BYTES).zip(activations.iter()) {
            let d = read_half(&block[0..2]);
            let dmin = read_half(&block[2..4]);
            let packed_scales = &block[4..16];
            let quants = block.as_ptr().add(16);
            let combined_scale = d * activation.d;

            for group in 0..8 {
                let pair = group / 2;
                let packed = _mm256_loadu_si256(quants.add(pair * 32).cast());
                let values = if group % 2 == 0 {
                    _mm256_and_si256(packed, nibble_mask)
                } else {
                    _mm256_and_si256(_mm256_srli_epi16::<4>(packed), nibble_mask)
                };
                let (scale, minimum) = scale_min(group, packed_scales);
                let activation_sum = i32::from(activation.bsums[group * 2])
                    + i32::from(activation.bsums[group * 2 + 1]);
                let dot = dot_u8_i8_32(values, activation.qs.as_ptr().add(group * 32));
                total += combined_scale * f32::from(scale) * dot as f32
                    - dmin * activation.d * f32::from(minimum) * activation_sum as f32;
            }
        }
        total
    }
}

/// Dot product of one Q5_K row against already-quantized Q8_K activations.
pub(crate) fn dot_q5_k_q8(encoded: &[u8], activations: &[BlockQ8K]) -> f32 {
    let blocks = encoded.len() / Q5K_BYTES;
    assert_eq!(encoded.len(), blocks * Q5K_BYTES, "partial Q5_K block");
    assert_eq!(activations.len(), blocks, "Q8_K block count mismatch");
    // SAFETY: lengths validated above; callers verified AVX2+FMA.
    unsafe { dot_q5_k_q8_impl(encoded, activations) }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q5_k_q8_impl(encoded: &[u8], activations: &[BlockQ8K]) -> f32 {
    unsafe {
        let nibble_mask = _mm256_set1_epi8(0x0f);
        let sixteen = _mm256_set1_epi8(16);
        let mut total = 0.0_f32;

        for (block, activation) in encoded.chunks_exact(Q5K_BYTES).zip(activations.iter()) {
            let d = read_half(&block[0..2]);
            let dmin = read_half(&block[2..4]);
            let packed_scales = &block[4..16];
            let high_bits = _mm256_loadu_si256(block.as_ptr().add(16).cast());
            let quants = block.as_ptr().add(48);
            let combined_scale = d * activation.d;

            for group in 0..8 {
                let pair = group / 2;
                let packed = _mm256_loadu_si256(quants.add(pair * 32).cast());
                let nibble = if group % 2 == 0 {
                    _mm256_and_si256(packed, nibble_mask)
                } else {
                    _mm256_and_si256(_mm256_srli_epi16::<4>(packed), nibble_mask)
                };
                let mask = _mm256_set1_epi8((1_u8 << group) as i8);
                let high_bit = _mm256_and_si256(
                    _mm256_cmpeq_epi8(_mm256_and_si256(high_bits, mask), mask),
                    sixteen,
                );
                let values = _mm256_or_si256(nibble, high_bit);
                let (scale, minimum) = scale_min(group, packed_scales);
                let activation_sum = i32::from(activation.bsums[group * 2])
                    + i32::from(activation.bsums[group * 2 + 1]);
                let dot = dot_u8_i8_32(values, activation.qs.as_ptr().add(group * 32));
                total += combined_scale * f32::from(scale) * dot as f32
                    - dmin * activation.d * f32::from(minimum) * activation_sum as f32;
            }
        }
        total
    }
}

/// Dot product of one Q6_K row against already-quantized Q8_K activations.
pub(crate) fn dot_q6_k_q8(encoded: &[u8], activations: &[BlockQ8K]) -> f32 {
    let blocks = encoded.len() / Q6K_BYTES;
    assert_eq!(encoded.len(), blocks * Q6K_BYTES, "partial Q6_K block");
    assert_eq!(activations.len(), blocks, "Q8_K block count mismatch");
    // SAFETY: lengths validated above; callers verified AVX2+FMA.
    unsafe { dot_q6_k_q8_impl(encoded, activations) }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q6_k_q8_impl(encoded: &[u8], activations: &[BlockQ8K]) -> f32 {
    unsafe {
        let low_mask = _mm256_set1_epi8(0x0f);
        let m03 = _mm256_set1_epi8(0x03);
        let m0c = _mm256_set1_epi8(0x0c);
        let m30 = _mm256_set1_epi8(0x30);
        let mut total = 0.0_f32;

        for (block, activation) in encoded.chunks_exact(Q6K_BYTES).zip(activations.iter()) {
            let d = read_half(&block[208..210]);
            let scales = &block[192..208];
            let combined_scale = d * activation.d;

            for half in 0..2 {
                let ql = block.as_ptr().add(half * 64);
                let qh = block.as_ptr().add(128 + half * 32);
                let l0 = _mm256_loadu_si256(ql.cast());
                let l1 = _mm256_loadu_si256(ql.add(32).cast());
                let high = _mm256_loadu_si256(qh.cast());

                // Four 32-value planes reconstruct the unsigned [0, 63] codes.
                let h0 = _mm256_slli_epi16::<4>(_mm256_and_si256(high, m03));
                let h1 = _mm256_slli_epi16::<2>(_mm256_and_si256(high, m0c));
                let h2 = _mm256_and_si256(high, m30);
                let h3 = _mm256_and_si256(_mm256_srli_epi16::<2>(high), m30);
                let planes = [
                    _mm256_or_si256(_mm256_and_si256(l0, low_mask), h0),
                    _mm256_or_si256(_mm256_and_si256(l1, low_mask), h1),
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16::<4>(l0), low_mask), h2),
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16::<4>(l1), low_mask), h3),
                ];

                for (plane_index, plane) in planes.iter().enumerate() {
                    let group = half * 8 + plane_index * 2;
                    let activation_offset = half * 128 + plane_index * 32;
                    let (dot_low, dot_high) =
                        dot_u6_i8_32(*plane, activation.qs.as_ptr().add(activation_offset));

                    let low_sum = i32::from(activation.bsums[group]);
                    let high_sum = i32::from(activation.bsums[group + 1]);
                    let low_scale = f32::from(scales[group] as i8);
                    let high_scale = f32::from(scales[group + 1] as i8);
                    let low_corrected = dot_low - 32 * low_sum;
                    let high_corrected = dot_high - 32 * high_sum;

                    total += combined_scale
                        * (low_scale * low_corrected as f32 + high_scale * high_corrected as f32);
                }
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_k_quantization_preserves_block_sums_and_zero_blocks() {
        let mut input = vec![0.0_f32; QK_K * 2];
        for (index, value) in input[..QK_K].iter_mut().enumerate() {
            *value = (index as f32 - 127.0) / 31.0;
        }
        let mut blocks = Vec::new();

        assert!(quantize_activations(&input, &mut blocks));
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[1].d, 0.0);
        assert!(blocks[1].qs.iter().all(|value| *value == 0));
        assert!(blocks[1].bsums.iter().all(|value| *value == 0));

        for block in &blocks[..1] {
            for (group, sum) in block.bsums.iter().enumerate() {
                let start = group * 16;
                let expected: i16 = block.qs[start..start + 16]
                    .iter()
                    .map(|value| i16::from(*value))
                    .sum();
                assert_eq!(*sum, expected);
            }
            assert!(block.qs.iter().all(|value| (-127..=127).contains(value)));
        }
    }

    #[test]
    fn q8_k_quantization_declines_non_finite_inputs() {
        let mut input = vec![0.0_f32; QK_K];
        input[17] = f32::NAN;
        let mut blocks = Vec::new();
        assert!(!quantize_activations(&input, &mut blocks));
    }
}
