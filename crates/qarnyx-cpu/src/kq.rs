//! Experimental AVX2/FMA direct-block Q4_K dot product.
//!
//! This alternative consumes the original F32 activation vector rather than
//! quantizing it to Q8_K. Per-32 activation sums are computed once per matvec
//! and reused for the Q4_K minimum correction term. It is kept as an A/B
//! candidate against the production Q4_K x Q8_K integer kernel.
//!
//! Callers must verify AVX2+FMA support before invoking the dot kernel.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

/// Elements per K-quant super-block.
pub(crate) const QK_K: usize = 256;
const Q4K_BYTES: usize = 144;

/// Sums every 32-element group of `input` into `out` (`out.len() == input.len() / 32`).
///
/// Q4_K and Q5_K need `sum(x)` per 32-element group for their min term. It does
/// not depend on the weights, so one pass per matvec serves every row.
pub(crate) fn group_sums(input: &[f32], out: &mut [f32]) {
    assert_eq!(input.len() % 32, 0, "input length must be a multiple of 32");
    assert_eq!(out.len(), input.len() / 32, "one sum per 32-element group");
    for (group, slot) in input.chunks_exact(32).zip(out.iter_mut()) {
        let mut lanes = [0.0_f32; 8];
        for chunk in group.chunks_exact(8) {
            for (lane, value) in lanes.iter_mut().zip(chunk) {
                *lane += *value;
            }
        }
        *slot = lanes.iter().sum();
    }
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

/// GGML `get_scale_min_k4`: 6-bit scale and min for group `index` (0..8).
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
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum(v: __m256) -> f32 {
    let low = _mm256_castps256_ps128(v);
    let high = _mm256_extractf128_ps::<1>(v);
    let sum = _mm_add_ps(low, high);
    let sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    let sum = _mm_add_ss(sum, _mm_shuffle_ps::<1>(sum, sum));
    _mm_cvtss_f32(sum)
}

/// Loads 8 unsigned bytes starting at `lane` of a 32-byte vector, as `f32`.
macro_rules! u8_lanes_to_f32 {
    ($halves:expr, $chunk:expr) => {{
        // `$halves` is [low128, high128]; chunk 0..4 selects an 8-byte slice.
        let source = $halves[$chunk / 2];
        let shifted = if $chunk % 2 == 0 {
            source
        } else {
            _mm_srli_si128::<8>(source)
        };
        _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(shifted))
    }};
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn halves(v: __m256i) -> [__m128i; 2] {
    [_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v)]
}

/// Dot product of one Q4_K row with `input`.
///
/// `sums` must come from [`group_sums`] for the same `input`.
pub(crate) fn dot_q4_k(encoded: &[u8], input: &[f32], sums: &[f32]) -> f32 {
    let blocks = encoded.len() / Q4K_BYTES;
    assert_eq!(encoded.len(), blocks * Q4K_BYTES, "partial Q4_K block");
    assert_eq!(input.len(), blocks * QK_K, "input length must match blocks");
    assert_eq!(sums.len(), blocks * 8, "one group sum per 32 inputs");
    // SAFETY: lengths validated above; callers verified AVX2+FMA support.
    unsafe { dot_q4_k_impl(encoded, input, sums) }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q4_k_impl(encoded: &[u8], input: &[f32], sums: &[f32]) -> f32 {
    unsafe {
        let low_mask = _mm256_set1_epi8(0x0f);
        let mut acc_even = _mm256_setzero_ps();
        let mut acc_odd = _mm256_setzero_ps();
        let mut min_total = 0.0_f32;

        for (block_index, block) in encoded.chunks_exact(Q4K_BYTES).enumerate() {
            let d = read_half(&block[0..2]);
            let dmin = read_half(&block[2..4]);
            let packed_scales = &block[4..16];
            let quants = block.as_ptr().add(16);
            let x_base = input.as_ptr().add(block_index * QK_K);
            let sum_base = block_index * 8;

            for pair in 0..4 {
                let packed = _mm256_loadu_si256(quants.add(pair * 32).cast());
                let low = halves(_mm256_and_si256(packed, low_mask));
                let high = halves(_mm256_and_si256(_mm256_srli_epi16::<4>(packed), low_mask));

                let x_even = x_base.add(pair * 64);
                let x_odd = x_even.add(32);

                // Even group: low nibbles. Two independent chains of two FMAs.
                let mut a0 = _mm256_mul_ps(u8_lanes_to_f32!(low, 0), _mm256_loadu_ps(x_even));
                let mut a1 =
                    _mm256_mul_ps(u8_lanes_to_f32!(low, 1), _mm256_loadu_ps(x_even.add(8)));
                a0 = _mm256_fmadd_ps(
                    u8_lanes_to_f32!(low, 2),
                    _mm256_loadu_ps(x_even.add(16)),
                    a0,
                );
                a1 = _mm256_fmadd_ps(
                    u8_lanes_to_f32!(low, 3),
                    _mm256_loadu_ps(x_even.add(24)),
                    a1,
                );

                // Odd group: high nibbles.
                let mut b0 = _mm256_mul_ps(u8_lanes_to_f32!(high, 0), _mm256_loadu_ps(x_odd));
                let mut b1 =
                    _mm256_mul_ps(u8_lanes_to_f32!(high, 1), _mm256_loadu_ps(x_odd.add(8)));
                b0 = _mm256_fmadd_ps(
                    u8_lanes_to_f32!(high, 2),
                    _mm256_loadu_ps(x_odd.add(16)),
                    b0,
                );
                b1 = _mm256_fmadd_ps(
                    u8_lanes_to_f32!(high, 3),
                    _mm256_loadu_ps(x_odd.add(24)),
                    b1,
                );

                let (sc_even, mn_even) = scale_min(pair * 2, packed_scales);
                let (sc_odd, mn_odd) = scale_min(pair * 2 + 1, packed_scales);
                acc_even = _mm256_fmadd_ps(
                    _mm256_add_ps(a0, a1),
                    _mm256_set1_ps(d * f32::from(sc_even)),
                    acc_even,
                );
                acc_odd = _mm256_fmadd_ps(
                    _mm256_add_ps(b0, b1),
                    _mm256_set1_ps(d * f32::from(sc_odd)),
                    acc_odd,
                );
                min_total += dmin * f32::from(mn_even) * sums[sum_base + pair * 2]
                    + dmin * f32::from(mn_odd) * sums[sum_base + pair * 2 + 1];
            }
        }

        hsum(_mm256_add_ps(acc_even, acc_odd)) - min_total
    }
}

