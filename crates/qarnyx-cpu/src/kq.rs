//! AVX2/FMA direct-block dot products for K-quant weight rows.
//!
//! Each kernel computes `dot(weight_row, input)` straight from the packed GGUF
//! blocks. No F32 copy of the weight row is ever materialized, and the 4-bit /
//! 5-bit / 6-bit values are unpacked with byte-wise SIMD rather than scalar loops.
//!
//! Design notes (why these are faster than the earlier fused Q4_K kernel):
//! - Scales are applied once per 16/32-element group with a vector FMA, not once
//!   per weight: `sum(d*sc*q*x - dmin*m*x)` is evaluated as
//!   `d*sc*sum(q*x) - dmin*m*sum(x)`.
//! - `sum(x)` per 32-element group is independent of the weight row, so callers
//!   compute it once per matvec ([`group_sums`]) and reuse it for every row.
//! - Several independent accumulators hide FMA latency.
//!
//! Every public function here is safe to call: it validates lengths with hard
//! asserts and requires the caller to have verified AVX2+FMA support (see
//! [`available`]).

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

/// Elements per K-quant super-block.
pub(crate) const QK_K: usize = 256;
const Q4K_BYTES: usize = 144;
const Q5K_BYTES: usize = 176;
const Q6K_BYTES: usize = 210;

/// Returns whether the host can run these kernels.
#[inline]
pub(crate) fn available() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

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

macro_rules! i8_lanes_to_f32 {
    ($halves:expr, $chunk:expr) => {{
        let source = $halves[$chunk / 2];
        let shifted = if $chunk % 2 == 0 {
            source
        } else {
            _mm_srli_si128::<8>(source)
        };
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(shifted))
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
    // SAFETY: lengths validated above; callers verified `available()`.
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

/// Dot product of one Q5_K row with `input`. `sums` as for [`dot_q4_k`].
pub(crate) fn dot_q5_k(encoded: &[u8], input: &[f32], sums: &[f32]) -> f32 {
    let blocks = encoded.len() / Q5K_BYTES;
    assert_eq!(encoded.len(), blocks * Q5K_BYTES, "partial Q5_K block");
    assert_eq!(input.len(), blocks * QK_K, "input length must match blocks");
    assert_eq!(sums.len(), blocks * 8, "one group sum per 32 inputs");
    // SAFETY: lengths validated above; callers verified `available()`.
    unsafe { dot_q5_k_impl(encoded, input, sums) }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q5_k_impl(encoded: &[u8], input: &[f32], sums: &[f32]) -> f32 {
    unsafe {
        let low_mask = _mm256_set1_epi8(0x0f);
        let sixteen = _mm256_set1_epi8(16);
        let mut acc_even = _mm256_setzero_ps();
        let mut acc_odd = _mm256_setzero_ps();
        let mut min_total = 0.0_f32;

        for (block_index, block) in encoded.chunks_exact(Q5K_BYTES).enumerate() {
            let d = read_half(&block[0..2]);
            let dmin = read_half(&block[2..4]);
            let packed_scales = &block[4..16];
            let high_bits = _mm256_loadu_si256(block.as_ptr().add(16).cast());
            let quants = block.as_ptr().add(48);
            let x_base = input.as_ptr().add(block_index * QK_K);
            let sum_base = block_index * 8;

            for pair in 0..4 {
                let packed = _mm256_loadu_si256(quants.add(pair * 32).cast());
                let mask_even = _mm256_set1_epi8((1_u8 << (pair * 2)) as i8);
                let mask_odd = _mm256_set1_epi8((2_u8 << (pair * 2)) as i8);
                let bit_even = _mm256_and_si256(
                    _mm256_cmpeq_epi8(_mm256_and_si256(high_bits, mask_even), mask_even),
                    sixteen,
                );
                let bit_odd = _mm256_and_si256(
                    _mm256_cmpeq_epi8(_mm256_and_si256(high_bits, mask_odd), mask_odd),
                    sixteen,
                );
                let low = halves(_mm256_or_si256(
                    _mm256_and_si256(packed, low_mask),
                    bit_even,
                ));
                let high = halves(_mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16::<4>(packed), low_mask),
                    bit_odd,
                ));

                let x_even = x_base.add(pair * 64);
                let x_odd = x_even.add(32);

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

/// Dot product of one Q6_K row with `input`. No group sums are needed: the
/// `-32` bias is removed in the integer domain before conversion to `f32`.
pub(crate) fn dot_q6_k(encoded: &[u8], input: &[f32]) -> f32 {
    let blocks = encoded.len() / Q6K_BYTES;
    assert_eq!(encoded.len(), blocks * Q6K_BYTES, "partial Q6_K block");
    assert_eq!(input.len(), blocks * QK_K, "input length must match blocks");
    // SAFETY: lengths validated above; callers verified `available()`.
    unsafe { dot_q6_k_impl(encoded, input) }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q6_k_impl(encoded: &[u8], input: &[f32]) -> f32 {
    unsafe {
        let low_mask = _mm256_set1_epi8(0x0f);
        let bias = _mm256_set1_epi8(32);
        let m03 = _mm256_set1_epi8(0x03);
        let m0c = _mm256_set1_epi8(0x0c);
        let m30 = _mm256_set1_epi8(0x30);
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        for (block_index, block) in encoded.chunks_exact(Q6K_BYTES).enumerate() {
            let d = read_half(&block[208..210]);
            let scales = &block[192..208];
            let x_block = input.as_ptr().add(block_index * QK_K);

            for half in 0..2 {
                let ql = block.as_ptr().add(half * 64);
                let qh = block.as_ptr().add(128 + half * 32);
                let x_half = x_block.add(half * 128);

                let l0 = _mm256_loadu_si256(ql.cast());
                let l1 = _mm256_loadu_si256(ql.add(32).cast());
                let h = _mm256_loadu_si256(qh.cast());

                // Four 32-value planes: 128 weights per half block.
                let h0 = _mm256_slli_epi16::<4>(_mm256_and_si256(h, m03));
                let h1 = _mm256_slli_epi16::<2>(_mm256_and_si256(h, m0c));
                let h2 = _mm256_and_si256(h, m30);
                let h3 = _mm256_and_si256(_mm256_srli_epi16::<2>(h), m30);

                let planes = [
                    _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(l0, low_mask), h0), bias),
                    _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(l1, low_mask), h1), bias),
                    _mm256_sub_epi8(
                        _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16::<4>(l0), low_mask), h2),
                        bias,
                    ),
                    _mm256_sub_epi8(
                        _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16::<4>(l1), low_mask), h3),
                        bias,
                    ),
                ];

                for (plane_index, plane) in planes.iter().enumerate() {
                    let parts = halves(*plane);
                    let x = x_half.add(plane_index * 32);
                    let scale_low = f32::from(scales[half * 8 + plane_index * 2] as i8) * d;
                    let scale_high = f32::from(scales[half * 8 + plane_index * 2 + 1] as i8) * d;

                    // Bytes 0..16 use one scale, bytes 16..32 the next.
                    let g0 = _mm256_fmadd_ps(
                        i8_lanes_to_f32!(parts, 1),
                        _mm256_loadu_ps(x.add(8)),
                        _mm256_mul_ps(i8_lanes_to_f32!(parts, 0), _mm256_loadu_ps(x)),
                    );
                    let g1 = _mm256_fmadd_ps(
                        i8_lanes_to_f32!(parts, 3),
                        _mm256_loadu_ps(x.add(24)),
                        _mm256_mul_ps(i8_lanes_to_f32!(parts, 2), _mm256_loadu_ps(x.add(16))),
                    );
                    acc0 = _mm256_fmadd_ps(g0, _mm256_set1_ps(scale_low), acc0);
                    acc1 = _mm256_fmadd_ps(g1, _mm256_set1_ps(scale_high), acc1);
                }
            }
        }

        hsum(_mm256_add_ps(acc0, acc1))
    }
}
