//! Runtime-dispatched CPU execution backend.
//!
//! Matrix kernels use reusable F32 scratch rows for general quantized formats.
//! Experimental direct-block Q4_K, Q5_K, and Q6_K matvec paths avoid F32 row
//! materialization. On x86_64, AVX2/FMA accelerates Q4_K and F32 dot products;
//! AVX2 handles Q5_K/Q6_K block expansion. Scalar kernels remain the reference
//! and portable fallback. Large matrix operations partition independent rows.

use qarnyx_core::{Backend, BackendError, DType, TensorView};
use qarnyx_quant::{
    dequantize_into, dot_q4_k as dot_q4_k_scalar, dot_q5_k as dot_q5_k_scalar,
    dot_q6_k as dot_q6_k_scalar, encoded_len,
};
use rayon::ThreadPool;
use rayon::prelude::*;
use std::sync::Arc;
use thiserror::Error;

/// Errors raised while constructing the CPU backend.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CpuBackendError {
    /// Worker count must be non-zero.
    #[error("CPU backend requires at least one worker")]
    ZeroWorkers,

    /// The configured persistent worker pool could not be created.
    #[error("failed to create CPU worker pool: {0}")]
    WorkerPoolBuild(String),
}

/// Runtime-detected CPU features used for kernel dispatch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuFeatures {
    /// x86 AVX2 support.
    pub avx2: bool,
    /// x86 AVX-512 Foundation support.
    pub avx512f: bool,
    /// x86 FMA support.
    pub fma: bool,
    /// AArch64 Advanced SIMD support.
    pub neon: bool,
}

impl CpuFeatures {
    /// Detects features available to this process on the current CPU.
    pub fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            return Self {
                avx2: std::is_x86_feature_detected!("avx2"),
                avx512f: std::is_x86_feature_detected!("avx512f"),
                fma: std::is_x86_feature_detected!("fma"),
                neon: false,
            };
        }

        #[cfg(target_arch = "aarch64")]
        {
            return Self {
                avx2: false,
                avx512f: false,
                fma: false,
                neon: true,
            };
        }

        #[allow(unreachable_code)]
        Self::default()
    }
}

/// CPU backend configuration.
///
/// Kernels dispatch at runtime to AVX2/FMA on supported x86_64 hosts and
/// otherwise use the scalar reference implementation. Large matrix output
/// rows are scheduled on a persistent Rayon pool shared by backend clones.
#[derive(Debug, Clone)]
pub struct CpuBackend {
    worker_count: usize,
    pool: Arc<ThreadPool>,
    features: CpuFeatures,
    simd_enabled: bool,
    fused_q4k_matvec: bool,
    fused_q5k_q6k_matvec: bool,
}

impl CpuBackend {
    /// Creates the default backend with runtime SIMD dispatch.
    ///
    /// Quantized rows use the reference dequantization path before the F32 dot.
    pub fn new(worker_count: usize) -> Result<Self, CpuBackendError> {
        Self::with_options(worker_count, true, false, false)
    }

    /// Creates a scalar-only backend for reference testing and benchmarks.
    pub fn new_scalar(worker_count: usize) -> Result<Self, CpuBackendError> {
        Self::with_options(worker_count, false, false, false)
    }

    /// Creates a runtime-SIMD backend with the experimental fused Q4_K matvec.
    ///
    /// Q4_K rows are dotted directly from packed blocks using AVX2/FMA where
    /// available, with a scalar packed-block fallback.
    pub fn new_fused_q4k(worker_count: usize) -> Result<Self, CpuBackendError> {
        Self::with_options(worker_count, true, true, false)
    }

    /// Creates a runtime-SIMD backend with direct-block Q4_K, Q5_K, and Q6_K matvecs.
    ///
    /// Q4_K uses AVX2/FMA where available. Q5_K and Q6_K use AVX2 direct-block
    /// kernels on supported x86_64 hosts and scalar direct-block references otherwise.
    pub fn new_fused_qk(worker_count: usize) -> Result<Self, CpuBackendError> {
        Self::with_options(worker_count, true, true, true)
    }

    fn with_options(
        worker_count: usize,
        simd_enabled: bool,
        fused_q4k_matvec: bool,
        fused_q5k_q6k_matvec: bool,
    ) -> Result<Self, CpuBackendError> {
        if worker_count == 0 {
            return Err(CpuBackendError::ZeroWorkers);
        }

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .thread_name(|index| format!("qarnyx-cpu-{index}"))
            .build()
            .map_err(|error| CpuBackendError::WorkerPoolBuild(error.to_string()))?;

        Ok(Self {
            worker_count,
            pool: Arc::new(pool),
            features: CpuFeatures::detect(),
            simd_enabled,
            fused_q4k_matvec,
            fused_q5k_q6k_matvec,
        })
    }

    /// Returns the configured worker count.
    ///
    /// Large matrix operations schedule independent output rows on the
    /// persistent worker pool configured at construction time.
    pub const fn worker_count(&self) -> usize {
        self.worker_count
    }

    /// Returns detected CPU features.
    pub const fn features(&self) -> CpuFeatures {
        self.features
    }

    /// Returns whether runtime SIMD dispatch was requested and AVX2 is usable.
    pub const fn uses_avx2(&self) -> bool {
        cfg!(target_arch = "x86_64") && self.simd_enabled && self.features.avx2
    }

    fn dot_product(&self, lhs: &[f32], rhs: &[f32]) -> f32 {
        if self.simd_enabled && self.features.avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if self.features.fma {
                    return dot_avx2_fma(lhs, rhs);
                }
                return dot_avx2(lhs, rhs);
            }
        }
        dot_scalar(lhs, rhs)
    }

    fn fused_q4k_dot(&self, encoded: &[u8], input: &[f32]) -> Result<f32, BackendError> {
        if self.simd_enabled && self.features.avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if self.features.fma {
                    return Ok(dot_q4_k_avx2_fma(encoded, input));
                }
                return Ok(dot_q4_k_avx2(encoded, input));
            }
        }

        dot_q4_k_scalar(encoded, input).map_err(|error| shape_mismatch("matvec", error.to_string()))
    }

    fn fused_q4k_q8k_dot(&self, encoded: &[u8], input: &[Q8KBlock]) -> f32 {
        if self.simd_enabled && self.features.avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                return dot_q4_k_q8_k_avx2(encoded, input);
            }
        }

        dot_q4_k_q8_k_scalar(encoded, input)
    }

    fn fused_q5k_dot(&self, encoded: &[u8], input: &[f32]) -> Result<f32, BackendError> {
        if self.simd_enabled && self.features.avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if self.features.fma {
                    return Ok(dot_q5_k_avx2_fma(encoded, input));
                }
                return Ok(dot_q5_k_avx2(encoded, input));
            }
        }

        dot_q5_k_scalar(encoded, input).map_err(|error| shape_mismatch("matvec", error.to_string()))
    }

    fn fused_q6k_dot(&self, encoded: &[u8], input: &[f32]) -> Result<f32, BackendError> {
        if self.simd_enabled && self.features.avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if self.features.fma {
                    return Ok(dot_q6_k_avx2_fma(encoded, input));
                }
                return Ok(dot_q6_k_avx2(encoded, input));
            }
        }

        dot_q6_k_scalar(encoded, input).map_err(|error| shape_mismatch("matvec", error.to_string()))
    }

    fn fused_k_dot(
        &self,
        dtype: DType,
        encoded: &[u8],
        input: &[f32],
    ) -> Result<f32, BackendError> {
        match dtype {
            DType::Q4K => self.fused_q4k_dot(encoded, input),
            DType::Q5K => self.fused_q5k_dot(encoded, input),
            DType::Q6K => self.fused_q6k_dot(encoded, input),
            _ => Err(invalid_parameter(
                "matvec",
                "fused K-quantized dot received a non K-quantized tensor",
            )),
        }
    }

    fn matmul_row(&self, row: &[f32], input: &[f32], output: &mut [f32], columns: usize) {
        if self.simd_enabled && self.features.avx2 && columns >= 8 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if self.features.fma {
                    matmul_row_avx2_fma(row, input, output, columns);
                } else {
                    matmul_row_avx2(row, input, output, columns);
                }
                return;
            }
        }
        matmul_row_scalar(row, input, output, columns);
    }
}

fn dot_scalar(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs)
        .map(|(weight, value)| weight * value)
        .sum()
}

fn matmul_row_scalar(row: &[f32], input: &[f32], output: &mut [f32], columns: usize) {
    for column in 0..columns {
        let mut sum = 0.0_f32;
        for inner in 0..row.len() {
            sum += row[inner] * input[inner * columns + column];
        }
        output[column] = sum;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(lhs: &[f32], rhs: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let zero = _mm256_setzero_ps();
        let (mut acc0, mut acc1, mut acc2, mut acc3) = (zero, zero, zero, zero);
        let mut index = 0;
        while index + 32 <= lhs.len() {
            let a0 = _mm256_loadu_ps(lhs.as_ptr().add(index));
            let b0 = _mm256_loadu_ps(rhs.as_ptr().add(index));
            let a1 = _mm256_loadu_ps(lhs.as_ptr().add(index + 8));
            let b1 = _mm256_loadu_ps(rhs.as_ptr().add(index + 8));
            let a2 = _mm256_loadu_ps(lhs.as_ptr().add(index + 16));
            let b2 = _mm256_loadu_ps(rhs.as_ptr().add(index + 16));
            let a3 = _mm256_loadu_ps(lhs.as_ptr().add(index + 24));
            let b3 = _mm256_loadu_ps(rhs.as_ptr().add(index + 24));
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(a0, b0));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(a1, b1));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(a2, b2));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(a3, b3));
            index += 32;
        }
        while index + 8 <= lhs.len() {
            let a = _mm256_loadu_ps(lhs.as_ptr().add(index));
            let b = _mm256_loadu_ps(rhs.as_ptr().add(index));
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(a, b));
            index += 8;
        }

        let sum01 = _mm256_add_ps(acc0, acc1);
        let sum23 = _mm256_add_ps(acc2, acc3);
        let sum = _mm256_add_ps(sum01, sum23);
        let mut lanes = [0.0_f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
        let mut total = lanes.iter().sum::<f32>();
        while index < lhs.len() {
            total += lhs[index] * rhs[index];
            index += 1;
        }
        total
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2_fma(lhs: &[f32], rhs: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let zero = _mm256_setzero_ps();
        let (mut acc0, mut acc1, mut acc2, mut acc3) = (zero, zero, zero, zero);
        let mut index = 0;
        while index + 32 <= lhs.len() {
            let a0 = _mm256_loadu_ps(lhs.as_ptr().add(index));
            let b0 = _mm256_loadu_ps(rhs.as_ptr().add(index));
            let a1 = _mm256_loadu_ps(lhs.as_ptr().add(index + 8));
            let b1 = _mm256_loadu_ps(rhs.as_ptr().add(index + 8));
            let a2 = _mm256_loadu_ps(lhs.as_ptr().add(index + 16));
            let b2 = _mm256_loadu_ps(rhs.as_ptr().add(index + 16));
            let a3 = _mm256_loadu_ps(lhs.as_ptr().add(index + 24));
            let b3 = _mm256_loadu_ps(rhs.as_ptr().add(index + 24));
            acc0 = _mm256_fmadd_ps(a0, b0, acc0);
            acc1 = _mm256_fmadd_ps(a1, b1, acc1);
            acc2 = _mm256_fmadd_ps(a2, b2, acc2);
            acc3 = _mm256_fmadd_ps(a3, b3, acc3);
            index += 32;
        }
        while index + 8 <= lhs.len() {
            let a = _mm256_loadu_ps(lhs.as_ptr().add(index));
            let b = _mm256_loadu_ps(rhs.as_ptr().add(index));
            acc0 = _mm256_fmadd_ps(a, b, acc0);
            index += 8;
        }

        let sum01 = _mm256_add_ps(acc0, acc1);
        let sum23 = _mm256_add_ps(acc2, acc3);
        let sum = _mm256_add_ps(sum01, sum23);
        let mut lanes = [0.0_f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
        let mut total = lanes.iter().sum::<f32>();
        while index < lhs.len() {
            total += lhs[index] * rhs[index];
            index += 1;
        }
        total
    }
}

fn q4k_read_half(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = u32::from(bits & 0x03ff);
    let output = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let mut normalized = mantissa;
            let mut unbiased_exponent = -14_i32;
            while normalized & 0x0400 == 0 {
                normalized <<= 1;
                unbiased_exponent -= 1;
            }
            normalized &= 0x03ff;
            sign | (((unbiased_exponent + 127) as u32) << 23) | (normalized << 13)
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((u32::from(exponent) + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(output)
}

fn q4k_scale_min(index: usize, packed: &[u8]) -> (u8, u8) {
    if index < 4 {
        (packed[index] & 0x3f, packed[index + 4] & 0x3f)
    } else {
        (
            (packed[index + 4] & 0x0f) | ((packed[index - 4] >> 6) << 4),
            (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4),
        )
    }
}

/// One GGML Q8_K activation block, quantized once and reused for every output row.
#[derive(Clone)]
struct Q8KBlock {
    d: f32,
    qs: [i8; 256],
    bsums: [i16; 16],
}

fn quantize_q8_k(input: &[f32]) -> Result<Vec<Q8KBlock>, BackendError> {
    if input.is_empty() || input.len() % 256 != 0 {
        return Err(shape_mismatch(
            "matvec",
            format!(
                "Q8_K input length {} must be a non-zero multiple of 256",
                input.len()
            ),
        ));
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(invalid_parameter(
            "matvec",
            "Q8_K input values must all be finite",
        ));
    }

    let block_count = input.len() / 256;
    let mut blocks = Vec::new();
    blocks
        .try_reserve_exact(block_count)
        .map_err(|_| BackendError::AllocationFailed {
            operation: "matvec_q8k",
            elements: block_count,
        })?;

    for values in input.chunks_exact(256) {
        let max_abs = values
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f32, f32::max);
        let d = max_abs / 127.0;
        let mut block = Q8KBlock {
            d,
            qs: [0; 256],
            bsums: [0; 16],
        };

        if d > 0.0 {
            for (quantized, value) in block.qs.iter_mut().zip(values) {
                *quantized = (value / d).round().clamp(-127.0, 127.0) as i8;
            }
        }

        for group in 0..16 {
            let start = group * 16;
            block.bsums[group] = block.qs[start..start + 16]
                .iter()
                .map(|value| i16::from(*value))
                .sum();
        }

        blocks.push(block);
    }

    Ok(blocks)
}

fn dot_q4_k_q8_k_scalar(encoded: &[u8], input: &[Q8KBlock]) -> f32 {
    let mut total = 0.0_f32;

    for (block_index, weights) in encoded.chunks_exact(144).enumerate() {
        let activations = &input[block_index];
        let d = q4k_read_half(&weights[..2]);
        let min_d = q4k_read_half(&weights[2..4]);
        let packed_scales = &weights[4..16];
        let quants = &weights[16..144];
        let scaled_d = d * activations.d;
        let scaled_min_d = min_d * activations.d;

        for group in 0..8 {
            let pair = group / 2;
            let high_nibble = group % 2 != 0;
            let (local_scale, local_min) = q4k_scale_min(group, packed_scales);
            let quant_bytes = &quants[pair * 32..pair * 32 + 32];
            let input_offset = group * 32;
            let mut dot = 0_i32;

            for index in 0..32 {
                let packed = quant_bytes[index];
                let quant = if high_nibble {
                    packed >> 4
                } else {
                    packed & 0x0f
                };
                dot += i32::from(quant) * i32::from(activations.qs[input_offset + index]);
            }

            let input_sum = i32::from(activations.bsums[group * 2])
                + i32::from(activations.bsums[group * 2 + 1]);
            total += scaled_d * f32::from(local_scale) * dot as f32
                - scaled_min_d * f32::from(local_min) * input_sum as f32;
        }
    }

    total
}

/// AVX2 Q4_K x Q8_K dot product. Nibbles are unpacked in SIMD registers and
/// accumulated as integer products, avoiding per-element scalar expansion.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q4_k_q8_k_avx2(encoded: &[u8], input: &[Q8KBlock]) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let nibble_mask = _mm256_set1_epi8(0x0f);
        let ones = _mm256_set1_epi16(1);
        let mut total = 0.0_f32;

        for (block_index, weights) in encoded.chunks_exact(144).enumerate() {
            let activations = &input[block_index];
            let d = q4k_read_half(&weights[..2]) * activations.d;
            let min_d = q4k_read_half(&weights[2..4]) * activations.d;
            let packed_scales = &weights[4..16];
            let quants = &weights[16..144];

            for group in 0..8 {
                let pair = group / 2;
                let high_nibble = group % 2 != 0;
                let (local_scale, local_min) = q4k_scale_min(group, packed_scales);
                let packed = _mm256_loadu_si256(
                    quants.as_ptr().add(pair * 32).cast::<__m256i>(),
                );
                let unpacked = if high_nibble {
                    _mm256_and_si256(_mm256_srli_epi16(packed, 4), nibble_mask)
                } else {
                    _mm256_and_si256(packed, nibble_mask)
                };
                let activation_bytes = _mm256_loadu_si256(
                    activations.qs.as_ptr().add(group * 32).cast::<__m256i>(),
                );
                let products = _mm256_maddubs_epi16(unpacked, activation_bytes);
                let pair_sums = _mm256_madd_epi16(products, ones);
                let mut lanes = [0_i32; 8];
                _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), pair_sums);
                let dot = lanes.iter().map(|value| i64::from(*value)).sum::<i64>() as f32;
                let input_sum = i32::from(activations.bsums[group * 2])
                    + i32::from(activations.bsums[group * 2 + 1]);

                total += d * f32::from(local_scale) * dot
                    - min_d * f32::from(local_min) * input_sum as f32;
            }
        }

        total
    }
}

/// AVX2 direct-block Q4_K dot. Quantized values are expanded only in vector
/// registers; no F32 weight row or per-row heap scratch is materialized.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q4_k_avx2(encoded: &[u8], input: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let mut accumulator = _mm256_setzero_ps();
        for (block_index, block) in encoded.chunks_exact(144).enumerate() {
            let d = q4k_read_half(&block[..2]);
            let min_scale = q4k_read_half(&block[2..4]);
            let packed_scales = &block[4..16];
            let quants = &block[16..144];
            let input_base = block_index * 256;

            for group in 0..8 {
                let pair = group / 2;
                let high_nibble = group % 2 != 0;
                let (local_scale, local_min) = q4k_scale_min(group, packed_scales);
                let scale = _mm256_set1_ps(d * f32::from(local_scale));
                let min = _mm256_set1_ps(min_scale * f32::from(local_min));
                let quant_bytes = &quants[pair * 32..pair * 32 + 32];
                let input_offset = input_base + group * 32;

                for offset in (0..32).step_by(8) {
                    let mut values = [0_i32; 8];
                    for lane in 0..8 {
                        let packed = quant_bytes[offset + lane];
                        values[lane] = i32::from(if high_nibble {
                            packed >> 4
                        } else {
                            packed & 0x0f
                        });
                    }
                    let integers = _mm256_loadu_si256(values.as_ptr().cast());
                    let quant = _mm256_cvtepi32_ps(integers);
                    let weights = _mm256_sub_ps(_mm256_mul_ps(quant, scale), min);
                    let x = _mm256_loadu_ps(input.as_ptr().add(input_offset + offset));
                    accumulator = _mm256_add_ps(accumulator, _mm256_mul_ps(weights, x));
                }
            }
        }

        let mut lanes = [0.0_f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator);
        lanes.iter().sum()
    }
}

/// AVX2/FMA direct-block Q4_K dot with fused multiply-add accumulation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q4_k_avx2_fma(encoded: &[u8], input: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let mut accumulator = _mm256_setzero_ps();
        for (block_index, block) in encoded.chunks_exact(144).enumerate() {
            let d = q4k_read_half(&block[..2]);
            let min_scale = q4k_read_half(&block[2..4]);
            let packed_scales = &block[4..16];
            let quants = &block[16..144];
            let input_base = block_index * 256;

            for group in 0..8 {
                let pair = group / 2;
                let high_nibble = group % 2 != 0;
                let (local_scale, local_min) = q4k_scale_min(group, packed_scales);
                let scale = _mm256_set1_ps(d * f32::from(local_scale));
                let min = _mm256_set1_ps(min_scale * f32::from(local_min));
                let quant_bytes = &quants[pair * 32..pair * 32 + 32];
                let input_offset = input_base + group * 32;

                for offset in (0..32).step_by(8) {
                    let mut values = [0_i32; 8];
                    for lane in 0..8 {
                        let packed = quant_bytes[offset + lane];
                        values[lane] = i32::from(if high_nibble {
                            packed >> 4
                        } else {
                            packed & 0x0f
                        });
                    }
                    let integers = _mm256_loadu_si256(values.as_ptr().cast());
                    let quant = _mm256_cvtepi32_ps(integers);
                    let weights = _mm256_sub_ps(_mm256_mul_ps(quant, scale), min);
                    let x = _mm256_loadu_ps(input.as_ptr().add(input_offset + offset));
                    accumulator = _mm256_fmadd_ps(weights, x, accumulator);
                }
            }
        }

        let mut lanes = [0.0_f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator);
        lanes.iter().sum()
    }
}

/// AVX2 direct-block Q5_K dot. Five-bit values are unpacked into
/// temporary integer lanes; no dequantized F32 row is materialized.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q5_k_avx2(encoded: &[u8], input: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let mut accumulator = _mm256_setzero_ps();
        for (block_index, block) in encoded.chunks_exact(176).enumerate() {
            let d = q4k_read_half(&block[..2]);
            let min_scale = q4k_read_half(&block[2..4]);
            let packed_scales = &block[4..16];
            let high_bits = &block[16..48];
            let low_bits = &block[48..176];
            let input_base = block_index * 256;

            for group in 0..8 {
                let pair = group / 2;
                let half = group % 2;
                let mask = 1_u8 << (pair * 2 + half);
                let (local_scale, local_min) = q4k_scale_min(group, packed_scales);
                let scale = _mm256_set1_ps(d * f32::from(local_scale));
                let min = _mm256_set1_ps(min_scale * f32::from(local_min));
                let quant_offset = pair * 32;
                let input_offset = input_base + group * 32;

                for offset in (0..32).step_by(8) {
                    let mut values = [0_i32; 8];
                    for lane in 0..8 {
                        let index = offset + lane;
                        let packed = low_bits[quant_offset + index];
                        let low = if half == 0 {
                            packed & 0x0f
                        } else {
                            packed >> 4
                        };
                        let high = if high_bits[index] & mask != 0 {
                            1_i32
                        } else {
                            0_i32
                        };
                        values[lane] = i32::from(low) | (high << 4);
                    }
                    let integers = _mm256_loadu_si256(values.as_ptr().cast());
                    let quant = _mm256_cvtepi32_ps(integers);
                    let weights = _mm256_sub_ps(_mm256_mul_ps(quant, scale), min);
                    let x = _mm256_loadu_ps(input.as_ptr().add(input_offset + offset));
                    accumulator = _mm256_add_ps(accumulator, _mm256_mul_ps(weights, x));
                }
            }
        }

        let mut lanes = [0.0_f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator);
        lanes.iter().sum()
    }
}

/// AVX2/FMA direct-block Q5_K dot. The packed values are expanded in
/// temporary integer lanes, then accumulated with fused multiply-add.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q5_k_avx2_fma(encoded: &[u8], input: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let mut accumulator = _mm256_setzero_ps();
        for (block_index, block) in encoded.chunks_exact(176).enumerate() {
            let d = q4k_read_half(&block[..2]);
            let min_scale = q4k_read_half(&block[2..4]);
            let packed_scales = &block[4..16];
            let high_bits = &block[16..48];
            let low_bits = &block[48..176];
            let input_base = block_index * 256;

            for group in 0..8 {
                let pair = group / 2;
                let half = group % 2;
                let mask = 1_u8 << (pair * 2 + half);
                let (local_scale, local_min) = q4k_scale_min(group, packed_scales);
                let scale = _mm256_set1_ps(d * f32::from(local_scale));
                let min = _mm256_set1_ps(min_scale * f32::from(local_min));
                let quant_offset = pair * 32;
                let input_offset = input_base + group * 32;

                for offset in (0..32).step_by(8) {
                    let mut values = [0_i32; 8];
                    for lane in 0..8 {
                        let index = offset + lane;
                        let packed = low_bits[quant_offset + index];
                        let low = if half == 0 {
                            packed & 0x0f
                        } else {
                            packed >> 4
                        };
                        let high = if high_bits[index] & mask != 0 {
                            1_i32
                        } else {
                            0_i32
                        };
                        values[lane] = i32::from(low) | (high << 4);
                    }
                    let integers = _mm256_loadu_si256(values.as_ptr().cast());
                    let quant = _mm256_cvtepi32_ps(integers);
                    let weights = _mm256_sub_ps(_mm256_mul_ps(quant, scale), min);
                    let x = _mm256_loadu_ps(input.as_ptr().add(input_offset + offset));
                    accumulator = _mm256_fmadd_ps(weights, x, accumulator);
                }
            }
        }

        let mut lanes = [0.0_f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator);
        lanes.iter().sum()
    }
}

/// AVX2 direct-block Q6_K dot. Signed six-bit values and per-16-value
/// scale groups are expanded in vector registers only.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q6_k_avx2(encoded: &[u8], input: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let mut accumulator = _mm256_setzero_ps();
        for (block_index, block) in encoded.chunks_exact(210).enumerate() {
            let low_bits = &block[..128];
            let high_bits = &block[128..192];
            let scales = &block[192..208];
            let d = q4k_read_half(&block[208..210]);
            let input_base = block_index * 256;

            for half_block in 0..2 {
                let low_offset = half_block * 64;
                let high_offset = half_block * 32;
                let scale_offset = half_block * 8;
                let output_base = input_base + half_block * 128;

                for group in 0..4 {
                    for offset in (0..32).step_by(8) {
                        let mut values = [0_i32; 8];
                        for lane in 0..8 {
                            let index = offset + lane;
                            let high = high_bits[high_offset + index];
                            let low_a = low_bits[low_offset + index];
                            let low_b = low_bits[low_offset + index + 32];
                            let quant = match group {
                                0 => i32::from((low_a & 0x0f) | ((high & 0x03) << 4)),
                                1 => i32::from((low_b & 0x0f) | (((high >> 2) & 0x03) << 4)),
                                2 => i32::from((low_a >> 4) | (((high >> 4) & 0x03) << 4)),
                                _ => i32::from((low_b >> 4) | (((high >> 6) & 0x03) << 4)),
                            };
                            values[lane] = quant - 32;
                        }

                        let scale_index = scale_offset + offset / 16 + group * 2;
                        let signed_scale = (scales[scale_index] as i8) as f32;
                        let weight_scale = _mm256_set1_ps(d * signed_scale);
                        let integers = _mm256_loadu_si256(values.as_ptr().cast());
                        let quant = _mm256_cvtepi32_ps(integers);
                        let weights = _mm256_mul_ps(quant, weight_scale);
                        let input_offset = output_base + group * 32 + offset;
                        let x = _mm256_loadu_ps(input.as_ptr().add(input_offset));
                        accumulator = _mm256_add_ps(accumulator, _mm256_mul_ps(weights, x));
                    }
                }
            }
        }

        let mut lanes = [0.0_f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator);
        lanes.iter().sum()
    }
}

/// AVX2/FMA direct-block Q6_K dot with signed six-bit unpacking and
/// fused multiply-add accumulation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q6_k_avx2_fma(encoded: &[u8], input: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let mut accumulator = _mm256_setzero_ps();
        for (block_index, block) in encoded.chunks_exact(210).enumerate() {
            let low_bits = &block[..128];
            let high_bits = &block[128..192];
            let scales = &block[192..208];
            let d = q4k_read_half(&block[208..210]);
            let input_base = block_index * 256;

            for half_block in 0..2 {
                let low_offset = half_block * 64;
                let high_offset = half_block * 32;
                let scale_offset = half_block * 8;
                let output_base = input_base + half_block * 128;

                for group in 0..4 {
                    for offset in (0..32).step_by(8) {
                        let mut values = [0_i32; 8];
                        for lane in 0..8 {
                            let index = offset + lane;
                            let high = high_bits[high_offset + index];
                            let low_a = low_bits[low_offset + index];
                            let low_b = low_bits[low_offset + index + 32];
                            let quant = match group {
                                0 => i32::from((low_a & 0x0f) | ((high & 0x03) << 4)),
                                1 => i32::from((low_b & 0x0f) | (((high >> 2) & 0x03) << 4)),
                                2 => i32::from((low_a >> 4) | (((high >> 4) & 0x03) << 4)),
                                _ => i32::from((low_b >> 4) | (((high >> 6) & 0x03) << 4)),
                            };
                            values[lane] = quant - 32;
                        }

                        let scale_index = scale_offset + offset / 16 + group * 2;
                        let signed_scale = (scales[scale_index] as i8) as f32;
                        let weight_scale = _mm256_set1_ps(d * signed_scale);
                        let integers = _mm256_loadu_si256(values.as_ptr().cast());
                        let quant = _mm256_cvtepi32_ps(integers);
                        let weights = _mm256_mul_ps(quant, weight_scale);
                        let input_offset = output_base + group * 32 + offset;
                        let x = _mm256_loadu_ps(input.as_ptr().add(input_offset));
                        accumulator = _mm256_fmadd_ps(weights, x, accumulator);
                    }
                }
            }
        }

        let mut lanes = [0.0_f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator);
        lanes.iter().sum()
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn matmul_row_avx2(
    row: &[f32],
    input: &[f32],
    output: &mut [f32],
    columns: usize,
) {
    use std::arch::x86_64::*;

    unsafe {
        let vector_columns = columns & !7;
        for column in (0..vector_columns).step_by(8) {
            let mut acc = _mm256_setzero_ps();
            for (inner, weight) in row.iter().enumerate() {
                let values = _mm256_loadu_ps(input.as_ptr().add(inner * columns + column));
                let scale = _mm256_set1_ps(*weight);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(scale, values));
            }
            _mm256_storeu_ps(output.as_mut_ptr().add(column), acc);
        }
        for column in vector_columns..columns {
            let mut sum = 0.0_f32;
            for inner in 0..row.len() {
                sum += row[inner] * input[inner * columns + column];
            }
            output[column] = sum;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_row_avx2_fma(
    row: &[f32],
    input: &[f32],
    output: &mut [f32],
    columns: usize,
) {
    use std::arch::x86_64::*;

    unsafe {
        let vector_columns = columns & !7;
        for column in (0..vector_columns).step_by(8) {
            let mut acc = _mm256_setzero_ps();
            for (inner, weight) in row.iter().enumerate() {
                let values = _mm256_loadu_ps(input.as_ptr().add(inner * columns + column));
                let scale = _mm256_set1_ps(*weight);
                acc = _mm256_fmadd_ps(scale, values, acc);
            }
            _mm256_storeu_ps(output.as_mut_ptr().add(column), acc);
        }
        for column in vector_columns..columns {
            let mut sum = 0.0_f32;
            for inner in 0..row.len() {
                sum += row[inner] * input[inner * columns + column];
            }
            output[column] = sum;
        }
    }
}

fn shape_mismatch(operation: &'static str, details: impl Into<String>) -> BackendError {
    BackendError::ShapeMismatch {
        operation,
        details: details.into(),
    }
}

fn invalid_parameter(operation: &'static str, details: impl Into<String>) -> BackendError {
    BackendError::InvalidParameter {
        operation,
        details: details.into(),
    }
}

fn dtype_id(dtype: DType) -> u32 {
    match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::Q4_0 => 2,
        DType::Q8_0 => 8,
        DType::Q4K => 12,
        DType::Q5K => 13,
        DType::Q6K => 14,
    }
}

/// Validate row-major [rows, columns] storage and return its encoded row size.
fn matrix_row_layout(
    weights: &TensorView<'_>,
    rows: usize,
    columns: usize,
    operation: &'static str,
) -> Result<(u32, u64, usize), BackendError> {
    if rows == 0 || columns == 0 {
        return Err(shape_mismatch(
            operation,
            "matrix dimensions must both be non-zero",
        ));
    }

    if weights.shape != [rows, columns] {
        return Err(shape_mismatch(
            operation,
            format!(
                "weight tensor shape {:?} does not match [{rows}, {columns}]",
                weights.shape
            ),
        ));
    }

    weights
        .element_count()
        .map_err(|error| shape_mismatch(operation, error.to_string()))?;

    let columns_u64 = u64::try_from(columns).map_err(|_| {
        shape_mismatch(
            operation,
            "input feature count cannot be represented as u64",
        )
    })?;
    let raw_type = dtype_id(weights.dtype);
    let row_bytes_u64 = encoded_len(raw_type, &[columns_u64])
        .map_err(|error| shape_mismatch(operation, error.to_string()))?;
    let row_bytes = usize::try_from(row_bytes_u64)
        .map_err(|_| shape_mismatch(operation, "encoded row length exceeds address space"))?;
    let expected_bytes = row_bytes
        .checked_mul(rows)
        .ok_or_else(|| shape_mismatch(operation, "encoded matrix byte length overflowed usize"))?;

    if weights.data.len() != expected_bytes {
        return Err(shape_mismatch(
            operation,
            format!(
                "weight tensor has {} bytes; expected exactly {expected_bytes}",
                weights.data.len()
            ),
        ));
    }

    Ok((raw_type, columns_u64, row_bytes))
}

fn allocate_row_scratch(
    operation: &'static str,
    elements: usize,
) -> Result<Vec<f32>, BackendError> {
    let mut scratch = Vec::new();
    scratch
        .try_reserve_exact(elements)
        .map_err(|_| BackendError::AllocationFailed {
            operation,
            elements,
        })?;
    scratch.resize(elements, 0.0);
    Ok(scratch)
}

fn decode_weight_row(
    operation: &'static str,
    raw_type: u32,
    columns: u64,
    row_bytes: usize,
    row_index: usize,
    weights: &TensorView<'_>,
    scratch: &mut [f32],
) -> Result<(), BackendError> {
    let start = row_index
        .checked_mul(row_bytes)
        .ok_or_else(|| shape_mismatch(operation, "encoded row offset overflowed usize"))?;
    let end = start
        .checked_add(row_bytes)
        .ok_or_else(|| shape_mismatch(operation, "encoded row end overflowed usize"))?;
    let encoded_row = weights.data.get(start..end).ok_or_else(|| {
        shape_mismatch(
            operation,
            "encoded weight row is outside the tensor byte range",
        )
    })?;

    dequantize_into(raw_type, &[columns], encoded_row, scratch)
        .map_err(|error| shape_mismatch(operation, error.to_string()))
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn matvec(
        &self,
        weights: TensorView<'_>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), BackendError> {
        let Some((&out_features, &in_features)) = weights.shape.first().zip(weights.shape.get(1))
        else {
            return Err(shape_mismatch(
                "matvec",
                "weight tensor must be a rank-2 [out_features, in_features] matrix",
            ));
        };

        if input.len() != in_features {
            return Err(shape_mismatch(
                "matvec",
                format!("input has {} elements; expected {in_features}", input.len()),
            ));
        }
        if output.len() != out_features {
            return Err(shape_mismatch(
                "matvec",
                format!(
                    "output has {} elements; expected {out_features}",
                    output.len()
                ),
            ));
        }

        let (raw_type, columns, row_bytes) =
            matrix_row_layout(&weights, out_features, in_features, "matvec")?;

        let workers = self.worker_count.min(out_features);
        let parallel = workers > 1 && out_features >= workers.saturating_mul(2);
        let fused_q4k = self.fused_q4k_matvec && weights.dtype == DType::Q4K;
        // Q8_K activation quantization pays off for the wide hidden dimensions
        // used by Qwen3.5 and is performed once per matvec, not once per row.
        let q8k_input = if fused_q4k && in_features >= 2048 && self.uses_avx2() {
            Some(quantize_q8_k(input)?)
        } else {
            None
        };
        let fused_q5k_q6k =
            self.fused_q5k_q6k_matvec && matches!(weights.dtype, DType::Q5K | DType::Q6K);
        let fused = fused_q4k || fused_q5k_q6k;

        if fused {
            if parallel {
                let chunk_rows = out_features.div_ceil(workers);
                return self.pool.install(|| {
                    output.par_chunks_mut(chunk_rows).enumerate().try_for_each(
                        |(chunk_index, output_chunk)| {
                            let first_row = chunk_index * chunk_rows;
                            for (local_row, output_value) in output_chunk.iter_mut().enumerate() {
                                let row_index = first_row + local_row;
                                let start = row_index.checked_mul(row_bytes).ok_or_else(|| {
                                    shape_mismatch(
                                        "matvec",
                                        "encoded weight row offset overflowed usize",
                                    )
                                })?;
                                let end = start.checked_add(row_bytes).ok_or_else(|| {
                                    shape_mismatch(
                                        "matvec",
                                        "encoded weight row end overflowed usize",
                                    )
                                })?;
                                let encoded_row =
                                    weights.data.get(start..end).ok_or_else(|| {
                                        shape_mismatch(
                                            "matvec",
                                            "encoded weight row is outside the tensor byte range",
                                        )
                                    })?;
                                *output_value = match q8k_input.as_deref() {
                                    Some(quantized_input) => {
                                        self.fused_q4k_q8k_dot(encoded_row, quantized_input)
                                    }
                                    None => self.fused_k_dot(weights.dtype, encoded_row, input)?,
                                };
                            }
                            Ok(())
                        },
                    )
                });
            }

            for (row_index, output_value) in output.iter_mut().enumerate() {
                let start = row_index.checked_mul(row_bytes).ok_or_else(|| {
                    shape_mismatch("matvec", "encoded weight row offset overflowed usize")
                })?;
                let end = start.checked_add(row_bytes).ok_or_else(|| {
                    shape_mismatch("matvec", "encoded weight row end overflowed usize")
                })?;
                let encoded_row = weights.data.get(start..end).ok_or_else(|| {
                    shape_mismatch(
                        "matvec",
                        "encoded weight row is outside the tensor byte range",
                    )
                })?;
                *output_value = match q8k_input.as_deref() {
                    Some(quantized_input) => self.fused_q4k_q8k_dot(encoded_row, quantized_input),
                    None => self.fused_k_dot(weights.dtype, encoded_row, input)?,
                };
            }
            return Ok(());
        }

        if parallel {
            let chunk_rows = out_features.div_ceil(workers);
            let simd_enabled = self.simd_enabled;
            let features = self.features;

            return self.pool.install(|| {
                output
                    .par_chunks_mut(chunk_rows)
                    .enumerate()
                    .try_for_each(|(chunk_index, output_chunk)| {
                        let first_row = chunk_index * chunk_rows;
                        let mut row = allocate_row_scratch("matvec", in_features)?;
                        for (local_row, output_value) in output_chunk.iter_mut().enumerate() {
                            let row_index = first_row + local_row;
                            decode_weight_row(
                                "matvec",
                                raw_type,
                                columns,
                                row_bytes,
                                row_index,
                                &weights,
                                &mut row,
                            )?;
                            *output_value = if simd_enabled && features.avx2 {
                                #[cfg(target_arch = "x86_64")]
                                unsafe {
                                    if features.fma {
                                        dot_avx2_fma(&row, input)
                                    } else {
                                        dot_avx2(&row, input)
                                    }
                                }
                                #[cfg(not(target_arch = "x86_64"))]
                                dot_scalar(&row, input)
                            } else {
                                dot_scalar(&row, input)
                            };
                        }
                        Ok(())
                    })
            });
        }

        let mut row = allocate_row_scratch("matvec", in_features)?;
        for (row_index, output_value) in output.iter_mut().enumerate() {
            decode_weight_row(
                "matvec", raw_type, columns, row_bytes, row_index, &weights, &mut row,
            )?;
            *output_value = self.dot_product(&row, input);
        }
        Ok(())
    }

    fn matvec_batch(
        &self,
        weights: &[TensorView<'_>],
        input: &[f32],
        outputs: &mut [&mut [f32]],
    ) -> Result<(), BackendError> {
        if weights.len() != outputs.len() {
            return Err(shape_mismatch(
                "matvec_batch",
                format!(
                    "{} weight tensors have {} output buffers",
                    weights.len(),
                    outputs.len()
                ),
            ));
        }

        if weights.len() <= 1 || self.worker_count <= 1 {
            for (weight, output) in weights.iter().copied().zip(outputs.iter_mut()) {
                self.matvec(weight, input, *output)?;
            }
            return Ok(());
        }

        // Projection tasks and their row-parallel matvecs share this same
        // persistent pool. Rayon work-stealing prevents a second pool or
        // per-projection OS threads from being created.
        self.pool.install(|| {
            outputs
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(index, output)| self.matvec(weights[index], input, *output))
        })
    }

    fn matmul(
        &self,
        weights: TensorView<'_>,
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BackendError> {
        if m == 0 || n == 0 || k == 0 {
            return Err(shape_mismatch(
                "matmul",
                "m, n, and k dimensions must all be non-zero",
            ));
        }
        let input_len = k.checked_mul(n).ok_or_else(|| {
            shape_mismatch("matmul", "input matrix element count overflowed usize")
        })?;
        let output_len = m.checked_mul(n).ok_or_else(|| {
            shape_mismatch("matmul", "output matrix element count overflowed usize")
        })?;

        if input.len() != input_len {
            return Err(shape_mismatch(
                "matmul",
                format!(
                    "B has {} elements; expected {input_len} for [{k}, {n}]",
                    input.len()
                ),
            ));
        }
        if output.len() != output_len {
            return Err(shape_mismatch(
                "matmul",
                format!(
                    "C has {} elements; expected {output_len} for [{m}, {n}]",
                    output.len()
                ),
            ));
        }

        let (raw_type, columns, row_bytes) = matrix_row_layout(&weights, m, k, "matmul")?;
        let workers = self.worker_count.min(m);
        let parallel = workers > 1 && m >= workers.saturating_mul(2);

        if parallel {
            let chunk_rows = m.div_ceil(workers);
            return self.pool.install(|| {
                output
                    .par_chunks_mut(chunk_rows * n)
                    .enumerate()
                    .try_for_each(|(chunk_index, output_chunk)| {
                        let first_row = chunk_index * chunk_rows;
                        let mut row = allocate_row_scratch("matmul", k)?;
                        let rows_in_chunk = output_chunk.len() / n;

                        for local_row in 0..rows_in_chunk {
                            let row_index = first_row + local_row;
                            decode_weight_row(
                                "matmul",
                                raw_type,
                                columns,
                                row_bytes,
                                row_index,
                                &weights,
                                &mut row,
                            )?;
                            let output_start = local_row * n;
                            self.matmul_row(
                                &row,
                                input,
                                &mut output_chunk[output_start..output_start + n],
                                n,
                            );
                        }
                        Ok(())
                    })
            });
        }

        let mut row = allocate_row_scratch("matmul", k)?;
        for row_index in 0..m {
            decode_weight_row(
                "matmul", raw_type, columns, row_bytes, row_index, &weights, &mut row,
            )?;

            let output_start = row_index * n;
            self.matmul_row(
                &row,
                input,
                &mut output[output_start..output_start + n],
                n,
            );
        }
        Ok(())
    }

    fn parallel_for_each_mut(
        &self,
        values: &mut [f32],
        chunk_size: usize,
        operation: &(dyn Fn(usize, &mut [f32]) + Sync),
    ) -> Result<(), BackendError> {
        if chunk_size == 0 || values.len() % chunk_size != 0 {
            return Err(shape_mismatch(
                "parallel_for_each_mut",
                format!(
                    "buffer length {} must be divisible by non-zero chunk size {chunk_size}",
                    values.len()
                ),
            ));
        }

        let chunk_count = values.len() / chunk_size;
        if chunk_count <= 1 || self.worker_count <= 1 {
            for (index, chunk) in values.chunks_exact_mut(chunk_size).enumerate() {
                operation(index, chunk);
            }
            return Ok(());
        }

        self.pool.install(|| {
            values
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(index, chunk)| operation(index, chunk));
        });
        Ok(())
    }

    fn rms_norm(
        &self,
        input: &[f32],
        weight: &[f32],
        output: &mut [f32],
        epsilon: f32,
    ) -> Result<(), BackendError> {
        if input.is_empty() || weight.len() != input.len() || output.len() != input.len() {
            return Err(shape_mismatch(
                "rms_norm",
                format!(
                    "input, weight, and output must have the same non-zero length (got {}, {}, {})",
                    input.len(),
                    weight.len(),
                    output.len()
                ),
            ));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(invalid_parameter(
                "rms_norm",
                "epsilon must be finite and greater than zero",
            ));
        }

        let square_sum = input
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>();
        let mean_square = square_sum / input.len() as f64;
        let inverse_rms = (mean_square + f64::from(epsilon)).sqrt().recip();

        for ((output_value, input_value), weight_value) in output.iter_mut().zip(input).zip(weight)
        {
            *output_value =
                (f64::from(*input_value) * inverse_rms * f64::from(*weight_value)) as f32;
        }

        Ok(())
    }

    fn rope(
        &self,
        input: &mut [f32],
        position: usize,
        head_dim: usize,
        theta: f32,
    ) -> Result<(), BackendError> {
        if head_dim == 0 || head_dim % 2 != 0 || input.len() != head_dim {
            return Err(shape_mismatch(
                "rope",
                format!(
                    "input length ({}) must equal a non-zero even head_dim ({head_dim})",
                    input.len()
                ),
            ));
        }
        if !theta.is_finite() || theta <= 0.0 {
            return Err(invalid_parameter(
                "rope",
                "theta must be finite and greater than zero",
            ));
        }

        let half_dim = head_dim / 2;
        for index in 0..half_dim {
            let exponent = -((2 * index) as f64) / head_dim as f64;
            let frequency = f64::from(theta).powf(exponent);
            let angle = position as f64 * frequency;
            let cosine = angle.cos() as f32;
            let sine = angle.sin() as f32;

            let first = input[index];
            let second = input[index + half_dim];
            input[index] = first * cosine - second * sine;
            input[index + half_dim] = first * sine + second * cosine;
        }

        Ok(())
    }

    fn softmax(&self, input: &mut [f32]) -> Result<(), BackendError> {
        if input.is_empty() {
            return Err(shape_mismatch(
                "softmax",
                "input must contain at least one element",
            ));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(invalid_parameter(
                "softmax",
                "input values must all be finite",
            ));
        }

        let maximum = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exponential_sum = 0.0_f64;
        for value in input.iter_mut() {
            *value = (*value - maximum).exp();
            exponential_sum += f64::from(*value);
        }

        for value in input {
            *value = (f64::from(*value) / exponential_sum) as f32;
        }

        Ok(())
    }

    fn silu_mul(&self, gate: &mut [f32], up: &[f32]) -> Result<(), BackendError> {
        if gate.len() != up.len() {
            return Err(shape_mismatch(
                "silu_mul",
                format!(
                    "gate has {} elements; up has {} elements",
                    gate.len(),
                    up.len()
                ),
            ));
        }

        for (gate_value, up_value) in gate.iter_mut().zip(up) {
            let value = *gate_value;
            let sigmoid = if value >= 0.0 {
                1.0 / (1.0 + (-value).exp())
            } else {
                let exponential = value.exp();
                exponential / (1.0 + exponential)
            };
            *gate_value = value * sigmoid * up_value;
        }

        Ok(())
    }

    fn residual_add(&self, target: &mut [f32], residual: &[f32]) -> Result<(), BackendError> {
        if target.len() != residual.len() {
            return Err(shape_mismatch(
                "residual_add",
                format!(
                    "target has {} elements; residual has {} elements",
                    target.len(),
                    residual.len()
                ),
            ));
        }

        for (target_value, residual_value) in target.iter_mut().zip(residual) {
            *target_value += residual_value;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend() -> CpuBackend {
        CpuBackend::new(1).expect("one worker is valid")
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn parallel_for_each_mut_processes_disjoint_chunks() {
        let backend = CpuBackend::new(4).expect("four workers");
        let mut values = vec![0.0_f32; 32];

        backend
            .parallel_for_each_mut(&mut values, 8, &|index, chunk| {
                chunk.fill(index as f32);
            })
            .unwrap();

        let expected = (0..32)
            .map(|index| (index / 8) as f32)
            .collect::<Vec<_>>();
        assert_eq!(values, expected);
    }

    #[test]
    fn parallel_for_each_mut_rejects_invalid_chunk_shapes() {
        let backend = backend();
        let mut values = vec![0.0_f32; 8];

        for chunk_size in [0, 3] {
            assert!(matches!(
                backend.parallel_for_each_mut(&mut values, chunk_size, &|_, _| {}),
                Err(BackendError::ShapeMismatch {
                    operation: "parallel_for_each_mut",
                    ..
                })
            ));
        }
    }

    #[test]
    fn rejects_zero_workers() {
        assert_eq!(
            CpuBackend::new(0).unwrap_err(),
            CpuBackendError::ZeroWorkers
        );
    }

    #[test]
    fn stores_worker_count() {
        let backend = CpuBackend::new(4).expect("non-zero worker count");
        assert_eq!(backend.worker_count(), 4);
        assert_eq!(backend.name(), "cpu");
    }

    #[test]
    fn scalar_backend_disables_simd_dispatch() {
        let scalar = CpuBackend::new_scalar(1).expect("scalar backend");
        let automatic = CpuBackend::new(1).expect("automatic backend");
        assert!(!scalar.uses_avx2());
        assert_eq!(
            automatic.uses_avx2(),
            cfg!(target_arch = "x86_64") && automatic.features().avx2
        );
    }

    #[test]
    fn fused_q5k_q6k_matvec_matches_scalar_and_simd_dequant_references() {
        let shape = [4, 512];
        let input = (0..512)
            .map(|index| ((index * 11 % 73) as f32 - 36.0) / 41.0)
            .collect::<Vec<_>>();
        let scalar = CpuBackend::new_scalar(1).unwrap();
        let simd_dequant = CpuBackend::new(1).unwrap();
        let fused = CpuBackend::new_fused_qk(1).unwrap();
        // Four output rows and two workers exercise the parallel fused-row path.
        let parallel_fused = CpuBackend::new_fused_qk(2).unwrap();

        let mut q5_k = vec![0_u8; 176 * 8];
        let half_scales = [0x3c00_u16, 0x3800, 0x4000, 0x3a00];
        let half_mins = [0x3400_u16, 0x3800, 0x3a00, 0x3000];
        for (block_index, block) in q5_k.chunks_exact_mut(176).enumerate() {
            block[..2].copy_from_slice(&half_scales[block_index % 4].to_le_bytes());
            block[2..4].copy_from_slice(&half_mins[block_index % 4].to_le_bytes());
            for (index, byte) in block[4..16].iter_mut().enumerate() {
                *byte = (index * 19 + block_index * 31) as u8;
            }
            for (index, byte) in block[16..48].iter_mut().enumerate() {
                *byte = (index * 23 + block_index * 17) as u8;
            }
            for (index, byte) in block[48..].iter_mut().enumerate() {
                *byte = (index * 13 + block_index * 29) as u8;
            }
        }

        let mut q6_k = vec![0_u8; 210 * 8];
        for (block_index, block) in q6_k.chunks_exact_mut(210).enumerate() {
            for (index, byte) in block[..128].iter_mut().enumerate() {
                *byte = (index * 7 + block_index * 13) as u8;
            }
            for (index, byte) in block[128..192].iter_mut().enumerate() {
                *byte = (index * 11 + block_index * 19) as u8;
            }
            for (index, byte) in block[192..208].iter_mut().enumerate() {
                *byte = (index as i8 * 7 - 48) as u8;
            }
            block[208..210].copy_from_slice(&half_scales[block_index % 4].to_le_bytes());
        }

        for (dtype, bytes) in [(DType::Q5K, q5_k), (DType::Q6K, q6_k)] {
            let weights = TensorView::new(&bytes, &shape, dtype);
            let mut expected = [0.0_f32; 4];
            let mut dequant_output = [0.0_f32; 4];
            let mut fused_output = [0.0_f32; 4];
            let mut parallel_output = [0.0_f32; 4];

            scalar.matvec(weights, &input, &mut expected).unwrap();
            simd_dequant
                .matvec(weights, &input, &mut dequant_output)
                .unwrap();
            fused.matvec(weights, &input, &mut fused_output).unwrap();
            parallel_fused
                .matvec(weights, &input, &mut parallel_output)
                .unwrap();

            for row in 0..4 {
                let tolerance = 1.0e-3 * (1.0 + expected[row].abs());
                assert!(
                    (dequant_output[row] - expected[row]).abs() <= tolerance,
                    "SIMD/dequant {:?} mismatch at row {row}: expected={}, actual={}",
                    dtype,
                    expected[row],
                    dequant_output[row]
                );
                assert!(
                    (fused_output[row] - expected[row]).abs() <= tolerance,
                    "fused {:?} mismatch at row {row}: expected={}, actual={}",
                    dtype,
                    expected[row],
                    fused_output[row]
                );
                assert!(
                    (parallel_output[row] - expected[row]).abs() <= tolerance,
                    "parallel fused {:?} mismatch at row {row}: expected={}, actual={}",
                    dtype,
                    expected[row],
                    parallel_output[row]
                );
            }
        }
    }

    #[test]
    fn fused_q4_k_matvec_matches_scalar_and_simd_dequant_references() {
        let shape = [2, 512];
        let mut bytes = vec![0_u8; 144 * 4];
        let d_bits = [0x3c00_u16, 0x3800, 0x4000, 0x3a00];
        let min_bits = [0x3400_u16, 0x3800, 0x3a00, 0x3000];

        for (block_index, block) in bytes.chunks_exact_mut(144).enumerate() {
            block[..2].copy_from_slice(&d_bits[block_index].to_le_bytes());
            block[2..4].copy_from_slice(&min_bits[block_index].to_le_bytes());
            for (index, byte) in block[4..16].iter_mut().enumerate() {
                *byte = (index * 19 + block_index * 31) as u8;
            }
            for (index, byte) in block[16..].iter_mut().enumerate() {
                *byte = (index * 13 + block_index * 29) as u8;
            }
        }

        let weights = TensorView::new(&bytes, &shape, DType::Q4K);
        let input = (0..512)
            .map(|index| ((index * 7 % 41) as f32 - 20.0) / 23.0)
            .collect::<Vec<_>>();
        let scalar = CpuBackend::new_scalar(1).unwrap();
        let simd_dequant = CpuBackend::new(1).unwrap();
        let fused = CpuBackend::new_fused_q4k(1).unwrap();
        let mut expected = [0.0_f32; 2];
        let mut dequant_output = [0.0_f32; 2];
        let mut fused_output = [0.0_f32; 2];

        scalar.matvec(weights, &input, &mut expected).unwrap();
        simd_dequant
            .matvec(weights, &input, &mut dequant_output)
            .unwrap();
        fused.matvec(weights, &input, &mut fused_output).unwrap();
        let parallel_fused = CpuBackend::new_fused_q4k(4).unwrap();
        let mut parallel_output = [0.0_f32; 2];
        parallel_fused
            .matvec(weights, &input, &mut parallel_output)
            .unwrap();

        for row in 0..2 {
            let tolerance = 1.0e-3 * (1.0 + expected[row].abs());
            assert!(
                (dequant_output[row] - expected[row]).abs() <= tolerance,
                "SIMD/dequant mismatch at row {row}: expected={}, actual={}",
                expected[row],
                dequant_output[row]
            );
            assert!(
                (fused_output[row] - expected[row]).abs() <= tolerance,
                "fused Q4_K mismatch at row {row}: expected={}, actual={}",
                expected[row],
                fused_output[row]
            );
            assert!(
                (parallel_output[row] - expected[row]).abs() <= tolerance,
                "parallel fused Q4_K mismatch at row {row}: expected={}, actual={}",
                expected[row],
                parallel_output[row]
            );
        }
    }

    #[test]
    fn q4k_q8k_matvec_matches_f32_reference_with_bounded_error() {
        let shape = [4, 2048];
        let mut bytes = vec![0_u8; 144 * 8 * shape[0]];

        for (block_index, block) in bytes.chunks_exact_mut(144).enumerate() {
            block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
            block[2..4].copy_from_slice(&0x3400_u16.to_le_bytes());
            for (index, byte) in block[4..16].iter_mut().enumerate() {
                *byte = (index * 19 + block_index * 7) as u8;
            }
            for (index, byte) in block[16..].iter_mut().enumerate() {
                *byte = (index * 29 + block_index * 11) as u8;
            }
        }

        let weights = TensorView::new(&bytes, &shape, DType::Q4K);
        let input = (0..2048)
            .map(|index| ((index * 13 % 97) as f32 - 48.0) / 49.0)
            .collect::<Vec<_>>();
        let reference = CpuBackend::new_scalar(1).unwrap();
        let optimized = CpuBackend::new_fused_q4k(4).unwrap();
        let mut expected = [0.0_f32; 4];
        let mut actual = [0.0_f32; 4];

        reference.matvec(weights, &input, &mut expected).unwrap();
        optimized.matvec(weights, &input, &mut actual).unwrap();

        for row in 0..4 {
            let tolerance = 3.0e-2 * (1.0 + expected[row].abs());
            assert!(
                (actual[row] - expected[row]).abs() <= tolerance,
                "Q4_K x Q8_K mismatch at row {row}: expected={}, actual={}",
                expected[row],
                actual[row]
            );
        }
    }

    #[test]
    fn parallel_fused_q4k_matvec_matches_scalar_reference() {
        let shape = [4, 512];
        let mut bytes = vec![0_u8; 144 * 8];
        for (block_index, block) in bytes.chunks_exact_mut(144).enumerate() {
            block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
            block[2..4].copy_from_slice(&0x3400_u16.to_le_bytes());
            for (index, byte) in block[4..16].iter_mut().enumerate() {
                *byte = (index * 17 + block_index * 23) as u8;
            }
            for (index, byte) in block[16..].iter_mut().enumerate() {
                *byte = (index * 29 + block_index * 31) as u8;
            }
        }

        let weights = TensorView::new(&bytes, &shape, DType::Q4K);
        let input = (0..512)
            .map(|index| ((index * 13 % 67) as f32 - 33.0) / 37.0)
            .collect::<Vec<_>>();
        let scalar = CpuBackend::new_scalar(1).unwrap();
        // Four rows and two workers enter the parallel fused-row path.
        let fused = CpuBackend::new_fused_q4k(2).unwrap();
        let mut expected = [0.0_f32; 4];
        let mut actual = [0.0_f32; 4];

        scalar.matvec(weights, &input, &mut expected).unwrap();
        fused.matvec(weights, &input, &mut actual).unwrap();

        for row in 0..4 {
            let tolerance = 1.0e-3 * (1.0 + expected[row].abs());
            assert!(
                (actual[row] - expected[row]).abs() <= tolerance,
                "parallel fused Q4_K mismatch at row {row}: expected={}, actual={}",
                expected[row],
                actual[row]
            );
        }
    }

    #[test]
    fn avx2_matvec_and_matmul_match_scalar_reference() {
        let (m, k, n) = (17, 65, 13);
        let weight_values = (0..m * k)
            .map(|index| ((index % 37) as f32 - 18.0) / 23.0)
            .collect::<Vec<_>>();
        let weight_bytes = f32_bytes(&weight_values);
        let shape = [m, k];
        let weights = TensorView::new(&weight_bytes, &shape, DType::F32);
        let vector = (0..k)
            .map(|index| ((index % 19) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let matrix = (0..k * n)
            .map(|index| ((index % 29) as f32 - 14.0) / 31.0)
            .collect::<Vec<_>>();

        let scalar = CpuBackend::new_scalar(1).expect("scalar backend");
        let automatic = CpuBackend::new(4).expect("automatic backend");
        let cloned = automatic.clone();
        assert_eq!(automatic.pool.current_num_threads(), 4);
        assert!(std::sync::Arc::ptr_eq(&automatic.pool, &cloned.pool));

        let mut scalar_vector_output = vec![0.0; m];
        let mut automatic_vector_output = vec![0.0; m];
        scalar
            .matvec(weights, &vector, &mut scalar_vector_output)
            .unwrap();
        automatic
            .matvec(weights, &vector, &mut automatic_vector_output)
            .unwrap();

        for (expected, actual) in scalar_vector_output
            .iter()
            .zip(&automatic_vector_output)
        {
            assert!(
                (expected - actual).abs() <= 1.0e-3 * (1.0 + expected.abs()),
                "matvec mismatch: scalar={expected}, automatic={actual}"
            );
        }

        let mut scalar_matrix_output = vec![0.0; m * n];
        let mut automatic_matrix_output = vec![0.0; m * n];
        scalar
            .matmul(weights, &matrix, &mut scalar_matrix_output, m, n, k)
            .unwrap();
        automatic
            .matmul(weights, &matrix, &mut automatic_matrix_output, m, n, k)
            .unwrap();

        for (expected, actual) in scalar_matrix_output
            .iter()
            .zip(&automatic_matrix_output)
        {
            assert!(
                (expected - actual).abs() <= 1.0e-3 * (1.0 + expected.abs()),
                "matmul mismatch: scalar={expected}, automatic={actual}"
            );
        }
    }

    #[test]
    fn matvec_batch_matches_individual_projections() {
        let backend = CpuBackend::new(4).unwrap();
        let input = [0.5_f32, -1.0, 2.0];
        let bytes_a = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let bytes_b = f32_bytes(&[2.0, 0.0, -1.0, 3.0, 1.0, 2.0, -2.0, 1.0, 4.0]);
        let weights = [
            TensorView::new(&bytes_a, &[2, 3], DType::F32),
            TensorView::new(&bytes_b, &[3, 3], DType::F32),
        ];
        let mut first = [0.0_f32; 2];
        let mut second = [0.0_f32; 3];
        let mut outputs: [&mut [f32]; 2] = [&mut first, &mut second];

        backend.matvec_batch(&weights, &input, &mut outputs).unwrap();

        assert_eq!(first, [4.5, 9.0]);
        assert_eq!(second, [-1.0, 4.5, 6.0]);
    }

    #[test]
    fn matvec_batch_rejects_weight_output_count_mismatch() {
        let backend = CpuBackend::new(2).unwrap();
        let bytes = f32_bytes(&[1.0, 2.0]);
        let weight = TensorView::new(&bytes, &[1, 2], DType::F32);
        let weights = [weight, weight];
        let mut output = [0.0_f32; 1];
        let mut outputs: [&mut [f32]; 1] = [&mut output];

        assert!(matches!(
            backend.matvec_batch(&weights, &[1.0, 1.0], &mut outputs),
            Err(BackendError::ShapeMismatch {
                operation: "matvec_batch",
                ..
            })
        ));
    }

    #[test]
    fn computes_f32_matvec() {
        let bytes = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let shape = [2, 3];
        let weights = TensorView::new(&bytes, &shape, DType::F32);
        let input = [1.0, 2.0, 3.0];
        let mut output = [0.0; 2];

        backend().matvec(weights, &input, &mut output).unwrap();

        assert_eq!(output, [14.0, 32.0]);
    }

    #[test]
    fn computes_quantized_q4_0_matvec_row_by_row() {
        let mut bytes = vec![0_u8; 36];
        bytes[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        bytes[2..18].fill(0x98);
        bytes[18..20].copy_from_slice(&0x3800_u16.to_le_bytes());
        bytes[20..].fill(0x89);
        let shape = [2, 32];
        let weights = TensorView::new(&bytes, &shape, DType::Q4_0);
        let input = [1.0; 32];
        let mut output = [0.0; 2];

        backend().matvec(weights, &input, &mut output).unwrap();

        assert_eq!(output, [16.0, 8.0]);
    }

    #[test]
    fn computes_f16_and_q8_0_matvec() {
        let f16_bytes = [
            0x00_u8, 0x3c, // 1.0
            0x00, 0xc0, // -2.0
            0x00, 0x40, // 2.0
            0x00, 0x42, // 3.0
        ];
        let shape = [2, 2];
        let weights = TensorView::new(&f16_bytes, &shape, DType::F16);
        let mut output = [0.0; 2];

        backend().matvec(weights, &[2.0, 1.0], &mut output).unwrap();

        assert_eq!(output, [0.0, 7.0]);

        let mut q8_bytes = vec![0_u8; 68];
        q8_bytes[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
        q8_bytes[2..34].fill(2);
        q8_bytes[34..36].copy_from_slice(&0x3800_u16.to_le_bytes());
        q8_bytes[36..68].fill((-2_i8) as u8);
        let shape = [2, 32];
        let weights = TensorView::new(&q8_bytes, &shape, DType::Q8_0);
        let mut output = [0.0; 2];

        backend().matvec(weights, &[1.0; 32], &mut output).unwrap();

        assert_eq!(output, [32.0, -32.0]);
    }

    #[test]
    fn computes_k_quant_matvec_row_by_row() {
        let input = [1.0; 256];
        let shape = [2, 256];

        let mut q4_k = vec![0_u8; 144 * 2];
        for (row, scale, quant) in [(0, 0x3c00_u16, 0x11), (1, 0x3c00_u16, 0x22)] {
            let base = row * 144;
            q4_k[base..base + 2].copy_from_slice(&scale.to_le_bytes());
            q4_k[base + 4] = if row == 0 { 1 } else { 2 };
            q4_k[base + 5] = if row == 0 { 1 } else { 2 };
            q4_k[base + 16..base + 48].fill(quant);
        }
        let weights = TensorView::new(&q4_k, &shape, DType::Q4K);
        let mut output = [0.0; 2];
        backend().matvec(weights, &input, &mut output).unwrap();
        assert_eq!(output, [64.0, 256.0]);

        let mut q5_k = vec![0_u8; 176 * 2];
        for row in 0..2 {
            let base = row * 176;
            q5_k[base..base + 2].copy_from_slice(&0x3c00_u16.to_le_bytes());
            q5_k[base + 4] = if row == 0 { 1 } else { 2 };
            q5_k[base + 5] = if row == 0 { 1 } else { 2 };
            q5_k[base + 48..base + 80].fill(if row == 0 { 0x11 } else { 0x22 });
        }
        let weights = TensorView::new(&q5_k, &shape, DType::Q5K);
        let mut output = [0.0; 2];
        backend().matvec(weights, &input, &mut output).unwrap();
        assert_eq!(output, [64.0, 256.0]);

        let mut q6_k = vec![0_u8; 210 * 2];
        for row in 0..2 {
            let base = row * 210;
            q6_k[base..base + 32].fill(if row == 0 { 0x01 } else { 0x02 });
            q6_k[base + 128..base + 192].fill(0xaa);
            q6_k[base + 192..base + 208].fill(1);
            q6_k[base + 208..base + 210].copy_from_slice(&0x3c00_u16.to_le_bytes());
        }
        let weights = TensorView::new(&q6_k, &shape, DType::Q6K);
        let mut output = [0.0; 2];
        backend().matvec(weights, &input, &mut output).unwrap();
        assert_eq!(output, [32.0, 64.0]);
    }

    #[test]
    fn computes_quantized_q4_0_matmul() {
        let mut bytes = vec![0_u8; 36];
        bytes[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        bytes[2..18].fill(0x98);
        bytes[18..20].copy_from_slice(&0x3800_u16.to_le_bytes());
        bytes[20..36].fill(0x89);
        let shape = [2, 32];
        let weights = TensorView::new(&bytes, &shape, DType::Q4_0);
        let input = [1.0; 64];
        let mut output = [0.0; 4];

        backend()
            .matmul(weights, &input, &mut output, 2, 2, 32)
            .unwrap();

        assert_eq!(output, [16.0, 16.0, 8.0, 8.0]);
    }

    #[test]
    fn computes_row_major_f32_matmul() {
        let bytes = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let shape = [2, 2];
        let weights = TensorView::new(&bytes, &shape, DType::F32);
        let input = [5.0, 6.0, 7.0, 8.0];
        let mut output = [0.0; 4];

        backend()
            .matmul(weights, &input, &mut output, 2, 2, 2)
            .unwrap();

        assert_eq!(output, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn rejects_invalid_matrix_shapes_and_storage_lengths() {
        let bytes = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let shape = [2, 2];
        let weights = TensorView::new(&bytes, &shape, DType::F32);
        let mut output = [0.0; 2];

        assert!(matches!(
            backend().matvec(weights, &[1.0], &mut output),
            Err(BackendError::ShapeMismatch {
                operation: "matvec",
                ..
            })
        ));

        let short_weights = TensorView::new(&bytes[..8], &shape, DType::F32);
        assert!(matches!(
            backend().matvec(short_weights, &[1.0, 1.0], &mut output),
            Err(BackendError::ShapeMismatch {
                operation: "matvec",
                ..
            })
        ));
    }

    #[test]
    fn computes_rms_norm() {
        let input = [3.0, 4.0];
        let weight = [1.0, 1.0];
        let mut output = [0.0; 2];

        backend()
            .rms_norm(&input, &weight, &mut output, 1.0e-5)
            .unwrap();

        assert!((output[0] - 0.848_528).abs() < 1.0e-5);
        assert!((output[1] - 1.131_370_8).abs() < 1.0e-5);
    }

    #[test]
    fn rejects_invalid_rms_norm_epsilon() {
        let input = [1.0, 2.0];
        let mut output = [0.0; 2];

        assert!(matches!(
            backend().rms_norm(&input, &[1.0, 1.0], &mut output, 0.0),
            Err(BackendError::InvalidParameter {
                operation: "rms_norm",
                ..
            })
        ));
    }

    #[test]
    fn applies_llama_split_half_rope() {
        let mut input = [1.0, 2.0, 3.0, 4.0];

        backend().rope(&mut input, 1, 4, 10_000.0).unwrap();

        let expected_first = 1.0_f32 * 1.0_f32.cos() - 3.0 * 1.0_f32.sin();
        let expected_third = 1.0_f32 * 1.0_f32.sin() + 3.0 * 1.0_f32.cos();
        assert!((input[0] - expected_first).abs() < 1.0e-6);
        assert!((input[2] - expected_third).abs() < 1.0e-6);
        let expected_second = 2.0_f32 * 0.01_f32.cos() - 4.0 * 0.01_f32.sin();
        let expected_fourth = 2.0_f32 * 0.01_f32.sin() + 4.0 * 0.01_f32.cos();
        assert!((input[1] - expected_second).abs() < 1.0e-6);
        assert!((input[3] - expected_fourth).abs() < 1.0e-6);
    }

    #[test]
    fn computes_stable_softmax() {
        let mut input = [1000.0, 1001.0];

        backend().softmax(&mut input).unwrap();

        assert!((input[0] - 0.268_941_43).abs() < 1.0e-6);
        assert!((input[1] - 0.731_058_6).abs() < 1.0e-6);
        assert!((input.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn rejects_non_finite_softmax_input() {
        let mut input = [0.0, f32::INFINITY];

        assert!(matches!(
            backend().softmax(&mut input),
            Err(BackendError::InvalidParameter {
                operation: "softmax",
                ..
            })
        ));
    }

    #[test]
    fn computes_silu_times_up_with_extreme_inputs() {
        let mut gate = [0.0, 1.0, -1.0, -100.0, 100.0];
        let up = [2.0, 2.0, 2.0, 1.0, 1.0];

        backend().silu_mul(&mut gate, &up).unwrap();

        assert_eq!(gate[0], 0.0);
        assert!((gate[1] - 1.462_117_2).abs() < 1.0e-6);
        assert!((gate[2] - (-0.537_882_86)).abs() < 1.0e-6);
        assert!(gate[3].is_finite());
        assert!((gate[4] - 100.0).abs() < 1.0e-4);
    }

    #[test]
    fn adds_residual_in_place() {
        let mut target = [1.0, 2.0, 3.0];

        backend()
            .residual_add(&mut target, &[0.5, -1.0, 2.0])
            .unwrap();

        assert_eq!(target, [1.5, 1.0, 5.0]);
    }

    #[test]
    fn rejects_residual_shape_mismatch() {
        let mut target = [1.0, 2.0];

        assert!(matches!(
            backend().residual_add(&mut target, &[1.0]),
            Err(BackendError::ShapeMismatch {
                operation: "residual_add",
                ..
            })
        ));
    }
}
