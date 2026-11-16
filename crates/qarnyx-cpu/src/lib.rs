//! Scalar CPU execution backend.
//!
//! Matrix kernels decode one weight row at a time into a reusable scratch
//! buffer. This supports quantized GGUF weights without expanding a complete
//! matrix or model into F32 memory. SIMD dispatch and worker scheduling are
//! intentionally deferred until these scalar paths are established as the
//! correctness reference.

use qarnyx_core::{Backend, BackendError, DType, TensorView};
use qarnyx_quant::{dequantize_into, encoded_len};
use thiserror::Error;

/// Errors raised while constructing the CPU backend.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CpuBackendError {
    /// Worker count must be non-zero.
    #[error("CPU backend requires at least one worker")]
    ZeroWorkers,
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
/// The first kernels are scalar and single-threaded. The worker count and
/// detected features are retained for later scheduling and SIMD dispatch.
#[derive(Debug, Clone)]
pub struct CpuBackend {
    worker_count: usize,
    features: CpuFeatures,
}

impl CpuBackend {
    /// Creates a CPU backend configuration with a fixed worker count.
    pub fn new(worker_count: usize) -> Result<Self, CpuBackendError> {
        if worker_count == 0 {
            return Err(CpuBackendError::ZeroWorkers);
        }

        Ok(Self {
            worker_count,
            features: CpuFeatures::detect(),
        })
    }

    /// Returns the configured number of workers.
    pub const fn worker_count(&self) -> usize {
        self.worker_count
    }

    /// Returns detected CPU features.
    pub const fn features(&self) -> CpuFeatures {
        self.features
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
        let mut row = allocate_row_scratch("matvec", in_features)?;

        for (row_index, output_value) in output.iter_mut().enumerate() {
            decode_weight_row(
                "matvec", raw_type, columns, row_bytes, row_index, &weights, &mut row,
            )?;
            *output_value = row
                .iter()
                .zip(input)
                .map(|(weight, value)| weight * value)
                .sum();
        }

        Ok(())
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
        let mut row = allocate_row_scratch("matmul", k)?;

        for row_index in 0..m {
            decode_weight_row(
                "matmul", raw_type, columns, row_bytes, row_index, &weights, &mut row,
            )?;

            for column in 0..n {
                let mut sum = 0.0_f32;
                for inner in 0..k {
                    sum += row[inner] * input[inner * n + column];
                }
                output[row_index * n + column] = sum;
            }
        }

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
