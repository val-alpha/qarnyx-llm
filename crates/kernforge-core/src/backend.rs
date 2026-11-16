use thiserror::Error;

use crate::TensorView;

/// Errors returned by a tensor execution backend.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum BackendError {
    /// The requested kernel has not been implemented by this backend.
    #[error("backend kernel is not implemented: {operation}")]
    NotImplemented {
        /// Operation name.
        operation: &'static str,
    },

    /// A tensor or buffer does not have the expected dimensions.
    #[error("shape mismatch in {operation}: {details}")]
    ShapeMismatch {
        /// Operation name.
        operation: &'static str,
        /// Human-readable shape details.
        details: String,
    },
}

/// Hardware-independent tensor operations used by model implementations.
///
/// Logical matrix layouts and buffer dimensions must be documented and
/// validated by each implementation. Quantized weights should be consumed
/// directly by suitable kernels rather than expanded into a full F32 copy.
pub trait Backend: Send + Sync {
    /// Returns a stable backend name.
    fn name(&self) -> &'static str;

    /// Computes y = W * x for row-major logical weights shaped
    /// [out_features, in_features].
    fn matvec(
        &self,
        weights: TensorView<'_>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), BackendError>;

    /// Computes C = A * B where A is [m, k], B is [k, n], and C is [m, n].
    fn matmul(
        &self,
        weights: TensorView<'_>,
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BackendError>;

    /// Applies RMS normalization to input using the provided scale weights.
    fn rms_norm(
        &self,
        input: &[f32],
        weight: &[f32],
        output: &mut [f32],
        epsilon: f32,
    ) -> Result<(), BackendError>;

    /// Applies rotary positional embedding to one packed head vector.
    fn rope(
        &self,
        input: &mut [f32],
        position: usize,
        head_dim: usize,
        theta: f32,
    ) -> Result<(), BackendError>;

    /// Applies softmax in place.
    fn softmax(&self, input: &mut [f32]) -> Result<(), BackendError>;

    /// Computes gate = SiLU(gate) * up in place.
    fn silu_mul(&self, gate: &mut [f32], up: &[f32]) -> Result<(), BackendError>;
}
