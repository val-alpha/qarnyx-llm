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

    /// A scalar argument is outside the supported domain.
    #[error("invalid parameter in {operation}: {details}")]
    InvalidParameter {
        /// Operation name.
        operation: &'static str,
        /// Human-readable parameter details.
        details: String,
    },

    /// A temporary kernel buffer could not be reserved.
    #[error("allocation failed in {operation} for {elements} elements")]
    AllocationFailed {
        /// Operation name.
        operation: &'static str,
        /// Requested number of elements.
        elements: usize,
    },
}

/// Hardware-independent tensor operations used by model implementations.
///
/// Logical matrix layouts and buffer dimensions are documented and validated
/// by each operation. Quantized weights should be consumed directly by
/// suitable kernels rather than expanded into a full F32 copy.
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
    /// A is supplied as a row-major tensor view; B and C are row-major slices.
    fn matmul(
        &self,
        weights: TensorView<'_>,
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BackendError>;

    /// Applies an operation to each disjoint, fixed-size mutable chunk.
    ///
    /// The default implementation is sequential. Backends with a persistent
    /// scheduler may execute chunks concurrently. The operation must only mutate
    /// the chunk passed to it; chunks never overlap.
    fn parallel_for_each_mut(
        &self,
        values: &mut [f32],
        chunk_size: usize,
        operation: &(dyn Fn(usize, &mut [f32]) + Sync),
    ) -> Result<(), BackendError> {
        if chunk_size == 0 || values.len() % chunk_size != 0 {
            return Err(BackendError::ShapeMismatch {
                operation: "parallel_for_each_mut",
                details: format!(
                    "buffer length {} must be divisible by non-zero chunk size {chunk_size}",
                    values.len()
                ),
            });
        }

        for (index, chunk) in values.chunks_exact_mut(chunk_size).enumerate() {
            operation(index, chunk);
        }
        Ok(())
    }

    /// Applies RMS normalization to input using the provided scale weights.
    fn rms_norm(
        &self,
        input: &[f32],
        weight: &[f32],
        output: &mut [f32],
        epsilon: f32,
    ) -> Result<(), BackendError>;

    /// Applies LLaMA-style split-half rotary embedding to one head vector.
    ///
    /// The first and second halves form the rotation pairs. `input.len()`
    /// must equal `head_dim`, which must be even.
    fn rope(
        &self,
        input: &mut [f32],
        position: usize,
        head_dim: usize,
        theta: f32,
    ) -> Result<(), BackendError>;

    /// Applies numerically stable softmax in place.
    fn softmax(&self, input: &mut [f32]) -> Result<(), BackendError>;

    /// Computes gate = SiLU(gate) * up in place.
    fn silu_mul(&self, gate: &mut [f32], up: &[f32]) -> Result<(), BackendError>;

    /// Adds a residual slice to target in place.
    fn residual_add(&self, target: &mut [f32], residual: &[f32]) -> Result<(), BackendError>;
}
