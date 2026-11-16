//! CPU execution backend.
//!
//! The first implementation will be a scalar reference backend. SIMD kernels
//! must be selected through runtime feature detection and tested against the
//! scalar implementation.

use kernforge_core::{Backend, BackendError, TensorView};
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
/// Kernel execution and the persistent worker pool are not implemented yet.
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

    fn not_implemented<T>(operation: &'static str) -> Result<T, BackendError> {
        Err(BackendError::NotImplemented { operation })
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn matvec(
        &self,
        _weights: TensorView<'_>,
        _input: &[f32],
        _output: &mut [f32],
    ) -> Result<(), BackendError> {
        Self::not_implemented("matvec")
    }

    fn matmul(
        &self,
        _weights: TensorView<'_>,
        _input: &[f32],
        _output: &mut [f32],
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<(), BackendError> {
        Self::not_implemented("matmul")
    }

    fn rms_norm(
        &self,
        _input: &[f32],
        _weight: &[f32],
        _output: &mut [f32],
        _epsilon: f32,
    ) -> Result<(), BackendError> {
        Self::not_implemented("rms_norm")
    }

    fn rope(
        &self,
        _input: &mut [f32],
        _position: usize,
        _head_dim: usize,
        _theta: f32,
    ) -> Result<(), BackendError> {
        Self::not_implemented("rope")
    }

    fn softmax(&self, _input: &mut [f32]) -> Result<(), BackendError> {
        Self::not_implemented("softmax")
    }

    fn silu_mul(&self, _gate: &mut [f32], _up: &[f32]) -> Result<(), BackendError> {
        Self::not_implemented("silu_mul")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_workers() {
        assert_eq!(CpuBackend::new(0).unwrap_err(), CpuBackendError::ZeroWorkers);
    }

    #[test]
    fn stores_worker_count() {
        let backend = CpuBackend::new(4).expect("non-zero worker count");
        assert_eq!(backend.worker_count(), 4);
        assert_eq!(backend.name(), "cpu");
    }
}
