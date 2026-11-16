//! Shared types and contracts for Kernforge inference components.

pub mod backend;
pub mod dtype;
pub mod error;
pub mod tensor;

pub use backend::{Backend, BackendError};
pub use dtype::DType;
pub use error::TensorError;
pub use tensor::TensorView;
