use thiserror::Error;

/// Errors raised while validating tensor metadata or views.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TensorError {
    /// The tensor shape contains a zero-sized dimension.
    #[error("tensor shape contains a zero-sized dimension")]
    ZeroDimension,

    /// Multiplying tensor dimensions overflowed usize.
    #[error("tensor element count overflowed usize")]
    ElementCountOverflow,

    /// The tensor has no dimensions.
    #[error("tensor shape must contain at least one dimension")]
    EmptyShape,
}
