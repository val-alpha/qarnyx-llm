use crate::{DType, TensorError};

/// A non-owning view over tensor bytes.
///
/// The view does not allocate, own, or dequantize the underlying data.
/// Callers constructing views from files must validate offsets, byte lengths,
/// quantization block sizes, and alignment before creating the view.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    /// Encoded tensor bytes.
    pub data: &'a [u8],
    /// Logical tensor dimensions.
    pub shape: &'a [usize],
    /// Physical storage type.
    pub dtype: DType,
}

impl<'a> TensorView<'a> {
    /// Creates a view over already-validated tensor bytes.
    ///
    /// This constructor intentionally does not infer the required byte length:
    /// quantized formats use block encodings and need type-specific validation.
    pub const fn new(data: &'a [u8], shape: &'a [usize], dtype: DType) -> Self {
        Self { data, shape, dtype }
    }

    /// Returns the number of logical elements in the tensor.
    pub fn element_count(&self) -> Result<usize, TensorError> {
        if self.shape.is_empty() {
            return Err(TensorError::EmptyShape);
        }

        self.shape.iter().try_fold(1usize, |count, &dimension| {
            if dimension == 0 {
                return Err(TensorError::ZeroDimension);
            }

            count
                .checked_mul(dimension)
                .ok_or(TensorError::ElementCountOverflow)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_element_count() {
        let bytes = [0_u8; 24];
        let shape = [2, 3, 4];
        let view = TensorView::new(&bytes, &shape, DType::F32);

        assert_eq!(view.element_count(), Ok(24));
    }

    #[test]
    fn rejects_empty_shape() {
        let bytes = [0_u8; 1];
        let shape: [usize; 0] = [];
        let view = TensorView::new(&bytes, &shape, DType::F32);

        assert_eq!(view.element_count(), Err(TensorError::EmptyShape));
    }

    #[test]
    fn rejects_zero_dimension() {
        let bytes = [0_u8; 1];
        let shape = [2, 0, 4];
        let view = TensorView::new(&bytes, &shape, DType::F32);

        assert_eq!(view.element_count(), Err(TensorError::ZeroDimension));
    }
}
