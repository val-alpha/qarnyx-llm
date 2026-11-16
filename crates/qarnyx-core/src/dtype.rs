/// Data types supported by the initial GGUF runtime design.
///
/// This enum describes storage formats, not necessarily the arithmetic
/// accumulator type used by a kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    Q4_0,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
}
