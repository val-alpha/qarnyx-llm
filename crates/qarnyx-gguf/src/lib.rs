//! GGUF v3 parsing and validated tensor-directory access.
//!
//! The parser streams the header, metadata, and tensor descriptors from any
//! seekable reader. Tensor weight payloads are not copied into memory. Exact
//! per-tensor byte-range validation will be added with the GGML quantization
//! type registry so block-encoded formats are sized correctly.

mod parser;

pub use parser::{
    GgufError, GgufFile, MetadataEntry, MetadataValue, MetadataValueType, ParseOptions,
    TensorInfo,
};
