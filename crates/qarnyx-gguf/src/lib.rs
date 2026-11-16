//! GGUF v3 parsing and validated tensor-directory access.
//!
//! The parser streams the header, metadata, and tensor descriptors from any
//! seekable reader. It validates exact per-tensor byte ranges using the
//! qarnyx-quant GGML type registry without copying payload bytes into memory.

mod parser;

pub use parser::{
    GgufError, GgufFile, MetadataEntry, MetadataValue, MetadataValueType, ParseOptions, TensorInfo,
};
