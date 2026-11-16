//! GGUF file parsing and validated tensor access.
//!
//! Planned responsibilities:
//! - parse the GGUF header and metadata;
//! - parse tensor descriptors and alignment;
//! - validate all offsets and lengths against the file size;
//! - expose immutable tensor views over validated mapped bytes.
//!
//! Parsing is not implemented yet. Do not use this crate to load untrusted
//! model files until bounds checks and malformed-input tests are complete.
