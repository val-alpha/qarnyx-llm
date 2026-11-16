//! Defensive, streaming reader for the GGUF v3 container format.
//!
//! The parser reads the header, metadata, and tensor directory. It deliberately
//! does not read tensor payloads or infer their encoded byte lengths; those
//! checks belong with the quantization block registry.

use std::collections::HashSet;
use std::fmt;
use std::io::{self, Cursor, Read, Seek, SeekFrom};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: u32 = 32;
const MAX_TENSOR_DIMS: u32 = 4;
const MAX_METADATA_KEY_BYTES: u64 = u16::MAX as u64;

/// Bounds applied while parsing untrusted metadata and tensor descriptors.
#[derive(Debug, Clone)]
pub struct ParseOptions {
    pub max_tensor_count: u64,
    pub max_metadata_count: u64,
    pub max_metadata_bytes: u64,
    pub max_string_bytes: u64,
    pub max_array_elements: u64,
    pub max_array_depth: u32,
    pub max_alignment: u32,
}

impl Default for ParseOptions {
    fn default() -> Self {
        Self {
            max_tensor_count: 1_000_000,
            max_metadata_count: 1_000_000,
            max_metadata_bytes: 512 * 1024 * 1024,
            max_string_bytes: 16 * 1024 * 1024,
            max_array_elements: 4_000_000,
            max_array_depth: 8,
            max_alignment: 1024 * 1024,
        }
    }
}

/// Errors produced while reading or validating a GGUF file.
#[derive(Debug)]
pub enum GgufError {
    Io(io::Error),
    Truncated {
        offset: u64,
        context: &'static str,
    },
    InvalidMagic([u8; 4]),
    UnsupportedVersion(u32),
    BigEndianUnsupported,
    InvalidMetadataType(u32),
    InvalidBool(u8),
    InvalidUtf8(&'static str),
    InvalidMetadataKey(String),
    DuplicateMetadataKey(String),
    DuplicateTensorName(String),
    EmptyTensorName,
    DuplicateTensorOffset(u64),
    InvalidStringLength {
        length: u64,
        maximum: u64,
        context: &'static str,
    },
    LimitExceeded {
        kind: &'static str,
        actual: u64,
        limit: u64,
    },
    MetadataTooLarge {
        actual: u64,
        limit: u64,
    },
    ArrayDepthExceeded {
        limit: u32,
    },
    InvalidDimensionCount(u32),
    ZeroDimension(String),
    ShapeOverflow(String),
    InvalidAlignment(u32),
    InvalidAlignmentMetadataType,
    MisalignedTensorOffset {
        tensor: String,
        offset: u64,
        alignment: u32,
    },
    TensorDataOutOfBounds {
        tensor: String,
        absolute_offset: u64,
        file_size: u64,
    },
    NonZeroPadding {
        offset: u64,
    },
    Overflow(&'static str),
    AllocationFailed(&'static str),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Truncated { offset, context } => {
                write!(f, "truncated GGUF while reading {context} at byte {offset}")
            }
            Self::InvalidMagic(found) => {
                write!(f, "invalid GGUF magic: expected GGUF, found {found:?}")
            }
            Self::UnsupportedVersion(version) => {
                write!(
                    f,
                    "unsupported GGUF version {version}; this reader supports v3"
                )
            }
            Self::BigEndianUnsupported => {
                write!(f, "big-endian GGUF is not supported by this reader yet")
            }
            Self::InvalidMetadataType(value) => {
                write!(f, "invalid GGUF metadata value type {value}")
            }
            Self::InvalidBool(value) => {
                write!(f, "invalid GGUF boolean value {value}; expected 0 or 1")
            }
            Self::InvalidUtf8(context) => write!(f, "{context} is not valid UTF-8"),
            Self::InvalidMetadataKey(key) => {
                write!(
                    f,
                    "invalid metadata key {key}; keys must be non-empty ASCII"
                )
            }
            Self::DuplicateMetadataKey(key) => {
                write!(f, "duplicate GGUF metadata key {key}")
            }
            Self::DuplicateTensorName(name) => {
                write!(f, "duplicate GGUF tensor name {name}")
            }
            Self::EmptyTensorName => write!(f, "GGUF tensor name must not be empty"),
            Self::DuplicateTensorOffset(offset) => {
                write!(f, "multiple GGUF tensors use data offset {offset}")
            }
            Self::InvalidStringLength {
                length,
                maximum,
                context,
            } => write!(
                f,
                "{context} length {length} exceeds the configured maximum {maximum}"
            ),
            Self::LimitExceeded {
                kind,
                actual,
                limit,
            } => write!(
                f,
                "{kind} count {actual} exceeds the configured limit {limit}"
            ),
            Self::MetadataTooLarge { actual, limit } => write!(
                f,
                "metadata region is {actual} bytes, exceeding the configured limit {limit}"
            ),
            Self::ArrayDepthExceeded { limit } => {
                write!(
                    f,
                    "metadata array nesting exceeds the configured depth {limit}"
                )
            }
            Self::InvalidDimensionCount(count) => {
                write!(
                    f,
                    "invalid tensor dimension count {count}; expected 1..={MAX_TENSOR_DIMS}"
                )
            }
            Self::ZeroDimension(name) => {
                write!(f, "tensor {name} contains a zero-sized dimension")
            }
            Self::ShapeOverflow(name) => {
                write!(f, "element count overflows for tensor {name}")
            }
            Self::InvalidAlignment(alignment) => write!(
                f,
                "invalid GGUF alignment {alignment}; it must be a multiple of 8 and within the configured limit"
            ),
            Self::InvalidAlignmentMetadataType => {
                write!(f, "general.alignment must have metadata type UINT32")
            }
            Self::MisalignedTensorOffset {
                tensor,
                offset,
                alignment,
            } => write!(
                f,
                "tensor {tensor} has offset {offset}, which is not aligned to {alignment} bytes"
            ),
            Self::TensorDataOutOfBounds {
                tensor,
                absolute_offset,
                file_size,
            } => write!(
                f,
                "tensor {tensor} begins at byte {absolute_offset}, outside file size {file_size}"
            ),
            Self::NonZeroPadding { offset } => {
                write!(
                    f,
                    "GGUF alignment padding contains a non-zero byte at {offset}"
                )
            }
            Self::Overflow(context) => write!(f, "integer overflow while computing {context}"),
            Self::AllocationFailed(context) => {
                write!(f, "unable to reserve memory for {context}")
            }
        }
    }
}

impl std::error::Error for GgufError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for GgufError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

/// The scalar types defined by the GGUF metadata specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MetadataValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl MetadataValueType {
    fn from_raw(raw: u32) -> Result<Self, GgufError> {
        match raw {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            other => Err(GgufError::InvalidMetadataType(other)),
        }
    }
}

/// A decoded GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array {
        element_type: MetadataValueType,
        values: Vec<MetadataValue>,
    },
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl MetadataValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Uint32(value) => Some(*value),
            _ => None,
        }
    }
}

/// A metadata key/value pair, kept in file order.
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataEntry {
    pub key: String,
    pub value: MetadataValue,
}

/// Describes a tensor without loading its payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    /// Raw ggml_type numeric identifier. Decoding is handled by qarnyx-quant.
    pub ggml_type: u32,
    /// Offset relative to the start of the tensor data section.
    pub offset: u64,
    /// Absolute byte offset in the GGUF file.
    pub file_offset: u64,
}

impl TensorInfo {
    pub fn element_count(&self) -> Option<u64> {
        self.dimensions
            .iter()
            .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
    }
}

/// Parsed GGUF header, metadata, and tensor directory.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufFile {
    pub version: u32,
    pub metadata: Vec<MetadataEntry>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: u32,
    pub tensor_data_offset: u64,
    pub file_size: u64,
}

impl GgufFile {
    /// Parse a GGUF file from an in-memory byte slice.
    pub fn parse(data: &[u8]) -> Result<Self, GgufError> {
        Self::parse_with_options(data, &ParseOptions::default())
    }

    /// Parse a GGUF byte slice with explicit resource limits.
    pub fn parse_with_options(data: &[u8], options: &ParseOptions) -> Result<Self, GgufError> {
        let mut cursor = Cursor::new(data);
        Self::parse_reader_with_options(&mut cursor, options)
    }

    /// Parse a GGUF stream without reading the tensor payload into memory.
    ///
    /// The reader must support seeking so the parser can determine the file
    /// length and validate absolute tensor data offsets.
    pub fn parse_reader<R: Read + Seek>(reader: &mut R) -> Result<Self, GgufError> {
        Self::parse_reader_with_options(reader, &ParseOptions::default())
    }

    /// Parse a GGUF stream with explicit resource limits.
    pub fn parse_reader_with_options<R: Read + Seek>(
        reader: &mut R,
        options: &ParseOptions,
    ) -> Result<Self, GgufError> {
        let file_size = reader.seek(SeekFrom::End(0))?;
        reader.seek(SeekFrom::Start(0))?;

        let mut magic = [0_u8; 4];
        read_exact(reader, file_size, &mut magic, "magic")?;
        if &magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic(magic));
        }

        let mut version_bytes = [0_u8; 4];
        read_exact(reader, file_size, &mut version_bytes, "version")?;
        let version = u32::from_le_bytes(version_bytes);
        if version != GGUF_VERSION {
            if u32::from_be_bytes(version_bytes) == GGUF_VERSION {
                return Err(GgufError::BigEndianUnsupported);
            }
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = read_u64(reader, file_size, "tensor count")?;
        let metadata_count = read_u64(reader, file_size, "metadata count")?;
        check_limit("tensor", tensor_count, options.max_tensor_count)?;
        check_limit("metadata", metadata_count, options.max_metadata_count)?;

        ensure_minimum_remaining(reader, file_size, metadata_count, 13, "metadata entries")?;
        let metadata_start = reader.stream_position()?;
        let metadata_capacity = to_usize(metadata_count, "metadata count")?;
        let mut metadata = Vec::new();
        metadata
            .try_reserve_exact(metadata_capacity)
            .map_err(|_| GgufError::AllocationFailed("metadata entries"))?;
        let mut metadata_keys = HashSet::new();
        metadata_keys
            .try_reserve(metadata_capacity)
            .map_err(|_| GgufError::AllocationFailed("metadata key index"))?;

        for _ in 0..metadata_count {
            let key = read_string(
                reader,
                file_size,
                options.max_string_bytes.min(MAX_METADATA_KEY_BYTES),
                "metadata key",
            )?;
            if key.is_empty() || !key.is_ascii() {
                return Err(GgufError::InvalidMetadataKey(key));
            }
            if !metadata_keys.insert(key.clone()) {
                return Err(GgufError::DuplicateMetadataKey(key));
            }

            let raw_type = read_u32(reader, file_size, "metadata value type")?;
            let value_type = MetadataValueType::from_raw(raw_type)?;
            let value = read_metadata_value(reader, file_size, options, value_type, 0)?;
            metadata.push(MetadataEntry { key, value });

            let consumed = reader
                .stream_position()?
                .checked_sub(metadata_start)
                .ok_or(GgufError::Overflow("metadata byte count"))?;
            if consumed > options.max_metadata_bytes {
                return Err(GgufError::MetadataTooLarge {
                    actual: consumed,
                    limit: options.max_metadata_bytes,
                });
            }
        }

        let alignment = match metadata
            .iter()
            .find(|entry| entry.key == "general.alignment")
        {
            Some(entry) => match &entry.value {
                MetadataValue::Uint32(value) => *value,
                _ => return Err(GgufError::InvalidAlignmentMetadataType),
            },
            None => DEFAULT_ALIGNMENT,
        };
        if alignment == 0 || alignment % 8 != 0 || alignment > options.max_alignment {
            return Err(GgufError::InvalidAlignment(alignment));
        }

        ensure_minimum_remaining(reader, file_size, tensor_count, 32, "tensor descriptors")?;
        let tensor_capacity = to_usize(tensor_count, "tensor count")?;
        let mut tensors = Vec::new();
        tensors
            .try_reserve_exact(tensor_capacity)
            .map_err(|_| GgufError::AllocationFailed("tensor descriptors"))?;
        let mut tensor_names = HashSet::new();
        tensor_names
            .try_reserve(tensor_capacity)
            .map_err(|_| GgufError::AllocationFailed("tensor name index"))?;

        for _ in 0..tensor_count {
            let name = read_string(reader, file_size, 64, "tensor name")?;
            if name.is_empty() {
                return Err(GgufError::EmptyTensorName);
            }
            if !tensor_names.insert(name.clone()) {
                return Err(GgufError::DuplicateTensorName(name));
            }

            let dimension_count = read_u32(reader, file_size, "tensor dimension count")?;
            if dimension_count == 0 || dimension_count > MAX_TENSOR_DIMS {
                return Err(GgufError::InvalidDimensionCount(dimension_count));
            }

            let dimension_capacity = dimension_count as usize;
            let mut dimensions = Vec::new();
            dimensions
                .try_reserve_exact(dimension_capacity)
                .map_err(|_| GgufError::AllocationFailed("tensor dimensions"))?;

            let mut element_count = 1_u64;
            for _ in 0..dimension_count {
                let dimension = read_u64(reader, file_size, "tensor dimension")?;
                if dimension == 0 {
                    return Err(GgufError::ZeroDimension(name));
                }
                element_count = element_count
                    .checked_mul(dimension)
                    .ok_or_else(|| GgufError::ShapeOverflow(name.clone()))?;
                dimensions.push(dimension);
            }
            let _ = element_count;

            let ggml_type = read_u32(reader, file_size, "tensor type")?;
            let offset = read_u64(reader, file_size, "tensor data offset")?;
            if offset % u64::from(alignment) != 0 {
                return Err(GgufError::MisalignedTensorOffset {
                    tensor: name,
                    offset,
                    alignment,
                });
            }

            tensors.push(TensorInfo {
                name,
                dimensions,
                ggml_type,
                offset,
                file_offset: 0,
            });
        }

        let directory_end = reader.stream_position()?;
        let alignment_u64 = u64::from(alignment);
        let padding = (alignment_u64 - (directory_end % alignment_u64)) % alignment_u64;
        let tensor_data_offset = directory_end
            .checked_add(padding)
            .ok_or(GgufError::Overflow("tensor data section offset"))?;
        if tensor_data_offset > file_size {
            return Err(GgufError::Truncated {
                offset: directory_end,
                context: "tensor data alignment padding",
            });
        }

        validate_zero_padding(reader, file_size, padding)?;
        let data_bytes = file_size - tensor_data_offset;
        let mut tensor_offsets = HashSet::new();
        tensor_offsets
            .try_reserve(tensor_capacity)
            .map_err(|_| GgufError::AllocationFailed("tensor offset index"))?;

        for tensor in &mut tensors {
            if !tensor_offsets.insert(tensor.offset) {
                return Err(GgufError::DuplicateTensorOffset(tensor.offset));
            }

            let relative_offset = tensor.offset;
            let absolute_offset = tensor_data_offset
                .checked_add(relative_offset)
                .ok_or(GgufError::Overflow("absolute tensor offset"))?;
            if relative_offset >= data_bytes {
                return Err(GgufError::TensorDataOutOfBounds {
                    tensor: tensor.name.clone(),
                    absolute_offset,
                    file_size,
                });
            }
            tensor.file_offset = absolute_offset;
        }

        Ok(Self {
            version,
            metadata,
            tensors,
            alignment,
            tensor_data_offset,
            file_size,
        })
    }

    pub fn metadata_value(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata
            .iter()
            .find(|entry| entry.key == key)
            .map(|entry| &entry.value)
    }

    pub fn metadata_string(&self, key: &str) -> Option<&str> {
        self.metadata_value(key).and_then(MetadataValue::as_str)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }
}

fn check_limit(kind: &'static str, actual: u64, limit: u64) -> Result<(), GgufError> {
    if actual > limit {
        return Err(GgufError::LimitExceeded {
            kind,
            actual,
            limit,
        });
    }
    Ok(())
}

fn to_usize(value: u64, context: &'static str) -> Result<usize, GgufError> {
    usize::try_from(value).map_err(|_| GgufError::Overflow(context))
}

fn ensure_minimum_remaining<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
    count: u64,
    minimum_size: u64,
    context: &'static str,
) -> Result<(), GgufError> {
    let offset = reader.stream_position()?;
    let minimum_bytes = count
        .checked_mul(minimum_size)
        .ok_or(GgufError::Overflow("minimum encoded size"))?;
    let end = offset
        .checked_add(minimum_bytes)
        .ok_or(GgufError::Overflow("minimum encoded boundary"))?;
    if end > file_size {
        return Err(GgufError::Truncated { offset, context });
    }
    Ok(())
}

fn minimum_encoded_value_size(value_type: MetadataValueType) -> u64 {
    match value_type {
        MetadataValueType::Uint8 | MetadataValueType::Int8 | MetadataValueType::Bool => 1,
        MetadataValueType::Uint16 | MetadataValueType::Int16 => 2,
        MetadataValueType::Uint32 | MetadataValueType::Int32 | MetadataValueType::Float32 => 4,
        MetadataValueType::String => 8,
        MetadataValueType::Array => 12,
        MetadataValueType::Uint64 | MetadataValueType::Int64 | MetadataValueType::Float64 => 8,
    }
}

fn read_exact<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
    output: &mut [u8],
    context: &'static str,
) -> Result<(), GgufError> {
    let offset = reader.stream_position()?;
    let length = u64::try_from(output.len()).map_err(|_| GgufError::Overflow("read length"))?;
    let end = offset
        .checked_add(length)
        .ok_or(GgufError::Overflow("read boundary"))?;
    if end > file_size {
        return Err(GgufError::Truncated { offset, context });
    }
    reader.read_exact(output).map_err(|error| {
        if error.kind() == io::ErrorKind::UnexpectedEof {
            GgufError::Truncated { offset, context }
        } else {
            GgufError::Io(error)
        }
    })
}

fn read_u8<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
    context: &'static str,
) -> Result<u8, GgufError> {
    let mut bytes = [0_u8; 1];
    read_exact(reader, file_size, &mut bytes, context)?;
    Ok(bytes[0])
}

fn read_u16<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
    context: &'static str,
) -> Result<u16, GgufError> {
    let mut bytes = [0_u8; 2];
    read_exact(reader, file_size, &mut bytes, context)?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
    context: &'static str,
) -> Result<u32, GgufError> {
    let mut bytes = [0_u8; 4];
    read_exact(reader, file_size, &mut bytes, context)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
    context: &'static str,
) -> Result<u64, GgufError> {
    let mut bytes = [0_u8; 8];
    read_exact(reader, file_size, &mut bytes, context)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_string<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
    maximum: u64,
    context: &'static str,
) -> Result<String, GgufError> {
    let length = read_u64(reader, file_size, "string length")?;
    if length > maximum {
        return Err(GgufError::InvalidStringLength {
            length,
            maximum,
            context,
        });
    }

    ensure_minimum_remaining(reader, file_size, length, 1, context)?;
    let length_usize = to_usize(length, "string length")?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length_usize)
        .map_err(|_| GgufError::AllocationFailed("GGUF string"))?;
    bytes.resize(length_usize, 0);
    read_exact(reader, file_size, &mut bytes, context)?;

    String::from_utf8(bytes).map_err(|_| GgufError::InvalidUtf8(context))
}

fn read_metadata_value<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
    options: &ParseOptions,
    value_type: MetadataValueType,
    depth: u32,
) -> Result<MetadataValue, GgufError> {
    Ok(match value_type {
        MetadataValueType::Uint8 => {
            MetadataValue::Uint8(read_u8(reader, file_size, "UINT8 metadata value")?)
        }
        MetadataValueType::Int8 => {
            MetadataValue::Int8(read_u8(reader, file_size, "INT8 metadata value")? as i8)
        }
        MetadataValueType::Uint16 => {
            MetadataValue::Uint16(read_u16(reader, file_size, "UINT16 metadata value")?)
        }
        MetadataValueType::Int16 => {
            MetadataValue::Int16(read_u16(reader, file_size, "INT16 metadata value")? as i16)
        }
        MetadataValueType::Uint32 => {
            MetadataValue::Uint32(read_u32(reader, file_size, "UINT32 metadata value")?)
        }
        MetadataValueType::Int32 => {
            MetadataValue::Int32(read_u32(reader, file_size, "INT32 metadata value")? as i32)
        }
        MetadataValueType::Float32 => MetadataValue::Float32(f32::from_bits(read_u32(
            reader,
            file_size,
            "FLOAT32 metadata value",
        )?)),
        MetadataValueType::Bool => {
            let value = read_u8(reader, file_size, "BOOL metadata value")?;
            match value {
                0 => MetadataValue::Bool(false),
                1 => MetadataValue::Bool(true),
                other => return Err(GgufError::InvalidBool(other)),
            }
        }
        MetadataValueType::String => MetadataValue::String(read_string(
            reader,
            file_size,
            options.max_string_bytes,
            "metadata string",
        )?),
        MetadataValueType::Array => {
            if depth >= options.max_array_depth {
                return Err(GgufError::ArrayDepthExceeded {
                    limit: options.max_array_depth,
                });
            }

            let raw_element_type = read_u32(reader, file_size, "array element type")?;
            let element_type = MetadataValueType::from_raw(raw_element_type)?;
            let count = read_u64(reader, file_size, "array element count")?;
            check_limit("metadata array element", count, options.max_array_elements)?;
            ensure_minimum_remaining(
                reader,
                file_size,
                count,
                minimum_encoded_value_size(element_type),
                "metadata array elements",
            )?;
            let capacity = to_usize(count, "metadata array element count")?;
            let mut values = Vec::new();
            values
                .try_reserve_exact(capacity)
                .map_err(|_| GgufError::AllocationFailed("metadata array"))?;

            for _ in 0..count {
                values.push(read_metadata_value(
                    reader,
                    file_size,
                    options,
                    element_type,
                    depth + 1,
                )?);
            }

            MetadataValue::Array {
                element_type,
                values,
            }
        }
        MetadataValueType::Uint64 => {
            MetadataValue::Uint64(read_u64(reader, file_size, "UINT64 metadata value")?)
        }
        MetadataValueType::Int64 => {
            MetadataValue::Int64(read_u64(reader, file_size, "INT64 metadata value")? as i64)
        }
        MetadataValueType::Float64 => MetadataValue::Float64(f64::from_bits(read_u64(
            reader,
            file_size,
            "FLOAT64 metadata value",
        )?)),
    })
}

fn validate_zero_padding<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
    padding: u64,
) -> Result<(), GgufError> {
    let mut remaining = padding;
    let mut buffer = [0_u8; 64];

    while remaining > 0 {
        let chunk_length = remaining.min(buffer.len() as u64) as usize;
        read_exact(
            reader,
            file_size,
            &mut buffer[..chunk_length],
            "tensor data padding",
        )?;
        if let Some(index) = buffer[..chunk_length].iter().position(|byte| *byte != 0) {
            let offset = reader
                .stream_position()?
                .checked_sub(chunk_length as u64)
                .and_then(|start| start.checked_add(index as u64))
                .ok_or(GgufError::Overflow("padding error offset"))?;
            return Err(GgufError::NonZeroPadding { offset });
        }
        remaining -= chunk_length as u64;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TensorFixture {
        name: &'static str,
        dimensions: &'static [u64],
        ggml_type: u32,
        offset: u64,
    }

    fn push_u32(output: &mut Vec<u8>, value: u32) {
        output.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(output: &mut Vec<u8>, value: u64) {
        output.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string(output: &mut Vec<u8>, value: &str) {
        push_u64(output, value.len() as u64);
        output.extend_from_slice(value.as_bytes());
    }

    fn string_value(value: &str) -> Vec<u8> {
        let mut output = Vec::new();
        push_string(&mut output, value);
        output
    }

    fn u32_value(value: u32) -> Vec<u8> {
        value.to_le_bytes().to_vec()
    }

    fn fixture(
        metadata: Vec<(&str, u32, Vec<u8>)>,
        tensors: &[TensorFixture],
        tensor_bytes: usize,
    ) -> Vec<u8> {
        let mut output = b"GGUF".to_vec();
        push_u32(&mut output, GGUF_VERSION);
        push_u64(&mut output, tensors.len() as u64);
        push_u64(&mut output, metadata.len() as u64);

        for (key, value_type, value) in metadata {
            push_string(&mut output, key);
            push_u32(&mut output, value_type);
            output.extend_from_slice(&value);
        }

        for tensor in tensors {
            push_string(&mut output, tensor.name);
            push_u32(&mut output, tensor.dimensions.len() as u32);
            for dimension in tensor.dimensions {
                push_u64(&mut output, *dimension);
            }
            push_u32(&mut output, tensor.ggml_type);
            push_u64(&mut output, tensor.offset);
        }

        let alignment = 32_usize;
        let padding = (alignment - output.len() % alignment) % alignment;
        output.resize(output.len() + padding, 0);
        output.resize(output.len() + tensor_bytes, 0);
        output
    }

    fn base_metadata() -> Vec<(&'static str, u32, Vec<u8>)> {
        vec![
            (
                "general.architecture",
                MetadataValueType::String as u32,
                string_value("llama"),
            ),
            (
                "general.alignment",
                MetadataValueType::Uint32 as u32,
                u32_value(32),
            ),
        ]
    }

    fn one_tensor() -> Vec<TensorFixture> {
        vec![TensorFixture {
            name: "token_embd.weight",
            dimensions: &[4, 8],
            ggml_type: 0,
            offset: 0,
        }]
    }

    #[test]
    fn parses_header_metadata_and_tensor_directory() {
        let bytes = fixture(base_metadata(), &one_tensor(), 128);
        let parsed = GgufFile::parse(&bytes).expect("valid GGUF fixture");

        assert_eq!(parsed.version, 3);
        assert_eq!(parsed.alignment, 32);
        assert_eq!(
            parsed.metadata_string("general.architecture"),
            Some("llama")
        );
        assert_eq!(parsed.tensors.len(), 1);
        assert_eq!(parsed.tensors[0].name, "token_embd.weight");
        assert_eq!(parsed.tensors[0].dimensions, [4, 8]);
        assert_eq!(parsed.tensors[0].ggml_type, 0);
        assert_eq!(parsed.tensors[0].offset, 0);
        assert_eq!(parsed.tensors[0].file_offset, parsed.tensor_data_offset);
        assert_eq!(parsed.tensors[0].element_count(), Some(32));
    }

    #[test]
    fn parses_string_arrays_without_reading_tensor_payload() {
        let mut array = Vec::new();
        push_u32(&mut array, MetadataValueType::String as u32);
        push_u64(&mut array, 2);
        push_string(&mut array, "hello");
        push_string(&mut array, "world");

        let mut metadata = base_metadata();
        metadata.push((
            "tokenizer.ggml.tokens",
            MetadataValueType::Array as u32,
            array,
        ));
        let bytes = fixture(metadata, &one_tensor(), 128);
        let parsed = GgufFile::parse(&bytes).expect("valid array metadata");

        assert_eq!(
            parsed.metadata_value("tokenizer.ggml.tokens"),
            Some(&MetadataValue::Array {
                element_type: MetadataValueType::String,
                values: vec![
                    MetadataValue::String("hello".into()),
                    MetadataValue::String("world".into())
                ]
            })
        );
    }

    #[test]
    fn rejects_invalid_magic_and_version() {
        let mut bytes = fixture(base_metadata(), &one_tensor(), 128);
        bytes[0] = b'X';
        assert!(matches!(
            GgufFile::parse(&bytes),
            Err(GgufError::InvalidMagic(_))
        ));

        let mut bytes = fixture(base_metadata(), &one_tensor(), 128);
        bytes[4..8].copy_from_slice(&99_u32.to_le_bytes());
        assert!(matches!(
            GgufFile::parse(&bytes),
            Err(GgufError::UnsupportedVersion(99))
        ));
    }

    #[test]
    fn rejects_truncated_input() {
        let bytes = fixture(base_metadata(), &one_tensor(), 128);
        assert!(matches!(
            GgufFile::parse(&bytes[..12]),
            Err(GgufError::Truncated { .. })
        ));
    }

    #[test]
    fn rejects_duplicate_metadata_keys() {
        let mut metadata = base_metadata();
        metadata.push((
            "general.architecture",
            MetadataValueType::String as u32,
            string_value("other"),
        ));
        let bytes = fixture(metadata, &one_tensor(), 128);
        assert!(matches!(
            GgufFile::parse(&bytes),
            Err(GgufError::DuplicateMetadataKey(_))
        ));
    }

    #[test]
    fn rejects_invalid_boolean_values() {
        let mut metadata = base_metadata();
        metadata.push(("test.enabled", MetadataValueType::Bool as u32, vec![2]));
        let bytes = fixture(metadata, &one_tensor(), 128);
        assert!(matches!(
            GgufFile::parse(&bytes),
            Err(GgufError::InvalidBool(2))
        ));
    }

    #[test]
    fn rejects_zero_dimensions_and_overflow() {
        let zero_dimension = vec![TensorFixture {
            name: "zero",
            dimensions: &[4, 0],
            ggml_type: 0,
            offset: 0,
        }];
        let bytes = fixture(base_metadata(), &zero_dimension, 128);
        assert!(matches!(
            GgufFile::parse(&bytes),
            Err(GgufError::ZeroDimension(_))
        ));

        let overflow = vec![TensorFixture {
            name: "overflow",
            dimensions: &[u64::MAX, 2],
            ggml_type: 0,
            offset: 0,
        }];
        let bytes = fixture(base_metadata(), &overflow, 128);
        assert!(matches!(
            GgufFile::parse(&bytes),
            Err(GgufError::ShapeOverflow(_))
        ));
    }

    #[test]
    fn rejects_misaligned_and_out_of_bounds_tensor_offsets() {
        let misaligned = vec![TensorFixture {
            name: "misaligned",
            dimensions: &[4],
            ggml_type: 0,
            offset: 1,
        }];
        let bytes = fixture(base_metadata(), &misaligned, 128);
        assert!(matches!(
            GgufFile::parse(&bytes),
            Err(GgufError::MisalignedTensorOffset { .. })
        ));

        let out_of_bounds = vec![TensorFixture {
            name: "outside",
            dimensions: &[4],
            ggml_type: 0,
            offset: 128,
        }];
        let bytes = fixture(base_metadata(), &out_of_bounds, 128);
        assert!(matches!(
            GgufFile::parse(&bytes),
            Err(GgufError::TensorDataOutOfBounds { .. })
        ));
    }

    #[test]
    fn rejects_non_zero_alignment_padding() {
        let mut bytes = fixture(base_metadata(), &one_tensor(), 128);
        let parsed = GgufFile::parse(&bytes).expect("initial fixture parses");
        let padding_start = parsed.tensor_data_offset as usize - 1;
        bytes[padding_start] = 1;

        assert!(matches!(
            GgufFile::parse(&bytes),
            Err(GgufError::NonZeroPadding { .. })
        ));
    }

    #[test]
    fn enforces_configured_resource_limits() {
        let bytes = fixture(base_metadata(), &one_tensor(), 128);
        let options = ParseOptions {
            max_tensor_count: 0,
            ..ParseOptions::default()
        };
        assert!(matches!(
            GgufFile::parse_with_options(&bytes, &options),
            Err(GgufError::LimitExceeded { kind: "tensor", .. })
        ));
    }

    #[test]
    fn parses_from_a_seekable_reader() {
        let bytes = fixture(base_metadata(), &one_tensor(), 128);
        let mut cursor = Cursor::new(bytes);
        let parsed = GgufFile::parse_reader(&mut cursor).expect("seekable GGUF reader");

        assert_eq!(parsed.tensors.len(), 1);
        assert_eq!(
            parsed.metadata_string("general.architecture"),
            Some("llama")
        );
    }
}
