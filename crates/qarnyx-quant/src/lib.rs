//! GGML tensor type registry and scalar reference dequantizers.
//!
//! The registry intentionally supports a bounded set of formats. Unknown or
//! unimplemented type IDs are rejected rather than assigned guessed sizes.
//! Decoders are safe Rust and write into caller-owned output buffers.

use std::fmt;

/// GGML tensor types currently understood by Qarnyx.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q8_0 = 8,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
}

/// Storage layout for one GGML block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgmlTypeInfo {
    pub kind: GgmlType,
    pub name: &'static str,
    pub block_size: u64,
    pub type_size: u64,
}

impl GgmlType {
    /// Return a supported GGML type for a raw GGUF type ID.
    pub const fn from_raw(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            8 => Some(Self::Q8_0),
            12 => Some(Self::Q4_K),
            13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K),
            _ => None,
        }
    }

    pub const fn info(self) -> GgmlTypeInfo {
        match self {
            Self::F32 => GgmlTypeInfo { kind: self, name: "F32", block_size: 1, type_size: 4 },
            Self::F16 => GgmlTypeInfo { kind: self, name: "F16", block_size: 1, type_size: 2 },
            Self::Q4_0 => GgmlTypeInfo { kind: self, name: "Q4_0", block_size: 32, type_size: 18 },
            Self::Q8_0 => GgmlTypeInfo { kind: self, name: "Q8_0", block_size: 32, type_size: 34 },
            Self::Q4_K => GgmlTypeInfo {
                kind: self,
                name: "Q4_K",
                block_size: 256,
                type_size: 144,
            },
            Self::Q5_K => GgmlTypeInfo {
                kind: self,
                name: "Q5_K",
                block_size: 256,
                type_size: 176,
            },
            Self::Q6_K => GgmlTypeInfo {
                kind: self,
                name: "Q6_K",
                block_size: 256,
                type_size: 210,
            },
        }
    }
}

/// Errors from GGML type sizing and scalar dequantization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantError {
    UnsupportedType(u32),
    EmptyShape,
    ZeroDimension,
    RowDimensionNotBlockAligned { ggml_type: u32, row_elements: u64, block_size: u64 },
    ShapeOverflow,
    ByteLengthOverflow,
    InputLengthMismatch { expected: u64, actual: usize },
    OutputLengthMismatch { expected: u64, actual: usize },
    AllocationFailed,
}

impl fmt::Display for QuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedType(raw) => write!(
                f,
                "GGML tensor type {raw} is not supported by Qarnyx yet"
            ),
            Self::EmptyShape => write!(f, "tensor shape must contain at least one dimension"),
            Self::ZeroDimension => write!(f, "tensor shape contains a zero-sized dimension"),
            Self::RowDimensionNotBlockAligned {
                ggml_type,
                row_elements,
                block_size,
            } => write!(
                f,
                "GGML type {ggml_type} row length {row_elements} is not divisible by block size {block_size}"
            ),
            Self::ShapeOverflow => write!(f, "tensor element count overflowed u64"),
            Self::ByteLengthOverflow => write!(f, "encoded tensor byte length overflowed u64"),
            Self::InputLengthMismatch { expected, actual } => write!(
                f,
                "encoded tensor has {actual} bytes; expected exactly {expected}"
            ),
            Self::OutputLengthMismatch { expected, actual } => write!(
                f,
                "output buffer has {actual} elements; expected exactly {expected}"
            ),
            Self::AllocationFailed => write!(f, "unable to reserve the dequantized output buffer"),
        }
    }
}

impl std::error::Error for QuantError {}

/// Return the supported layout for a raw GGML type ID.
pub const fn type_info(raw: u32) -> Option<GgmlTypeInfo> {
    match GgmlType::from_raw(raw) {
        Some(kind) => Some(kind.info()),
        None => None,
    }
}

/// Calculate the exact encoded tensor length for a GGML type and GGML shape.
///
/// Quantized rows are independently block encoded, so dimension 0 must be
/// divisible by the block size even when the total element count is divisible.
pub fn encoded_len(ggml_type: u32, dimensions: &[u64]) -> Result<u64, QuantError> {
    let (_, _, byte_len) = shape_layout(ggml_type, dimensions)?;
    Ok(byte_len)
}

/// Dequantize a tensor into a caller-owned F32 output slice.
///
/// Both input and output lengths must match the exact shape-derived sizes.
pub fn dequantize_into(
    ggml_type: u32,
    dimensions: &[u64],
    encoded: &[u8],
    output: &mut [f32],
) -> Result<(), QuantError> {
    let (info, element_count, byte_len) = shape_layout(ggml_type, dimensions)?;
    if encoded.len() as u64 != byte_len {
        return Err(QuantError::InputLengthMismatch { expected: byte_len, actual: encoded.len() });
    }
    if output.len() as u64 != element_count {
        return Err(QuantError::OutputLengthMismatch {
            expected: element_count,
            actual: output.len(),
        });
    }

    match info.kind {
        GgmlType::F32 => dequantize_f32(encoded, output),
        GgmlType::F16 => dequantize_f16(encoded, output),
        GgmlType::Q4_0 => dequantize_q4_0(encoded, output),
        GgmlType::Q8_0 => dequantize_q8_0(encoded, output),
        GgmlType::Q4_K => dequantize_q4_k(encoded, output),
        GgmlType::Q5_K => dequantize_q5_k(encoded, output),
        GgmlType::Q6_K => dequantize_q6_k(encoded, output),
    }
    Ok(())
}

/// Allocate and return an F32 tensor decoded from its GGML storage format.
///
/// Prefer dequantize_into when the caller manages reusable output buffers.
pub fn dequantize(
    ggml_type: u32,
    dimensions: &[u64],
    encoded: &[u8],
) -> Result<Vec<f32>, QuantError> {
    let (_, element_count, _) = shape_layout(ggml_type, dimensions)?;
    let capacity = usize::try_from(element_count).map_err(|_| QuantError::AllocationFailed)?;
    let mut output = Vec::new();
    output.try_reserve_exact(capacity).map_err(|_| QuantError::AllocationFailed)?;
    output.resize(capacity, 0.0);
    dequantize_into(ggml_type, dimensions, encoded, &mut output)?;
    Ok(output)
}

fn shape_layout(
    ggml_type: u32,
    dimensions: &[u64],
) -> Result<(GgmlTypeInfo, u64, u64), QuantError> {
    let kind = GgmlType::from_raw(ggml_type).ok_or(QuantError::UnsupportedType(ggml_type))?;
    let info = kind.info();
    let Some(&row_elements) = dimensions.first() else {
        return Err(QuantError::EmptyShape);
    };
    if dimensions.contains(&0) {
        return Err(QuantError::ZeroDimension);
    }
    if row_elements % info.block_size != 0 {
        return Err(QuantError::RowDimensionNotBlockAligned {
            ggml_type, row_elements, block_size: info.block_size,
        });
    }

    let element_count = dimensions.iter().try_fold(1_u64, |count, dimension| {
        count.checked_mul(*dimension).ok_or(QuantError::ShapeOverflow)
    })?;
    let row_count = element_count / row_elements;
    let blocks_per_row = row_elements / info.block_size;
    let byte_len = row_count.checked_mul(blocks_per_row)
        .and_then(|blocks| blocks.checked_mul(info.type_size))
        .ok_or(QuantError::ByteLengthOverflow)?;
    Ok((info, element_count, byte_len))
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = u32::from(bits & 0x03ff);

    let output = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let mut normalized = mantissa;
            let mut unbiased_exponent = -14_i32;
            while normalized & 0x0400 == 0 {
                normalized <<= 1;
                unbiased_exponent -= 1;
            }
            normalized &= 0x03ff;
            sign | (((unbiased_exponent + 127) as u32) << 23) | (normalized << 13)
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((u32::from(exponent) + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(output)
}

fn read_half(bytes: &[u8]) -> f32 {
    half_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn dequantize_f32(encoded: &[u8], output: &mut [f32]) {
    for (bytes, value) in encoded.chunks_exact(4).zip(output.iter_mut()) {
        *value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    }
}

fn dequantize_f16(encoded: &[u8], output: &mut [f32]) {
    for (bytes, value) in encoded.chunks_exact(2).zip(output.iter_mut()) {
        *value = read_half(bytes);
    }
}

fn dequantize_q4_0(encoded: &[u8], output: &mut [f32]) {
    for (block_index, block) in encoded.chunks_exact(18).enumerate() {
        let scale = read_half(&block[..2]);
        let base = block_index * 32;
        for index in 0..16 {
            let packed = block[2 + index];
            let low = i32::from(packed & 0x0f) - 8;
            let high = i32::from(packed >> 4) - 8;
            output[base + index] = low as f32 * scale;
            output[base + index + 16] = high as f32 * scale;
        }
    }
}

fn dequantize_q8_0(encoded: &[u8], output: &mut [f32]) {
    for (block_index, block) in encoded.chunks_exact(34).enumerate() {
        let scale = read_half(&block[..2]);
        let base = block_index * 32;
        for index in 0..32 {
            output[base + index] = (block[2 + index] as i8) as f32 * scale;
        }
    }
}

fn get_scale_min_k4(index: usize, packed: &[u8]) -> (u8, u8) {
    if index < 4 {
        (packed[index] & 0x3f, packed[index + 4] & 0x3f)
    } else {
        (
            (packed[index + 4] & 0x0f) | ((packed[index - 4] >> 6) << 4),
            (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4),
        )
    }
}

fn dequantize_q4_k(encoded: &[u8], output: &mut [f32]) {
    for (block_index, block) in encoded.chunks_exact(144).enumerate() {
        let scale = read_half(&block[..2]);
        let min_scale = read_half(&block[2..4]);
        let packed_scales = &block[4..16];
        let quants = &block[16..144];
        let base = block_index * 256;

        for pair in 0..4 {
            let quant_offset = pair * 32;
            for half in 0..2 {
                let scale_index = pair * 2 + half;
                let (local_scale, local_min) = get_scale_min_k4(scale_index, packed_scales);
                let d = scale * f32::from(local_scale);
                let m = min_scale * f32::from(local_min);
                for index in 0..32 {
                    let packed = quants[quant_offset + index];
                    let quant = if half == 0 { packed & 0x0f } else { packed >> 4 };
                    let output_index = base + pair * 64 + half * 32 + index;
                    output[output_index] = d * f32::from(quant) - m;
                }
            }
        }
    }
}

fn dequantize_q5_k(encoded: &[u8], output: &mut [f32]) {
    for (block_index, block) in encoded.chunks_exact(176).enumerate() {
        let scale = read_half(&block[..2]);
        let min_scale = read_half(&block[2..4]);
        let packed_scales = &block[4..16];
        let high_bits = &block[16..48];
        let low_bits = &block[48..176];
        let base = block_index * 256;

        for pair in 0..4 {
            let quant_offset = pair * 32;
            let low_mask = 1_u8 << (pair * 2);
            let high_mask = 2_u8 << (pair * 2);
            for half in 0..2 {
                let scale_index = pair * 2 + half;
                let (local_scale, local_min) = get_scale_min_k4(scale_index, packed_scales);
                let d = scale * f32::from(local_scale);
                let m = min_scale * f32::from(local_min);
                for index in 0..32 {
                    let packed = low_bits[quant_offset + index];
                    let mask = if half == 0 { low_mask } else { high_mask };
                    let high = u8::from(high_bits[index] & mask != 0);
                    let low = if half == 0 { packed & 0x0f } else { packed >> 4 };
                    let quant = low | (high << 4);
                    let output_index = base + pair * 64 + half * 32 + index;
                    output[output_index] = d * f32::from(quant) - m;
                }
            }
        }
    }
}

fn dequantize_q6_k(encoded: &[u8], output: &mut [f32]) {
    for (block_index, block) in encoded.chunks_exact(210).enumerate() {
        let low_bits = &block[..128];
        let high_bits = &block[128..192];
        let scales = &block[192..208];
        let scale = read_half(&block[208..210]);
        let base = block_index * 256;

        for half_block in 0..2 {
            let low_offset = half_block * 64;
            let high_offset = half_block * 32;
            let scale_offset = half_block * 8;
            let output_base = base + half_block * 128;

            for index in 0..32 {
                let high = high_bits[high_offset + index];
                let low_a = low_bits[low_offset + index];
                let low_b = low_bits[low_offset + index + 32];
                let scale_group = index / 16;

                let q1 = i32::from((low_a & 0x0f) | ((high & 0x03) << 4)) - 32;
                let q2 = i32::from((low_b & 0x0f) | (((high >> 2) & 0x03) << 4)) - 32;
                let q3 = i32::from((low_a >> 4) | (((high >> 4) & 0x03) << 4)) - 32;
                let q4 = i32::from((low_b >> 4) | (((high >> 6) & 0x03) << 4)) - 32;

                output[output_base + index] =
                    scale * (scales[scale_offset + scale_group] as i8) as f32 * q1 as f32;
                output[output_base + 32 + index] =
                    scale * (scales[scale_offset + scale_group + 2] as i8) as f32 * q2 as f32;
                output[output_base + 64 + index] =
                    scale * (scales[scale_offset + scale_group + 4] as i8) as f32 * q3 as f32;
                output[output_base + 96 + index] =
                    scale * (scales[scale_offset + scale_group + 6] as i8) as f32 * q4 as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_half(bytes: &mut [u8], offset: usize, value: f32) {
        let bits = match value {
            1.0 => 0x3c00_u16,
            0.5 => 0x3800_u16,
            -1.0 => 0xbc00_u16,
            _ => panic!("test helper only handles exact fixtures"),
        };
        bytes[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
    }

    #[test]
    fn registry_reports_supported_block_layouts() {
        let cases = [
            (0, "F32", 1, 4),
            (1, "F16", 1, 2),
            (2, "Q4_0", 32, 18),
            (8, "Q8_0", 32, 34),
            (12, "Q4_K", 256, 144),
            (13, "Q5_K", 256, 176),
            (14, "Q6_K", 256, 210),
        ];
        for (raw, name, block_size, type_size) in cases {
            let info = type_info(raw).expect("supported type");
            assert_eq!(info.name, name);
            assert_eq!(info.block_size, block_size);
            assert_eq!(info.type_size, type_size);
        }
        assert!(type_info(15).is_none());
    }

    #[test]
    fn encoded_len_uses_row_block_alignment() {
        assert_eq!(encoded_len(12, &[2048, 2048]), Ok(2_359_296));
        assert_eq!(encoded_len(13, &[256]), Ok(176));
        assert_eq!(encoded_len(13, &[2048, 6144]), Ok(8_650_752));
        assert_eq!(encoded_len(14, &[512, 3]), Ok(1_260));
        assert_eq!(encoded_len(14, &[2048, 248_320]), Ok(417_177_600));
        assert_eq!(
            encoded_len(12, &[128, 2]),
            Err(QuantError::RowDimensionNotBlockAligned {
                ggml_type: 12,
                row_elements: 128,
                block_size: 256,
            })
        );
    }

    #[test]
    fn decodes_f32_and_f16_values() {
        let f32_bytes = [1.0_f32.to_le_bytes(), (-2.0_f32).to_le_bytes()].concat();
        assert_eq!(dequantize(0, &[2], &f32_bytes).unwrap(), [1.0, -2.0]);

        let f16_bytes = [0x3c_u8, 0x00, 0x00, 0xc0, 0x01, 0x00];
        let decoded = dequantize(1, &[3], &f16_bytes).unwrap();
        assert_eq!(decoded[0], 1.0);
        assert_eq!(decoded[1], -2.0);
        assert_eq!(decoded[2], 2.0_f32.powi(-24));
    }

    #[test]
    fn decodes_q4_0_nibbles_in_low_then_high_halves() {
        let mut block = vec![0_u8; 18];
        put_half(&mut block, 0, 1.0);
        block[2..].fill(0x98);
        let decoded = dequantize(2, &[32], &block).unwrap();
        assert!(decoded[..16].iter().all(|value| *value == 0.0));
        assert!(decoded[16..].iter().all(|value| *value == 1.0));
    }

    #[test]
    fn decodes_q8_0_signed_values() {
        let mut block = vec![0_u8; 34];
        put_half(&mut block, 0, 0.5);
        block[2] = 2;
        block[3] = (-2_i8) as u8;
        let decoded = dequantize(8, &[32], &block).unwrap();
        assert_eq!(decoded[0], 1.0);
        assert_eq!(decoded[1], -1.0);
        assert!(decoded[2..].iter().all(|value| *value == 0.0));
    }

    #[test]
    fn decodes_q4_k_scale_min_groups_and_nibbles() {
        let mut block = vec![0_u8; 144];
        put_half(&mut block, 0, 1.0);
        block[4] = 1;
        block[5] = 2;
        block[16..48].fill(0x21);

        let decoded = dequantize(12, &[256], &block).unwrap();
        assert!(decoded[..32].iter().all(|value| *value == 1.0));
        assert!(decoded[32..64].iter().all(|value| *value == 4.0));
        assert!(decoded[64..].iter().all(|value| *value == 0.0));
    }

    #[test]
    fn decodes_q5_k_fifth_bit_planes() {
        let mut block = vec![0_u8; 176];
        put_half(&mut block, 0, 1.0);
        block[4] = 1;
        block[5] = 2;
        block[6] = 3;
        block[7] = 4;
        block[48..80].fill(0x21);
        block[80..112].fill(0x43);
        block[16] = 0b0000_1110;

        let decoded = dequantize(13, &[256], &block).unwrap();
        assert_eq!(decoded[0], 1.0);
        assert_eq!(decoded[1], 1.0);
        assert_eq!(decoded[32], 36.0);
        assert_eq!(decoded[64], 57.0);
        assert_eq!(decoded[96], 80.0);
        assert_eq!(decoded[128], 0.0);
    }

    #[test]
    fn decodes_q6_k_six_bit_values_and_signed_scales() {
        let mut block = vec![0_u8; 210];
        block[0] = 0x21;
        block[128] = 0b1110_0100;
        block[192..208].fill(1);
        put_half(&mut block, 208, 1.0);

        let decoded = dequantize(14, &[256], &block).unwrap();
        assert_eq!(decoded[0], -31.0);
        assert_eq!(decoded[32], -16.0);
        assert_eq!(decoded[64], 2.0);
        assert_eq!(decoded[96], 16.0);
    }

    #[test]
    fn decodes_second_q6_k_half_block_with_its_own_scales() {
        let mut block = vec![0_u8; 210];
        block[64] = 0x03;
        block[160] = 0b0000_0001;
        block[200] = 2;
        put_half(&mut block, 208, 1.0);

        let decoded = dequantize(14, &[256], &block).unwrap();
        assert_eq!(decoded[128], -26.0);
        assert_eq!(decoded[129], -64.0);
    }

    #[test]
    fn rejects_unknown_types_and_inexact_encoded_lengths() {
        assert_eq!(encoded_len(999, &[1]), Err(QuantError::UnsupportedType(999)));

        let err = dequantize(8, &[32], &[0_u8; 33]).unwrap_err();
        assert_eq!(err, QuantError::InputLengthMismatch { expected: 34, actual: 33 });
    }
}
