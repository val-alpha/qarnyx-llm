use qarnyx_core::{DType, TensorView};
use qarnyx_gguf::{GgufFile, TensorInfo};
use qarnyx_quant::{dequantize, GgmlType};
use thiserror::Error;

use crate::{forward_block, forward_block_with_cache, LlamaConfig, LlamaKvCache, LlamaLayerWeights, ModelError};

/// Errors raised while mapping a parsed GGUF tensor directory onto a
/// LLaMA-style model.
#[derive(Debug, Error)]
pub enum TensorLoadError {
    #[error("invalid model configuration: {0}")]
    Config(String),

    #[error("missing model tensor: {0}")]
    MissingTensor(String),

    #[error("tensor {name} uses unsupported GGML type {ggml_type} for a model weight")]
    UnsupportedType { name: String, ggml_type: u32 },

    #[error("tensor {name} has shape {actual:?}; expected {expected:?}")]
    ShapeMismatch {
        name: String,
        actual: Vec<u64>,
        expected: Vec<usize>,
    },

    #[error("tensor {name} payload [{offset}, {end}) exceeds input buffer of {len} bytes")]
    PayloadOutOfBounds {
        name: String,
        offset: usize,
        end: usize,
        len: usize,
    },

    #[error("tensor {name} payload offset {value} exceeds usize")]
    OffsetTooLarge { name: String, value: u64 },

    #[error("tensor {name} payload length {value} exceeds usize")]
    ByteLengthTooLarge { name: String, value: u64 },

    #[error("failed to decode tensor {name}: {source}")]
    Decode {
        name: String,
        source: qarnyx_quant::QuantError,
    },
}

impl TensorLoadError {
    fn expected_shape(name: &str, actual: &[u64], expected: &[usize]) -> Self {
        Self::ShapeMismatch {
            name: name.into(),
            actual: actual.to_vec(),
            expected: expected.to_vec(),
        }
    }
}

/// File-backed tensor mapping for one LLaMA decoder block.
///
/// Matrix weights remain encoded and borrow directly from the GGUF payload.
/// RMSNorm vectors are decoded to F32 because the execution backend currently
/// consumes RMSNorm scales as `&[f32]`.
#[derive(Debug)]
pub struct LlamaLayerStorage<'a> {
    pub attention_norm: Vec<f32>,
    pub q_proj: TensorView<'a>,
    pub k_proj: TensorView<'a>,
    pub v_proj: TensorView<'a>,
    pub o_proj: TensorView<'a>,
    pub ffn_norm: Vec<f32>,
    pub gate_proj: TensorView<'a>,
    pub up_proj: TensorView<'a>,
    pub down_proj: TensorView<'a>,
}

impl<'a> LlamaLayerStorage<'a> {
    fn weights(&self) -> LlamaLayerWeights<'_> {
        LlamaLayerWeights {
            attention_norm: &self.attention_norm,
            q_proj: self.q_proj,
            k_proj: self.k_proj,
            v_proj: self.v_proj,
            o_proj: self.o_proj,
            ffn_norm: &self.ffn_norm,
            gate_proj: self.gate_proj,
            up_proj: self.up_proj,
            down_proj: self.down_proj,
        }
    }
}

/// A complete LLaMA-style decoder stack loaded from a validated GGUF
/// tensor directory.
#[derive(Debug)]
pub struct LlamaModel<'a> {
    pub config: LlamaConfig,
    pub layers: Vec<LlamaLayerStorage<'a>>,
}

impl<'a> LlamaModel<'a> {
    /// Build a LLaMA model from an already-parsed GGUF file and its bytes.
    ///
    /// Matrix tensor payloads are borrowed directly from `data`; the caller
    /// must keep the GGUF byte buffer alive for the lifetime of this model.
    pub fn from_gguf(gguf: &GgufFile, data: &'a [u8]) -> Result<Self, TensorLoadError> {
        let config =
            LlamaConfig::from_gguf(gguf).map_err(|error| TensorLoadError::Config(error.to_string()))?;
        let mut layers = Vec::with_capacity(config.block_count);

        for block in 0..config.block_count {
            layers.push(load_layer(gguf, data, &config, block)?);
        }

        Ok(Self { config, layers })
    }

    /// Run every decoder block for one token without retaining a KV cache.
    pub fn forward<B: qarnyx_core::Backend>(
        &self,
        backend: &B,
        input: &[f32],
        position: usize,
    ) -> Result<Vec<f32>, ModelError> {
        let mut cache = LlamaKvCache::new(&self.config);
        self.forward_with_cache(backend, input, position, &mut cache)
    }

    /// Run one token through the full decoder stack and append its K/V states.
    pub fn forward_with_cache<B: qarnyx_core::Backend>(
        &self,
        backend: &B,
        input: &[f32],
        position: usize,
        cache: &mut LlamaKvCache,
    ) -> Result<Vec<f32>, ModelError> {
        if self.layers.len() != self.config.block_count {
            return Err(ModelError::InvalidConfig(
                "model layer count does not match configuration",
            ));
        }
        cache.validate(&self.config)?;
        if cache.sequence_len() != position {
            return Err(ModelError::InvalidConfig(
                "token position must equal the current KV cache sequence length",
            ));
        }

        let mut hidden = input.to_vec();
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let weights = layer.weights();
            hidden = forward_block_with_cache(
                backend,
                &self.config,
                &weights,
                &hidden,
                position,
                cache,
                layer_index,
            )?;
        }
        cache.advance();
        Ok(hidden)
    }
}

fn load_layer<'a>(
    gguf: &GgufFile,
    data: &'a [u8],
    config: &LlamaConfig,
    block: usize,
) -> Result<LlamaLayerStorage<'a>, TensorLoadError> {
    let prefix = format!("blk.{block}");
    let hidden = config.embedding_length;
    let kv_hidden = config.attention_head_count_kv * config.attention_key_length;
    let ff = config.feed_forward_length;

    Ok(LlamaLayerStorage {
        attention_norm: load_f32_tensor(
            gguf,
            data,
            &format!("{prefix}.attn_norm.weight"),
            &[hidden],
        )?,
        q_proj: load_weight(
            gguf,
            data,
            &format!("{prefix}.attn_q.weight"),
            &[hidden, hidden],
        )?,
        k_proj: load_weight(
            gguf,
            data,
            &format!("{prefix}.attn_k.weight"),
            &[kv_hidden, hidden],
        )?,
        v_proj: load_weight(
            gguf,
            data,
            &format!("{prefix}.attn_v.weight"),
            &[kv_hidden, hidden],
        )?,
        o_proj: load_weight(
            gguf,
            data,
            &format!("{prefix}.attn_output.weight"),
            &[hidden, hidden],
        )?,
        ffn_norm: load_f32_tensor(
            gguf,
            data,
            &format!("{prefix}.ffn_norm.weight"),
            &[hidden],
        )?,
        gate_proj: load_weight(
            gguf,
            data,
            &format!("{prefix}.ffn_gate.weight"),
            &[ff, hidden],
        )?,
        up_proj: load_weight(
            gguf,
            data,
            &format!("{prefix}.ffn_up.weight"),
            &[ff, hidden],
        )?,
        down_proj: load_weight(
            gguf,
            data,
            &format!("{prefix}.ffn_down.weight"),
            &[hidden, ff],
        )?,
    })
}

fn load_weight<'a>(
    gguf: &GgufFile,
    data: &'a [u8],
    name: &str,
    expected: &[usize],
) -> Result<TensorView<'a>, TensorLoadError> {
    let tensor = find_tensor(gguf, name)?;
    validate_matrix_shape(tensor, expected)?;
    let dtype = dtype_from_ggml(tensor)?;
    let bytes = tensor_bytes(tensor, data)?;

    // GGUF stores ggml matrix dimensions as [columns, rows]. The CPU
    // backend exposes the logical matrix as [rows, columns].
    Ok(TensorView::new(bytes, leaked_shape(expected), dtype))
}

fn validate_matrix_shape(tensor: &TensorInfo, expected: &[usize]) -> Result<(), TensorLoadError> {
    if expected.len() != 2
        || tensor.dimensions.len() != 2
        || tensor.dimensions[0] != expected[1] as u64
        || tensor.dimensions[1] != expected[0] as u64
    {
        return Err(TensorLoadError::expected_shape(
            &tensor.name,
            &tensor.dimensions,
            &[expected[1], expected[0]],
        ));
    }
    Ok(())
}

fn load_f32_tensor(
    gguf: &GgufFile,
    data: &[u8],
    name: &str,
    expected: &[usize],
) -> Result<Vec<f32>, TensorLoadError> {
    let tensor = find_tensor(gguf, name)?;
    validate_shape(tensor, expected)?;
    let bytes = tensor_bytes(tensor, data)?;

    dequantize(tensor.ggml_type, &tensor.dimensions, bytes).map_err(|source| {
        TensorLoadError::Decode {
            name: name.into(),
            source,
        }
    })
}

fn find_tensor<'a>(gguf: &'a GgufFile, name: &str) -> Result<&'a TensorInfo, TensorLoadError> {
    gguf.tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| TensorLoadError::MissingTensor(name.into()))
}

fn validate_shape(tensor: &TensorInfo, expected: &[usize]) -> Result<(), TensorLoadError> {
    if tensor.dimensions.len() != expected.len()
        || tensor
            .dimensions
            .iter()
            .zip(expected)
            .any(|(&actual, &expected)| actual != expected as u64)
    {
        return Err(TensorLoadError::expected_shape(
            &tensor.name,
            &tensor.dimensions,
            expected,
        ));
    }
    Ok(())
}

fn tensor_bytes<'a>(
    tensor: &TensorInfo,
    data: &'a [u8],
) -> Result<&'a [u8], TensorLoadError> {
    let offset = usize::try_from(tensor.file_offset).map_err(|_| {
        TensorLoadError::OffsetTooLarge {
            name: tensor.name.clone(),
            value: tensor.file_offset,
        }
    })?;
    let byte_len = usize::try_from(tensor.byte_len).map_err(|_| {
        TensorLoadError::ByteLengthTooLarge {
            name: tensor.name.clone(),
            value: tensor.byte_len,
        }
    })?;
    let end = offset
        .checked_add(byte_len)
        .ok_or(TensorLoadError::PayloadOutOfBounds {
            name: tensor.name.clone(),
            offset,
            end: usize::MAX,
            len: data.len(),
        })?;
    if end > data.len() {
        return Err(TensorLoadError::PayloadOutOfBounds {
            name: tensor.name.clone(),
            offset,
            end,
            len: data.len(),
        });
    }
    Ok(&data[offset..end])
}

fn dtype_from_ggml(tensor: &TensorInfo) -> Result<DType, TensorLoadError> {
    match GgmlType::from_raw(tensor.ggml_type) {
        Some(GgmlType::F32) => Ok(DType::F32),
        Some(GgmlType::F16) => Ok(DType::F16),
        Some(GgmlType::Q4_0) => Ok(DType::Q4_0),
        Some(GgmlType::Q8_0) => Ok(DType::Q8_0),
        Some(GgmlType::Q4_K) => Ok(DType::Q4K),
        Some(GgmlType::Q5_K) => Ok(DType::Q5K),
        Some(GgmlType::Q6_K) => Ok(DType::Q6K),
        None => Err(TensorLoadError::UnsupportedType {
            name: tensor.name.clone(),
            ggml_type: tensor.ggml_type,
        }),
    }
}

fn leaked_shape(shape: &[usize]) -> &'static [usize] {
    Box::leak(shape.to_vec().into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn push_metadata_string(output: &mut Vec<u8>, key: &str, value: &str) {
        push_string(output, key);
        push_u32(output, 8);
        push_string(output, value);
    }

    fn push_metadata_u32(output: &mut Vec<u8>, key: &str, value: u32) {
        push_string(output, key);
        push_u32(output, 4);
        push_u32(output, value);
    }

    fn push_metadata_f32(output: &mut Vec<u8>, key: &str, value: f32) {
        push_string(output, key);
        push_u32(output, 6);
        push_u32(output, value.to_bits());
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn build_tiny_llama_gguf() -> Vec<u8> {
        let tensors = [
            ("blk.0.attn_norm.weight", vec![1.0, 1.0, 1.0, 1.0], vec![4_u64]),
            (
                "blk.0.attn_q.weight",
                vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                    1.0,
                ],
                vec![4, 4],
            ),
            (
                "blk.0.attn_k.weight",
                vec![0.0; 8],
                vec![4, 2],
            ),
            (
                "blk.0.attn_v.weight",
                vec![0.0; 8],
                vec![4, 2],
            ),
            (
                "blk.0.attn_output.weight",
                vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                    1.0,
                ],
                vec![4, 4],
            ),
            ("blk.0.ffn_norm.weight", vec![1.0, 1.0, 1.0, 1.0], vec![4]),
            ("blk.0.ffn_gate.weight", vec![0.0; 32], vec![4, 8]),
            ("blk.0.ffn_up.weight", vec![0.0; 32], vec![4, 8]),
            ("blk.0.ffn_down.weight", vec![0.0; 32], vec![8, 4]),
        ];

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, tensors.len() as u64);
        push_u64(&mut bytes, 9);
        push_metadata_string(&mut bytes, "general.architecture", "llama");
        push_metadata_u32(&mut bytes, "general.alignment", 32);
        push_metadata_u32(&mut bytes, "llama.block_count", 1);
        push_metadata_u32(&mut bytes, "llama.embedding_length", 4);
        push_metadata_u32(&mut bytes, "llama.feed_forward_length", 8);
        push_metadata_u32(&mut bytes, "llama.attention.head_count", 2);
        push_metadata_u32(&mut bytes, "llama.attention.head_count_kv", 1);
        push_metadata_u32(&mut bytes, "llama.attention.key_length", 2);
        push_metadata_u32(&mut bytes, "llama.attention.value_length", 2);
        push_metadata_f32(
            &mut bytes,
            "llama.attention.layer_norm_rms_epsilon",
            1.0e-5,
        );

        let directory_start = bytes.len();
        let mut offsets = Vec::with_capacity(tensors.len());
        let mut payload_offset = 0_u64;

        for (name, values, dimensions) in &tensors {
            let padding = (32 - (payload_offset as usize % 32)) % 32;
            payload_offset += padding as u64;
            offsets.push(payload_offset);
            push_string(&mut bytes, name);
            push_u32(&mut bytes, dimensions.len() as u32);
            for dimension in dimensions {
                push_u64(&mut bytes, *dimension);
            }
            push_u32(&mut bytes, 0);
            push_u64(&mut bytes, payload_offset);
            payload_offset += (values.len() * 4) as u64;
        }

        let tensor_data_offset = ((bytes.len() + 31) / 32) * 32;
        bytes.resize(tensor_data_offset, 0);

        for ((_, values, _), offset) in tensors.iter().zip(offsets) {
            let absolute = tensor_data_offset + offset as usize;
            let payload = f32_bytes(values);
            let end = absolute + payload.len();
            if bytes.len() < end {
                bytes.resize(end, 0);
            }
            bytes[absolute..end].copy_from_slice(&payload);
        }

        let _ = directory_start;
        bytes
    }

    #[test]
    fn loads_tiny_llama_model_end_to_end() {
        let data = build_tiny_llama_gguf();
        let gguf = GgufFile::parse(&data).expect("tiny GGUF fixture parses");

        let model = LlamaModel::from_gguf(&gguf, &data).expect("LLaMA tensor mapping succeeds");

        assert_eq!(model.config.embedding_length, 4);
        assert_eq!(model.layers.len(), 1);
        assert_eq!(model.layers[0].attention_norm, vec![1.0; 4]);
        assert_eq!(model.layers[0].q_proj.dtype, DType::F32);
        assert_eq!(model.layers[0].q_proj.shape, [4, 4]);
        assert_eq!(
            model.layers[0].q_proj.data,
            f32_bytes(&[
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0
            ])
        );
        assert_eq!(model.layers[0].k_proj.shape, [2, 4]);
        assert_eq!(model.layers[0].v_proj.shape, [2, 4]);
        assert_eq!(model.layers[0].gate_proj.shape, [8, 4]);
        assert_eq!(model.layers[0].down_proj.shape, [4, 8]);

        let backend = qarnyx_cpu::CpuBackend::new(1).expect("CPU backend");
        let output = model
            .forward(&backend, &[1.0, 2.0, 3.0, 4.0], 0)
            .expect("tiny model forward");

        assert_eq!(output, [1.0, 2.0, 3.0, 4.0]);
    }
}
