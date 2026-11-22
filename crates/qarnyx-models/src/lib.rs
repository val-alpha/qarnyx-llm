//! LLaMA-style decoder-only Transformer configuration and scalar forward pass.
//!
//! M4 keeps tensor loading separate from execution. GGUF tensor descriptors
//! can later be mapped to file-backed TensorView values without changing the
//! forward graph.

mod loader;
mod qwen35;

pub use loader::{LlamaLayerStorage, LlamaModel, TensorLoadError};
pub use qwen35::{Qwen35Cache, Qwen35Config, Qwen35Layer, Qwen35Model, Qwen35Profile};

use qarnyx_core::{Backend, BackendError, TensorView};
use qarnyx_gguf::{GgufFile, MetadataValue};
use thiserror::Error;

/// Errors raised while constructing or executing a LLaMA-style model.
#[derive(Debug, Error)]
pub enum ModelError {
    #[error("missing GGUF metadata key: {0}")]
    MissingMetadata(&'static str),

    #[error("invalid GGUF metadata for {key}: expected {expected}")]
    InvalidMetadata {
        key: &'static str,
        expected: &'static str,
    },

    #[error("invalid LLaMA configuration: {0}")]
    InvalidConfig(&'static str),

    #[error("backend operation failed: {0}")]
    Backend(#[from] BackendError),

    #[error("model buffer shape mismatch: {0}")]
    ShapeMismatch(String),
}

/// Configuration needed by a LLaMA-style decoder block.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaConfig {
    pub block_count: usize,
    pub embedding_length: usize,
    pub feed_forward_length: usize,
    pub attention_head_count: usize,
    pub attention_head_count_kv: usize,
    pub attention_key_length: usize,
    pub attention_value_length: usize,
    pub attention_layer_norm_rms_epsilon: f32,
    pub rope_freq_base: f32,
}

impl LlamaConfig {
    /// Extracts the LLaMA configuration from standard GGUF metadata keys.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, ModelError> {
        let key_dim = metadata_usize(gguf, "llama.attention.key_length")?;
        let value_dim = metadata_usize(gguf, "llama.attention.value_length")?;
        let embedding_length = metadata_usize(gguf, "llama.embedding_length")?;
        let head_count = metadata_usize(gguf, "llama.attention.head_count")?;
        let head_count_kv = metadata_usize(gguf, "llama.attention.head_count_kv")?;

        if head_count == 0 || head_count_kv == 0 || head_count % head_count_kv != 0 {
            return Err(ModelError::InvalidConfig(
                "attention head count must be non-zero and divisible by KV head count",
            ));
        }
        if key_dim == 0 || value_dim == 0 {
            return Err(ModelError::InvalidConfig(
                "attention key/value dimensions must be non-zero",
            ));
        }
        if embedding_length != head_count * key_dim {
            return Err(ModelError::InvalidConfig(
                "embedding length must equal query head count times key length",
            ));
        }

        let config = Self {
            block_count: metadata_usize(gguf, "llama.block_count")?,
            embedding_length,
            feed_forward_length: metadata_usize(gguf, "llama.feed_forward_length")?,
            attention_head_count: head_count,
            attention_head_count_kv: head_count_kv,
            attention_key_length: key_dim,
            attention_value_length: value_dim,
            attention_layer_norm_rms_epsilon: metadata_f32(
                gguf,
                "llama.attention.layer_norm_rms_epsilon",
            )?,
            rope_freq_base: metadata_f32(gguf, "llama.rope.freq_base")?,
        };

        if config.block_count == 0 || config.feed_forward_length == 0 {
            return Err(ModelError::InvalidConfig(
                "block count and feed-forward length must be non-zero",
            ));
        }
        if !config.attention_layer_norm_rms_epsilon.is_finite()
            || config.attention_layer_norm_rms_epsilon <= 0.0
        {
            return Err(ModelError::InvalidConfig(
                "RMSNorm epsilon must be finite and greater than zero",
            ));
        }
        if !config.rope_freq_base.is_finite() || config.rope_freq_base <= 0.0 {
            return Err(ModelError::InvalidConfig(
                "RoPE frequency base must be finite and greater than zero",
            ));
        }

        Ok(config)
    }
}

/// Borrowed tensors required by one LLaMA decoder block.
#[derive(Debug, Clone, Copy)]
pub struct LlamaLayerWeights<'a> {
    pub attention_norm: &'a [f32],
    pub q_proj: TensorView<'a>,
    pub k_proj: TensorView<'a>,
    pub v_proj: TensorView<'a>,
    pub o_proj: TensorView<'a>,
    pub ffn_norm: &'a [f32],
    pub gate_proj: TensorView<'a>,
    pub up_proj: TensorView<'a>,
    pub down_proj: TensorView<'a>,
}

/// Mutable key/value cache owned by one LLaMA decoding sequence.
///
/// Construct a separate cache for each independent generation stream; do not
/// share one mutable cache between concurrent requests.
#[derive(Debug, Clone)]
pub struct LlamaKvCache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    sequence_len: usize,
}

impl LlamaKvCache {
    pub fn new(config: &LlamaConfig) -> Self {
        let key_width = config.attention_head_count_kv * config.attention_key_length;
        let value_width = config.attention_head_count_kv * config.attention_value_length;
        Self {
            keys: vec![Vec::with_capacity(key_width); config.block_count],
            values: vec![Vec::with_capacity(value_width); config.block_count],
            sequence_len: 0,
        }
    }

    /// Creates a KV cache with per-layer capacity reserved for the full context.
    pub fn with_context_capacity(
        config: &LlamaConfig,
        context_tokens: usize,
    ) -> Result<Self, ModelError> {
        let key_width = config
            .attention_head_count_kv
            .checked_mul(config.attention_key_length)
            .ok_or(ModelError::InvalidConfig("LLaMA KV key width overflow"))?;
        let value_width = config
            .attention_head_count_kv
            .checked_mul(config.attention_value_length)
            .ok_or(ModelError::InvalidConfig("LLaMA KV value width overflow"))?;
        let key_capacity = key_width
            .checked_mul(context_tokens)
            .ok_or(ModelError::InvalidConfig("LLaMA KV key capacity overflow"))?;
        let value_capacity =
            value_width
                .checked_mul(context_tokens)
                .ok_or(ModelError::InvalidConfig(
                    "LLaMA KV value capacity overflow",
                ))?;

        Ok(Self {
            keys: (0..config.block_count)
                .map(|_| Vec::with_capacity(key_capacity))
                .collect(),
            values: (0..config.block_count)
                .map(|_| Vec::with_capacity(value_capacity))
                .collect(),
            sequence_len: 0,
        })
    }

    pub fn sequence_len(&self) -> usize {
        self.sequence_len
    }

    pub fn clear(&mut self) {
        for values in &mut self.keys {
            values.clear();
        }
        for values in &mut self.values {
            values.clear();
        }
        self.sequence_len = 0;
    }

    pub(crate) fn validate(&self, config: &LlamaConfig) -> Result<(), ModelError> {
        if self.keys.len() != config.block_count || self.values.len() != config.block_count {
            return Err(ModelError::InvalidConfig(
                "KV cache layer count does not match model",
            ));
        }
        let key_width = config.attention_head_count_kv * config.attention_key_length;
        let value_width = config.attention_head_count_kv * config.attention_value_length;
        for layer in 0..config.block_count {
            if self.keys[layer].len() != self.sequence_len * key_width
                || self.values[layer].len() != self.sequence_len * value_width
            {
                return Err(ModelError::InvalidConfig(
                    "KV cache tensor length is inconsistent",
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn append_layer(&mut self, layer: usize, key: &[f32], value: &[f32]) {
        self.keys[layer].extend_from_slice(key);
        self.values[layer].extend_from_slice(value);
    }

    pub(crate) fn layer(&self, layer: usize) -> (&[f32], &[f32]) {
        (&self.keys[layer], &self.values[layer])
    }

    pub(crate) fn advance(&mut self) {
        self.sequence_len += 1;
    }
}

/// Executes one LLaMA-style decoder block.
///
/// M4 intentionally supports a single-token sequence. KV caching and
/// multi-token causal attention are introduced in M5.
pub fn forward_block<B: Backend>(
    backend: &B,
    config: &LlamaConfig,
    weights: &LlamaLayerWeights<'_>,
    input: &[f32],
    position: usize,
) -> Result<Vec<f32>, ModelError> {
    validate_layer_shapes(config, weights, input)?;

    let hidden = config.embedding_length;
    let q_len = config.attention_head_count * config.attention_key_length;
    let kv_len = config.attention_head_count_kv * config.attention_key_length;
    let value_len = config.attention_head_count_kv * config.attention_value_length;

    let mut norm = vec![0.0_f32; hidden];
    backend.rms_norm(
        input,
        weights.attention_norm,
        &mut norm,
        config.attention_layer_norm_rms_epsilon,
    )?;

    let mut q = vec![0.0_f32; q_len];
    let mut k = vec![0.0_f32; kv_len];
    let mut v = vec![0.0_f32; value_len];
    backend.matvec(weights.q_proj, &norm, &mut q)?;
    backend.matvec(weights.k_proj, &norm, &mut k)?;
    backend.matvec(weights.v_proj, &norm, &mut v)?;

    for head in 0..config.attention_head_count {
        let start = head * config.attention_key_length;
        backend.rope(
            &mut q[start..start + config.attention_key_length],
            position,
            config.attention_key_length,
            config.rope_freq_base,
        )?;
    }
    for head in 0..config.attention_head_count_kv {
        let start = head * config.attention_key_length;
        backend.rope(
            &mut k[start..start + config.attention_key_length],
            position,
            config.attention_key_length,
            config.rope_freq_base,
        )?;
    }

    let mut attention_output = vec![0.0_f32; hidden];
    let heads_per_kv = config.attention_head_count / config.attention_head_count_kv;
    let scale = (config.attention_key_length as f32).sqrt().recip();

    for q_head in 0..config.attention_head_count {
        let kv_head = q_head / heads_per_kv;
        let q_start = q_head * config.attention_key_length;
        let k_start = kv_head * config.attention_key_length;
        let v_start = kv_head * config.attention_value_length;

        let mut score = 0.0_f32;
        for index in 0..config.attention_key_length {
            score += q[q_start + index] * k[k_start + index];
        }
        let mut probabilities = [score * scale];
        backend.softmax(&mut probabilities)?;

        let output_start = q_head * config.attention_value_length;
        for index in 0..config.attention_value_length {
            attention_output[output_start + index] = probabilities[0] * v[v_start + index];
        }
    }

    let mut projected = vec![0.0_f32; hidden];
    backend.matvec(weights.o_proj, &attention_output, &mut projected)?;
    backend.residual_add(&mut projected, input)?;

    backend.rms_norm(
        &projected,
        weights.ffn_norm,
        &mut norm,
        config.attention_layer_norm_rms_epsilon,
    )?;

    let mut gate = vec![0.0_f32; config.feed_forward_length];
    let mut up = vec![0.0_f32; config.feed_forward_length];
    backend.matvec(weights.gate_proj, &norm, &mut gate)?;
    backend.matvec(weights.up_proj, &norm, &mut up)?;
    backend.silu_mul(&mut gate, &up)?;

    let mut output = vec![0.0_f32; hidden];
    backend.matvec(weights.down_proj, &gate, &mut output)?;
    backend.residual_add(&mut output, &projected)?;

    Ok(output)
}
pub fn forward_block_with_cache<B: Backend>(
    backend: &B,
    config: &LlamaConfig,
    weights: &LlamaLayerWeights<'_>,
    input: &[f32],
    position: usize,
    cache: &mut LlamaKvCache,
    layer_index: usize,
) -> Result<Vec<f32>, ModelError> {
    validate_layer_shapes(config, weights, input)?;
    if position != cache.sequence_len() {
        return Err(ModelError::InvalidConfig(
            "token position must equal the current KV cache sequence length",
        ));
    }

    let hidden = config.embedding_length;
    let q_len = config.attention_head_count * config.attention_key_length;
    let kv_len = config.attention_head_count_kv * config.attention_key_length;
    let value_len = config.attention_head_count_kv * config.attention_value_length;

    let mut norm = vec![0.0_f32; hidden];
    backend.rms_norm(
        input,
        weights.attention_norm,
        &mut norm,
        config.attention_layer_norm_rms_epsilon,
    )?;

    let mut q = vec![0.0_f32; q_len];
    let mut k = vec![0.0_f32; kv_len];
    let mut v = vec![0.0_f32; value_len];
    backend.matvec(weights.q_proj, &norm, &mut q)?;
    backend.matvec(weights.k_proj, &norm, &mut k)?;
    backend.matvec(weights.v_proj, &norm, &mut v)?;

    for head in 0..config.attention_head_count {
        let start = head * config.attention_key_length;
        backend.rope(
            &mut q[start..start + config.attention_key_length],
            position,
            config.attention_key_length,
            config.rope_freq_base,
        )?;
    }
    for head in 0..config.attention_head_count_kv {
        let start = head * config.attention_key_length;
        backend.rope(
            &mut k[start..start + config.attention_key_length],
            position,
            config.attention_key_length,
            config.rope_freq_base,
        )?;
    }

    cache.append_layer(layer_index, &k, &v);
    let (cached_keys, cached_values) = cache.layer(layer_index);
    let sequence_len = cache.sequence_len() + 1;
    let heads_per_kv = config.attention_head_count / config.attention_head_count_kv;
    let scale = (config.attention_key_length as f32).sqrt().recip();
    let mut attention_output = vec![0.0_f32; hidden];

    for q_head in 0..config.attention_head_count {
        let kv_head = q_head / heads_per_kv;
        let q_start = q_head * config.attention_key_length;
        let mut scores = vec![0.0_f32; sequence_len];

        for token in 0..sequence_len {
            let k_start = token * kv_len + kv_head * config.attention_key_length;
            scores[token] = q[q_start..q_start + config.attention_key_length]
                .iter()
                .zip(&cached_keys[k_start..k_start + config.attention_key_length])
                .map(|(a, b)| a * b)
                .sum::<f32>()
                * scale;
        }

        backend.softmax(&mut scores)?;
        let output_start = q_head * config.attention_value_length;
        for token in 0..sequence_len {
            let v_start = token * value_len + kv_head * config.attention_value_length;
            for index in 0..config.attention_value_length {
                attention_output[output_start + index] +=
                    scores[token] * cached_values[v_start + index];
            }
        }
    }

    let mut projected = vec![0.0_f32; hidden];
    backend.matvec(weights.o_proj, &attention_output, &mut projected)?;
    backend.residual_add(&mut projected, input)?;
    backend.rms_norm(
        &projected,
        weights.ffn_norm,
        &mut norm,
        config.attention_layer_norm_rms_epsilon,
    )?;

    let mut gate = vec![0.0_f32; config.feed_forward_length];
    let mut up = vec![0.0_f32; config.feed_forward_length];
    backend.matvec(weights.gate_proj, &norm, &mut gate)?;
    backend.matvec(weights.up_proj, &norm, &mut up)?;
    backend.silu_mul(&mut gate, &up)?;

    let mut output = vec![0.0_f32; hidden];
    backend.matvec(weights.down_proj, &gate, &mut output)?;
    backend.residual_add(&mut output, &projected)?;
    Ok(output)
}

fn validate_layer_shapes(
    config: &LlamaConfig,
    weights: &LlamaLayerWeights<'_>,
    input: &[f32],
) -> Result<(), ModelError> {
    if input.len() != config.embedding_length {
        return Err(ModelError::ShapeMismatch(format!(
            "input has {} elements; expected {}",
            input.len(),
            config.embedding_length
        )));
    }
    if weights.attention_norm.len() != config.embedding_length
        || weights.ffn_norm.len() != config.embedding_length
    {
        return Err(ModelError::ShapeMismatch(
            "RMSNorm weights do not match embedding length".into(),
        ));
    }

    let expected = [
        (
            "q_proj",
            weights.q_proj,
            config.embedding_length,
            config.embedding_length,
        ),
        (
            "k_proj",
            weights.k_proj,
            config.attention_head_count_kv * config.attention_key_length,
            config.embedding_length,
        ),
        (
            "v_proj",
            weights.v_proj,
            config.attention_head_count_kv * config.attention_value_length,
            config.embedding_length,
        ),
        (
            "o_proj",
            weights.o_proj,
            config.embedding_length,
            config.embedding_length,
        ),
        (
            "gate_proj",
            weights.gate_proj,
            config.feed_forward_length,
            config.embedding_length,
        ),
        (
            "up_proj",
            weights.up_proj,
            config.feed_forward_length,
            config.embedding_length,
        ),
        (
            "down_proj",
            weights.down_proj,
            config.embedding_length,
            config.feed_forward_length,
        ),
    ];

    for (name, tensor, rows, columns) in expected {
        if tensor.shape != [rows, columns] {
            return Err(ModelError::ShapeMismatch(format!(
                "{name} has shape {:?}; expected [{rows}, {columns}]",
                tensor.shape
            )));
        }
    }

    if config.attention_value_length != config.attention_key_length {
        return Err(ModelError::InvalidConfig(
            "M4 attention path currently requires equal key and value head dimensions",
        ));
    }

    Ok(())
}

fn metadata_usize(gguf: &GgufFile, key: &'static str) -> Result<usize, ModelError> {
    let value = gguf
        .metadata_value(key)
        .ok_or(ModelError::MissingMetadata(key))?;
    let raw = match value {
        MetadataValue::Uint32(value) => u64::from(*value),
        MetadataValue::Uint64(value) => *value,
        MetadataValue::Int32(value) if *value >= 0 => *value as u64,
        MetadataValue::Int64(value) if *value >= 0 => *value as u64,
        _ => {
            return Err(ModelError::InvalidMetadata {
                key,
                expected: "unsigned integer",
            });
        }
    };
    usize::try_from(raw).map_err(|_| ModelError::InvalidConfig("metadata value exceeds usize"))
}

fn metadata_f32(gguf: &GgufFile, key: &'static str) -> Result<f32, ModelError> {
    let value = gguf
        .metadata_value(key)
        .ok_or(ModelError::MissingMetadata(key))?;
    match value {
        MetadataValue::Float32(value) => Ok(*value),
        MetadataValue::Float64(value) if value.is_finite() => Ok(*value as f32),
        _ => Err(ModelError::InvalidMetadata {
            key,
            expected: "FLOAT32 or FLOAT64",
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qarnyx_core::DType;

    fn config() -> LlamaConfig {
        LlamaConfig {
            block_count: 1,
            embedding_length: 4,
            feed_forward_length: 8,
            attention_head_count: 2,
            attention_head_count_kv: 1,
            attention_key_length: 2,
            attention_value_length: 2,
            attention_layer_norm_rms_epsilon: 1.0e-5,
            rope_freq_base: 10_000.0,
        }
    }

    fn f32_view(bytes: &[u8], rows: usize, cols: usize) -> TensorView<'_> {
        TensorView::new(
            bytes,
            Box::leak(vec![rows, cols].into_boxed_slice()),
            DType::F32,
        )
    }

    fn identity_bytes(size: usize) -> Vec<u8> {
        let mut values = vec![0.0_f32; size * size];
        for index in 0..size {
            values[index * size + index] = 1.0;
        }
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn validates_llama_metadata_shape() {
        let metadata = vec![
            ("llama.block_count", MetadataValue::Uint32(2)),
            ("llama.embedding_length", MetadataValue::Uint32(4096)),
            ("llama.feed_forward_length", MetadataValue::Uint32(11008)),
            ("llama.attention.head_count", MetadataValue::Uint32(32)),
            ("llama.attention.head_count_kv", MetadataValue::Uint32(8)),
            ("llama.attention.key_length", MetadataValue::Uint32(128)),
            ("llama.attention.value_length", MetadataValue::Uint32(128)),
            (
                "llama.attention.layer_norm_rms_epsilon",
                MetadataValue::Float32(1.0e-5),
            ),
            ("llama.rope.freq_base", MetadataValue::Float32(10_000.0)),
        ];

        let gguf = GgufFile {
            version: 3,
            metadata: metadata
                .into_iter()
                .map(|(key, value)| qarnyx_gguf::MetadataEntry {
                    key: key.into(),
                    value,
                })
                .collect(),
            tensors: Vec::new(),
            alignment: 32,
            tensor_data_offset: 0,
            file_size: 0,
        };

        let config = LlamaConfig::from_gguf(&gguf).unwrap();
        assert_eq!(config.block_count, 2);
        assert_eq!(config.embedding_length, 4096);
        assert_eq!(config.attention_head_count_kv, 8);
    }

    #[test]
    fn rejects_non_divisible_gqa_heads() {
        let mut gguf = GgufFile {
            version: 3,
            metadata: Vec::new(),
            tensors: Vec::new(),
            alignment: 32,
            tensor_data_offset: 0,
            file_size: 0,
        };
        let fields = [
            ("llama.block_count", MetadataValue::Uint32(1)),
            ("llama.embedding_length", MetadataValue::Uint32(4)),
            ("llama.feed_forward_length", MetadataValue::Uint32(8)),
            ("llama.attention.head_count", MetadataValue::Uint32(3)),
            ("llama.attention.head_count_kv", MetadataValue::Uint32(2)),
            ("llama.attention.key_length", MetadataValue::Uint32(2)),
            ("llama.attention.value_length", MetadataValue::Uint32(2)),
            (
                "llama.attention.layer_norm_rms_epsilon",
                MetadataValue::Float32(1.0e-5),
            ),
            ("llama.rope.freq_base", MetadataValue::Float32(10_000.0)),
        ];
        gguf.metadata = fields
            .into_iter()
            .map(|(key, value)| qarnyx_gguf::MetadataEntry {
                key: key.into(),
                value,
            })
            .collect();

        assert!(matches!(
            LlamaConfig::from_gguf(&gguf),
            Err(ModelError::InvalidConfig(_))
        ));
    }

    #[test]
    fn kv_cache_reserves_full_context_capacity() {
        let config = LlamaConfig {
            block_count: 2,
            embedding_length: 4,
            feed_forward_length: 8,
            attention_head_count: 2,
            attention_head_count_kv: 1,
            attention_key_length: 2,
            attention_value_length: 3,
            attention_layer_norm_rms_epsilon: 1.0e-5,
            rope_freq_base: 10_000.0,
        };
        let cache = LlamaKvCache::with_context_capacity(&config, 16).unwrap();
        assert_eq!(cache.keys[0].capacity(), 32);
        assert_eq!(cache.values[0].capacity(), 48);
        assert!(LlamaKvCache::with_context_capacity(&config, usize::MAX).is_err());
    }

    #[test]
    fn kv_cache_accumulates_incremental_tokens() {
        let config = config();
        let identity = identity_bytes(4);
        let k_zero = vec![0_u8; 2 * 4 * 4];
        let v_zero = vec![0_u8; 2 * 4 * 4];
        let zero_8x4 = vec![0_u8; 8 * 4 * 4];
        let zero_4x8 = vec![0_u8; 4 * 8 * 4];
        let norm = [1.0_f32; 4];

        let weights = LlamaLayerWeights {
            attention_norm: &norm,
            q_proj: f32_view(&identity, 4, 4),
            k_proj: f32_view(&k_zero, 2, 4),
            v_proj: f32_view(&v_zero, 2, 4),
            o_proj: f32_view(&identity, 4, 4),
            ffn_norm: &norm,
            gate_proj: f32_view(&zero_8x4, 8, 4),
            up_proj: f32_view(&zero_8x4, 8, 4),
            down_proj: f32_view(&zero_4x8, 4, 8),
        };

        let backend = qarnyx_cpu::CpuBackend::new(1).unwrap();
        let mut cache = LlamaKvCache::new(&config);
        let first = forward_block_with_cache(
            &backend,
            &config,
            &weights,
            &[1.0, 2.0, 3.0, 4.0],
            0,
            &mut cache,
            0,
        )
        .unwrap();
        cache.advance();
        let second = forward_block_with_cache(
            &backend,
            &config,
            &weights,
            &[4.0, 3.0, 2.0, 1.0],
            1,
            &mut cache,
            0,
        )
        .unwrap();

        assert_eq!(first, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(second, [4.0, 3.0, 2.0, 1.0]);
        cache.advance();
        assert_eq!(cache.sequence_len(), 2);
        assert_eq!(cache.keys[0].len(), 2 * config.attention_key_length);
        assert_eq!(cache.values[0].len(), 2 * config.attention_value_length);
    }

    #[test]
    fn forward_block_identity_weights() {
        let config = config();
        let identity = identity_bytes(4);
        let k_zero = vec![0_u8; 2 * 4 * 4];
        let v_zero = vec![0_u8; 2 * 4 * 4];
        let zero_8x4 = vec![0_u8; 8 * 4 * 4];
        let zero_4x8 = vec![0_u8; 4 * 8 * 4];
        let norm = [1.0_f32; 4];

        let weights = LlamaLayerWeights {
            attention_norm: &norm,
            q_proj: f32_view(&identity, 4, 4),
            k_proj: f32_view(&k_zero, 2, 4),
            v_proj: f32_view(&v_zero, 2, 4),
            o_proj: f32_view(&identity, 4, 4),
            ffn_norm: &norm,
            gate_proj: f32_view(&zero_8x4, 8, 4),
            up_proj: f32_view(&zero_8x4, 8, 4),
            down_proj: f32_view(&zero_4x8, 4, 8),
        };

        let output = forward_block(
            &qarnyx_cpu::CpuBackend::new(1).unwrap(),
            &config,
            &weights,
            &[1.0, 2.0, 3.0, 4.0],
            0,
        )
        .unwrap();
        assert_eq!(output, [1.0, 2.0, 3.0, 4.0]);
    }
}
