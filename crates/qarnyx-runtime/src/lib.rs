//! Autoregressive generation orchestration for Qarnyx model execution.

use qarnyx_core::Backend;
use qarnyx_models::{LlamaKvCache, LlamaModel, ModelError, Qwen35Cache, Qwen35Model, Qwen35Profile};
use qarnyx_tokenizer::{GgufTokenizer, TokenizerError};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GenerationError {
    #[error("model execution failed: {0}")]
    Model(#[from] ModelError),

    #[error("tokenizer failed: {0}")]
    Tokenizer(#[from] TokenizerError),

    #[error("generation requires at least one prompt token")]
    EmptyPrompt,

    #[error("generation configuration is invalid: {0}")]
    InvalidConfig(&'static str),

    #[error(
        "requested sequence ({prompt_tokens} prompt + {max_new_tokens} new tokens) exceeds context limit {limit}"
    )]
    ContextLimitExceeded {
        prompt_tokens: usize,
        max_new_tokens: usize,
        limit: usize,
    },

    #[error(
        "estimated session cache is {estimated_bytes} bytes, exceeding the configured budget of {limit_bytes} bytes"
    )]
    CacheBudgetExceeded {
        estimated_bytes: usize,
        limit_bytes: usize,
    },

    #[error("logits contain non-finite values")]
    NonFiniteLogits,
}

/// Sampling and generation controls.
#[derive(Debug, Clone, Copy)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    /// Maximum combined prompt and generated sequence length.
    pub max_context_tokens: usize,
    /// Maximum retained per-session cache/workspace memory, in MiB.
    pub max_cache_mib: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            max_context_tokens: 4096,
            max_cache_mib: 512,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            seed: 0x51_4e_52_59_58,
        }
    }
}

impl GenerationConfig {
    fn validate(&self) -> Result<(), GenerationError> {
        if self.max_new_tokens == 0 {
            return Err(GenerationError::InvalidConfig("max_new_tokens must be non-zero"));
        }
        if self.max_context_tokens == 0 {
            return Err(GenerationError::InvalidConfig(
                "max_context_tokens must be non-zero",
            ));
        }
        if self.max_cache_mib == 0
            || self.max_cache_mib.checked_mul(1024 * 1024).is_none()
        {
            return Err(GenerationError::InvalidConfig(
                "max_cache_mib must be non-zero and fit in bytes",
            ));
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(GenerationError::InvalidConfig("temperature must be finite and >= 0"));
        }
        if self.top_p <= 0.0 || self.top_p > 1.0 || !self.top_p.is_finite() {
            return Err(GenerationError::InvalidConfig("top_p must be in (0, 1]"));
        }
        Ok(())
    }

    fn validate_context(&self, prompt_tokens: usize) -> Result<(), GenerationError> {
        if prompt_tokens > self.max_context_tokens
            || self.max_new_tokens > self.max_context_tokens - prompt_tokens
        {
            return Err(GenerationError::ContextLimitExceeded {
                prompt_tokens,
                max_new_tokens: self.max_new_tokens,
                limit: self.max_context_tokens,
            });
        }
        Ok(())
    }

    fn validate_cache_budget(&self, estimated_bytes: usize) -> Result<(), GenerationError> {
        let limit_bytes = self.max_cache_mib.checked_mul(1024 * 1024).ok_or(
            GenerationError::InvalidConfig("max_cache_mib must fit in bytes"),
        )?;
        if estimated_bytes > limit_bytes {
            return Err(GenerationError::CacheBudgetExceeded {
                estimated_bytes,
                limit_bytes,
            });
        }
        Ok(())
    }
}

fn estimate_llama_cache_bytes(
    model: &LlamaModel<'_>,
    context_tokens: usize,
) -> Result<usize, GenerationError> {
    let config = &model.config;
    let key_width = config
        .attention_head_count_kv
        .checked_mul(config.attention_key_length)
        .ok_or(GenerationError::InvalidConfig("LLaMA KV key width overflow"))?;
    let value_width = config
        .attention_head_count_kv
        .checked_mul(config.attention_value_length)
        .ok_or(GenerationError::InvalidConfig("LLaMA KV value width overflow"))?;
    key_width
        .checked_add(value_width)
        .and_then(|width| width.checked_mul(config.block_count))
        .and_then(|floats| floats.checked_mul(context_tokens))
        .and_then(|floats| floats.checked_mul(std::mem::size_of::<f32>()))
        .ok_or(GenerationError::InvalidConfig("LLaMA KV cache size overflow"))
}

/// Result of an autoregressive generation run.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationResult {
    pub token_ids: Vec<usize>,
    pub text: String,
}

pub struct LlamaGenerator {
    rng_state: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use qarnyx_core::{DType, TensorView};
    use qarnyx_models::{LlamaConfig, LlamaLayerStorage, LlamaModel};
    use qarnyx_tokenizer::GgufTokenizer;

    fn matrix(data: &[f32], rows: usize, cols: usize) -> TensorView<'_> {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for value in data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        TensorView::new(bytes, Box::leak(vec![rows, cols].into_boxed_slice()), DType::F32)
    }

    #[test]
    fn end_to_end_greedy_generation_uses_kv_cache() {
        let identity = [1.0_f32, 0.0, 0.0, 1.0];
        let zero_2x2 = [0.0_f32; 4];
        let zero_4x2 = [0.0_f32; 8];
        let zero_2x4 = [0.0_f32; 8];

        let layer = LlamaLayerStorage {
            attention_norm: vec![1.0, 1.0],
            q_proj: matrix(&identity, 2, 2),
            k_proj: matrix(&zero_2x2, 2, 2),
            v_proj: matrix(&zero_2x2, 2, 2),
            o_proj: matrix(&identity, 2, 2),
            ffn_norm: vec![1.0, 1.0],
            gate_proj: matrix(&zero_4x2, 4, 2),
            up_proj: matrix(&zero_4x2, 4, 2),
            down_proj: matrix(&zero_2x4, 2, 4),
        };

        let embedding = matrix(&identity, 2, 2);
        let output = matrix(&identity, 2, 2);
        let model = LlamaModel {
            config: LlamaConfig {
                block_count: 1,
                embedding_length: 2,
                feed_forward_length: 4,
                attention_head_count: 1,
                attention_head_count_kv: 1,
                attention_key_length: 2,
                attention_value_length: 2,
                attention_layer_norm_rms_epsilon: 1.0e-5,
                rope_freq_base: 10_000.0,
            },
            layers: vec![layer],
            token_embedding: embedding,
            output_norm: vec![1.0, 1.0],
            output,
        };

        let tokenizer = GgufTokenizer {
            model: "test".into(),
            pre: None,
            tokens: vec!["a".into(), "b".into()],
            scores: vec![0.0, 0.0],
            merges: Vec::new(),
            bos_token_id: None,
            eos_token_id: None,
        };
        let backend = qarnyx_cpu::CpuBackend::new(1).unwrap();
        let mut generator = LlamaGenerator::new(123);

        let result = generator
            .generate(&model, &tokenizer, &backend, "a", GenerationConfig {
                max_new_tokens: 3,
                max_context_tokens: 8,
                max_cache_mib: 512,
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                seed: 123,
            })
            .unwrap();

        assert_eq!(result.token_ids, [0, 0, 0, 0]);
        assert_eq!(result.text, "aaaa");
    }


    #[test]
    fn greedy_sampling_is_deterministic() {
        let mut generator = LlamaGenerator::new(7);
        let config = GenerationConfig {
            max_new_tokens: 1,
            max_context_tokens: 8,
            max_cache_mib: 512,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 7,
        };
        assert_eq!(generator.sample(&[1.0, 4.0, 2.0], config).unwrap(), 1);
    }

    #[test]
    fn seeded_sampling_repeats() {
        let config = GenerationConfig {
            max_new_tokens: 1,
            max_context_tokens: 8,
            max_cache_mib: 512,
            temperature: 1.0,
            top_k: 3,
            top_p: 1.0,
            seed: 42,
        };
        let logits = [0.0, 1.0, 2.0];
        let mut a = LlamaGenerator::new(config.seed);
        let mut b = LlamaGenerator::new(config.seed);
        let first_a = a.sample(&logits, config).unwrap();
        let first_b = b.sample(&logits, config).unwrap();
        assert_eq!(first_a, first_b);
    }

    #[test]
    fn rejects_invalid_generation_config() {
        let config = GenerationConfig {
            max_new_tokens: 0,
            ..GenerationConfig::default()
        };
        assert!(config.validate().is_err());

        let zero_context = GenerationConfig {
            max_context_tokens: 0,
            ..GenerationConfig::default()
        };
        assert!(zero_context.validate().is_err());

        let zero_budget = GenerationConfig {
            max_cache_mib: 0,
            ..GenerationConfig::default()
        };
        assert!(zero_budget.validate().is_err());
    }

    #[test]
    fn cache_budget_accepts_exact_fit_and_rejects_excess() {
        let config = GenerationConfig {
            max_cache_mib: 1,
            ..GenerationConfig::default()
        };
        assert!(config.validate_cache_budget(1024 * 1024).is_ok());
        assert!(matches!(
            config.validate_cache_budget(1024 * 1024 + 1),
            Err(GenerationError::CacheBudgetExceeded {
                estimated_bytes,
                limit_bytes,
            }) if estimated_bytes == 1024 * 1024 + 1
                && limit_bytes == 1024 * 1024
        ));
    }

    #[test]
    fn context_limit_accepts_exact_fit_and_rejects_overflow() {
        let config = GenerationConfig {
            max_new_tokens: 4,
            max_context_tokens: 8,
            ..GenerationConfig::default()
        };

        assert!(config.validate_context(4).is_ok());
        assert!(matches!(
            config.validate_context(5),
            Err(GenerationError::ContextLimitExceeded {
                prompt_tokens: 5,
                max_new_tokens: 4,
                limit: 8,
            })
        ));
    }
}


impl LlamaGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng_state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Generates new tokens from an already-tokenized prompt.
    pub fn generate_ids<B: Backend>(
        &mut self,
        model: &LlamaModel<'_>,
        tokenizer: &GgufTokenizer,
        backend: &B,
        prompt: &[usize],
        config: GenerationConfig,
    ) -> Result<GenerationResult, GenerationError> {
        config.validate()?;
        if prompt.is_empty() {
            return Err(GenerationError::EmptyPrompt);
        }
        config.validate_context(prompt.len())?;
        let estimated_cache_bytes =
            estimate_llama_cache_bytes(model, config.max_context_tokens)?;
        config.validate_cache_budget(estimated_cache_bytes)?;

        let mut cache =
            LlamaKvCache::with_context_capacity(&model.config, config.max_context_tokens)?;
        let mut generated = prompt.to_vec();
        let mut logits = Vec::new();

        for (position, &token_id) in prompt.iter().enumerate() {
            logits = model.forward_logits(backend, token_id, position, &mut cache)?;
        }

        for _ in 0..config.max_new_tokens {
            let next = self.sample(&logits, config)?;
            generated.push(next);
            if tokenizer.eos_token_id == Some(next) {
                break;
            }

            let position = cache.sequence_len();
            logits = model.forward_logits(backend, next, position, &mut cache)?;
        }

        let text = tokenizer.decode(&generated)?;
        Ok(GenerationResult {
            token_ids: generated,
            text,
        })
    }

    /// Tokenizes a prompt and runs autoregressive generation.
    pub fn generate<B: Backend>(
        &mut self,
        model: &LlamaModel<'_>,
        tokenizer: &GgufTokenizer,
        backend: &B,
        prompt: &str,
        config: GenerationConfig,
    ) -> Result<GenerationResult, GenerationError> {
        let prompt_tokens = tokenizer.encode(prompt)?;
        self.generate_ids(model, tokenizer, backend, &prompt_tokens, config)
    }

    fn sample(&mut self, logits: &[f32], config: GenerationConfig) -> Result<usize, GenerationError> {
        if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
            return Err(GenerationError::NonFiniteLogits);
        }

        let mut candidates: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();

        if config.temperature == 0.0 {
            return candidates
                .into_iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(id, _)| id)
                .ok_or(GenerationError::NonFiniteLogits);
        }

        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        if config.top_k > 0 && config.top_k < candidates.len() {
            candidates.truncate(config.top_k);
        }

        let temperature = config.temperature;
        let maximum = candidates[0].1;
        let mut weights = Vec::with_capacity(candidates.len());
        let mut sum = 0.0_f64;
        for &(_, logit) in &candidates {
            let weight = (f64::from((logit - maximum) / temperature)).exp();
            sum += weight;
            weights.push(weight);
        }

        let mut probabilities: Vec<(usize, f64)> = candidates
            .iter()
            .zip(weights)
            .map(|(&(id, _), weight)| (id, weight / sum))
            .collect();

        if config.top_p < 1.0 {
            let mut cumulative = 0.0;
            let mut keep = 0;
            for &(_, probability) in &probabilities {
                cumulative += probability;
                keep += 1;
                if cumulative >= f64::from(config.top_p) {
                    break;
                }
            }
            probabilities.truncate(keep.max(1));
        }

        let total = probabilities.iter().map(|(_, p)| p).sum::<f64>();
        let threshold = self.next_unit() * total;
        let mut cumulative = 0.0;
        for (token_id, probability) in probabilities {
            cumulative += probability;
            if cumulative >= threshold {
                return Ok(token_id);
            }
        }

        Ok(candidates[0].0)
    }

    fn next_unit(&mut self) -> f64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        (x as f64) / (u64::MAX as f64)
    }
}


/// Autoregressive generator for the Qwen3.5 hybrid decoder.
pub struct Qwen35Generator {
    rng_state: u64,
}

impl Qwen35Generator {
    pub fn new(seed: u64) -> Self {
        Self { rng_state: if seed == 0 { 1 } else { seed } }
    }

    pub fn generate_ids<B: Backend>(
        &mut self,
        model: &Qwen35Model<'_>,
        tokenizer: &GgufTokenizer,
        backend: &B,
        prompt: &[usize],
        config: GenerationConfig,
    ) -> Result<GenerationResult, GenerationError> {
        config.validate()?;
        if prompt.is_empty() {
            return Err(GenerationError::EmptyPrompt);
        }
        config.validate_context(prompt.len())?;
        let estimated_cache_bytes = model
            .config
            .estimated_cache_bytes(config.max_context_tokens)
            .ok_or(GenerationError::InvalidConfig(
                "Qwen3.5 session cache size overflow or invalid layer map",
            ))?;
        config.validate_cache_budget(estimated_cache_bytes)?;
        let mut cache =
            Qwen35Cache::with_context_capacity(&model.config, config.max_context_tokens)?;
        let mut generated = prompt.to_vec();
        let mut logits = Vec::new();
        let profiling = std::env::var_os("QARNYX_PROFILE").is_some();
        let mut profile = Qwen35Profile::default();

        for (position, &token_id) in prompt.iter().enumerate() {
            if profiling {
                logits = model.forward_logits_profiled(
                    backend, token_id, position, &mut cache, &mut profile,
                )?;
            } else {
                logits = model.forward_logits(backend, token_id, position, &mut cache)?;
            }
        }

        for _ in 0..config.max_new_tokens {
            let next = self.sample(&logits, config)?;
            generated.push(next);
            if tokenizer.eos_token_id == Some(next) {
                break;
            }
            let position = cache.sequence_len;
            if profiling {
                logits = model.forward_logits_profiled(
                    backend, next, position, &mut cache, &mut profile,
                )?;
            } else {
                logits = model.forward_logits(backend, next, position, &mut cache)?;
            }
        }

        if profiling {
            eprintln!("{}", profile.report());
        }

        Ok(GenerationResult {
            token_ids: generated.clone(),
            text: tokenizer.decode(&generated)?,
        })
    }

    pub fn generate<B: Backend>(
        &mut self,
        model: &Qwen35Model<'_>,
        tokenizer: &GgufTokenizer,
        backend: &B,
        prompt: &str,
        config: GenerationConfig,
    ) -> Result<GenerationResult, GenerationError> {
        let ids = tokenizer.encode(prompt)?;
        self.generate_ids(model, tokenizer, backend, &ids, config)
    }

    fn sample(&mut self, logits: &[f32], config: GenerationConfig) -> Result<usize, GenerationError> {
        if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
            return Err(GenerationError::NonFiniteLogits);
        }
        let mut candidates: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        if config.temperature == 0.0 {
            return candidates.into_iter().max_by(|a,b| a.1.total_cmp(&b.1))
                .map(|(id,_)| id).ok_or(GenerationError::NonFiniteLogits);
        }
        candidates.sort_by(|a,b| b.1.total_cmp(&a.1));
        if config.top_k > 0 && config.top_k < candidates.len() {
            candidates.truncate(config.top_k);
        }
        let max = candidates[0].1;
        let mut sum = 0.0f64;
        let mut weights = Vec::with_capacity(candidates.len());
        for &(_, logit) in &candidates {
            let w = f64::from((logit - max) / config.temperature).exp();
            sum += w;
            weights.push(w);
        }
        let mut probs: Vec<(usize,f64)> = candidates.iter().zip(weights)
            .map(|(&(id,_),w)|(id,w/sum)).collect();
        if config.top_p < 1.0 {
            let mut cumulative = 0.0;
            let mut keep = 0;
            for &(_, p) in &probs {
                cumulative += p;
                keep += 1;
                if cumulative >= f64::from(config.top_p) { break; }
            }
            probs.truncate(keep.max(1));
        }
        let total = probs.iter().map(|(_,p)| *p).sum::<f64>();
        let threshold = self.next_unit() * total;
        let mut cumulative = 0.0;
        for (id,p) in probs {
            cumulative += p;
            if cumulative >= threshold { return Ok(id); }
        }
        Ok(candidates[0].0)
    }

    fn next_unit(&mut self) -> f64 {
        let mut x = self.rng_state;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.rng_state = x;
        x as f64 / u64::MAX as f64
    }
}
