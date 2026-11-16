//! Autoregressive generation orchestration for Qarnyx model execution.

use qarnyx_core::Backend;
use qarnyx_models::{LlamaKvCache, LlamaModel, ModelError};
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

    #[error("logits contain non-finite values")]
    NonFiniteLogits,
}

/// Sampling and generation controls.
#[derive(Debug, Clone, Copy)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
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
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(GenerationError::InvalidConfig("temperature must be finite and >= 0"));
        }
        if self.top_p <= 0.0 || self.top_p > 1.0 || !self.top_p.is_finite() {
            return Err(GenerationError::InvalidConfig("top_p must be in (0, 1]"));
        }
        Ok(())
    }
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

        let mut cache = LlamaKvCache::new(&model.config);
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
