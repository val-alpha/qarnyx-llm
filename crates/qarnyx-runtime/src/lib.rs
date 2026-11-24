//! Autoregressive generation orchestration for Qarnyx model execution.

use qarnyx_core::Backend;
use qarnyx_models::{
    LlamaKvCache, LlamaModel, ModelError, Qwen35Cache, Qwen35Model, Qwen35Profile,
};
use qarnyx_tokenizer::{GgufTokenizer, TokenizerError};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GenerationError {
    #[error("model execution failed: {0}")]
    Model(#[from] ModelError),

    #[error("tokenizer failed: {0}")]
    Tokenizer(#[from] TokenizerError),

    #[error("generation requires at least one prompt token")]
    EmptyPrompt,

    #[error("generation cancelled")]
    Cancelled,

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

/// A clonable cancellation signal shared between a generation task and its caller.
#[derive(Debug, Clone)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
    parents: Vec<Arc<AtomicBool>>,
}

impl CancellationToken {
    /// Creates a token in the active (not cancelled) state.
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            parents: Vec::new(),
        }
    }

    /// Creates an independently cancellable child that is also cancelled when
    /// this token (or any of its ancestors) is cancelled.
    pub fn child_token(&self) -> Self {
        let mut parents = self.parents.clone();
        parents.push(Arc::clone(&self.cancelled));
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            parents,
        }
    }

    /// Requests cancellation. The generation task observes it at the next token boundary.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Returns whether this token or any parent has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
            || self
                .parents
                .iter()
                .any(|parent| parent.load(Ordering::Acquire))
    }

    fn check(&self) -> Result<(), GenerationError> {
        if self.is_cancelled() {
            Err(GenerationError::Cancelled)
        } else {
            Ok(())
        }
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
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
            return Err(GenerationError::InvalidConfig(
                "max_new_tokens must be non-zero",
            ));
        }
        if self.max_context_tokens == 0 {
            return Err(GenerationError::InvalidConfig(
                "max_context_tokens must be non-zero",
            ));
        }
        if self.max_cache_mib == 0 || self.max_cache_mib.checked_mul(1024 * 1024).is_none() {
            return Err(GenerationError::InvalidConfig(
                "max_cache_mib must be non-zero and fit in bytes",
            ));
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(GenerationError::InvalidConfig(
                "temperature must be finite and >= 0",
            ));
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
        let limit_bytes =
            self.max_cache_mib
                .checked_mul(1024 * 1024)
                .ok_or(GenerationError::InvalidConfig(
                    "max_cache_mib must fit in bytes",
                ))?;
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
        .ok_or(GenerationError::InvalidConfig(
            "LLaMA KV key width overflow",
        ))?;
    let value_width = config
        .attention_head_count_kv
        .checked_mul(config.attention_value_length)
        .ok_or(GenerationError::InvalidConfig(
            "LLaMA KV value width overflow",
        ))?;
    key_width
        .checked_add(value_width)
        .and_then(|width| width.checked_mul(config.block_count))
        .and_then(|floats| floats.checked_mul(context_tokens))
        .and_then(|floats| floats.checked_mul(std::mem::size_of::<f32>()))
        .ok_or(GenerationError::InvalidConfig(
            "LLaMA KV cache size overflow",
        ))
}

/// Result of an autoregressive generation run.
///
/// For backward compatibility, token_ids and text include the prompt followed
/// by generated tokens. GenerationEvent values contain generated tokens only.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationResult {
    pub token_ids: Vec<usize>,
    pub text: String,
}

/// Protocol-neutral events emitted by a generation stream.
///
/// Token events contain generated tokens only; prompt tokens and end-of-turn
/// markers are not emitted. A text delta can be empty while a byte-level
/// tokenizer is buffering an incomplete UTF-8 sequence. Any final lossy
/// replacement text is emitted as a TextDelta event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerationEvent {
    Token { token_id: usize, text_delta: String },
    TextDelta { text: String },
}

/// Sampling state for one generation stream.
///
/// Model weights and the backend may be shared across requests. Each concurrent
/// stream should own its generator; every generation call allocates a fresh,
/// request-local KV cache.
pub struct LlamaGenerator {
    rng_state: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use qarnyx_core::{Backend, BackendError, DType, TensorView};
    use qarnyx_models::{LlamaConfig, LlamaLayerStorage, LlamaModel};
    use qarnyx_tokenizer::GgufTokenizer;

    struct CancellingBackend {
        inner: qarnyx_cpu::CpuBackend,
        cancellation: CancellationToken,
    }

    impl Backend for CancellingBackend {
        fn name(&self) -> &'static str {
            self.inner.name()
        }

        fn matvec(
            &self,
            weights: TensorView<'_>,
            input: &[f32],
            output: &mut [f32],
        ) -> Result<(), BackendError> {
            self.inner.matvec(weights, input, output)?;
            self.cancellation.cancel();
            Ok(())
        }

        fn matmul(
            &self,
            weights: TensorView<'_>,
            input: &[f32],
            output: &mut [f32],
            m: usize,
            n: usize,
            k: usize,
        ) -> Result<(), BackendError> {
            self.inner.matmul(weights, input, output, m, n, k)
        }

        fn rms_norm(
            &self,
            input: &[f32],
            weight: &[f32],
            output: &mut [f32],
            epsilon: f32,
        ) -> Result<(), BackendError> {
            self.inner.rms_norm(input, weight, output, epsilon)
        }

        fn rope(
            &self,
            input: &mut [f32],
            position: usize,
            head_dim: usize,
            theta: f32,
        ) -> Result<(), BackendError> {
            self.inner.rope(input, position, head_dim, theta)
        }

        fn softmax(&self, input: &mut [f32]) -> Result<(), BackendError> {
            self.inner.softmax(input)
        }

        fn silu_mul(&self, gate: &mut [f32], up: &[f32]) -> Result<(), BackendError> {
            self.inner.silu_mul(gate, up)
        }

        fn residual_add(&self, target: &mut [f32], residual: &[f32]) -> Result<(), BackendError> {
            self.inner.residual_add(target, residual)
        }
    }

    fn matrix(data: &[f32], rows: usize, cols: usize) -> TensorView<'static> {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for value in data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        let shape: &'static [usize] = Box::leak(vec![rows, cols].into_boxed_slice());
        TensorView::new(bytes, shape, DType::F32)
    }

    fn tiny_llama_fixture() -> (LlamaModel<'static>, GgufTokenizer) {
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
            token_embedding: matrix(&identity, 2, 2),
            output_norm: vec![1.0, 1.0],
            output: matrix(&identity, 2, 2),
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

        (model, tokenizer)
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

        let config = GenerationConfig {
            max_new_tokens: 3,
            max_context_tokens: 8,
            max_cache_mib: 512,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 123,
        };
        let result = generator
            .generate(&model, &tokenizer, &backend, "a", config)
            .unwrap();

        assert_eq!(result.token_ids, [0, 0, 0, 0]);
        assert_eq!(result.text, "aaaa");

        let cancellation = CancellationToken::new();
        let cancelling_backend = CancellingBackend {
            inner: qarnyx_cpu::CpuBackend::new(1).unwrap(),
            cancellation: cancellation.clone(),
        };
        let mut cancelled_generator = LlamaGenerator::new(123);
        assert!(matches!(
            cancelled_generator.generate_with_cancellation(
                &model,
                &tokenizer,
                &cancelling_backend,
                "a",
                config,
                &cancellation,
            ),
            Err(GenerationError::Cancelled)
        ));
        assert!(cancellation.is_cancelled());
    }

    #[test]
    fn concurrent_generation_requests_keep_session_state_isolated() {
        use std::sync::Arc;

        let (model, tokenizer) = tiny_llama_fixture();
        let model = Arc::new(model);
        let tokenizer = Arc::new(tokenizer);
        let backend = Arc::new(qarnyx_cpu::CpuBackend::new(2).unwrap());
        let config = GenerationConfig {
            max_new_tokens: 3,
            max_context_tokens: 8,
            max_cache_mib: 1,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 123,
        };

        const SESSIONS: usize = 8;
        const REQUESTS_PER_SESSION: usize = 4;

        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(SESSIONS);
            for session_id in 0..SESSIONS {
                let model = Arc::clone(&model);
                let tokenizer = Arc::clone(&tokenizer);
                let backend = Arc::clone(&backend);

                handles.push(scope.spawn(move || {
                    let prompt_id = session_id % 2;
                    let expected_ids = vec![prompt_id; config.max_new_tokens + 1];
                    let expected_text = if prompt_id == 0 { "aaaa" } else { "bbbb" };
                    let mut generator = LlamaGenerator::new(session_id as u64 + 1);

                    for request_id in 0..REQUESTS_PER_SESSION {
                        let result = generator
                            .generate_ids(
                                model.as_ref(),
                                tokenizer.as_ref(),
                                backend.as_ref(),
                                &[prompt_id],
                                config,
                            )
                            .unwrap_or_else(|error| {
                                panic!("session {session_id}, request {request_id} failed: {error}")
                            });

                        assert_eq!(
                            result.token_ids, expected_ids,
                            "token state leaked into session {session_id}, request {request_id}"
                        );
                        assert_eq!(
                            result.text, expected_text,
                            "decoded output leaked into session {session_id}, request {request_id}"
                        );
                    }
                }));
            }

            for handle in handles {
                handle
                    .join()
                    .expect("generation stress-test worker panicked");
            }
        });
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
    fn cancellation_tokens_are_isolated_between_generation_sessions() {
        let first = CancellationToken::new();
        let second = CancellationToken::new();

        first.cancel();

        assert!(first.is_cancelled());
        assert!(!second.is_cancelled());
    }

    #[test]
    fn child_cancellation_is_independent_but_inherits_parent_shutdown() {
        let shutdown = CancellationToken::new();
        let first_request = shutdown.child_token();
        let second_request = shutdown.child_token();

        first_request.cancel();
        assert!(first_request.is_cancelled());
        assert!(!second_request.is_cancelled());
        assert!(!shutdown.is_cancelled());

        shutdown.cancel();
        assert!(first_request.is_cancelled());
        assert!(second_request.is_cancelled());
    }

    #[test]
    fn nested_child_tokens_inherit_all_ancestor_cancellation() {
        let root = CancellationToken::new();
        let child = root.child_token();
        let request = child.child_token();

        child.cancel();
        assert!(request.is_cancelled());
        assert!(!root.is_cancelled());
    }

    #[test]
    fn cancellation_token_clones_share_cancellation_state() {
        let token = CancellationToken::new();
        let signal = token.clone();

        assert!(!token.is_cancelled());
        signal.cancel();
        assert!(token.is_cancelled());
        assert!(matches!(token.check(), Err(GenerationError::Cancelled)));
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
    #[test]
    fn generation_events_emit_only_generated_tokens_and_incremental_text() {
        let (model, tokenizer) = tiny_llama_fixture();
        let backend = qarnyx_cpu::CpuBackend::new(1).unwrap();
        let cancellation = CancellationToken::new();
        let config = GenerationConfig {
            max_new_tokens: 3,
            max_context_tokens: 8,
            max_cache_mib: 1,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 123,
        };
        let mut generator = LlamaGenerator::new(123);
        let mut events = Vec::new();

        let result = generator
            .generate_ids_with_events(
                &model,
                &tokenizer,
                &backend,
                &[0],
                config,
                &cancellation,
                |event| events.push(event),
            )
            .unwrap();

        assert_eq!(
            events,
            [
                GenerationEvent::Token {
                    token_id: 0,
                    text_delta: "a".into(),
                },
                GenerationEvent::Token {
                    token_id: 0,
                    text_delta: "a".into(),
                },
                GenerationEvent::Token {
                    token_id: 0,
                    text_delta: "a".into(),
                },
            ]
        );
        assert_eq!(result.token_ids, [0, 0, 0, 0]);
        assert_eq!(result.text, "aaaa");
    }

    #[test]
    fn generation_event_callback_can_request_cancellation() {
        let (model, tokenizer) = tiny_llama_fixture();
        let backend = qarnyx_cpu::CpuBackend::new(1).unwrap();
        let cancellation = CancellationToken::new();
        let config = GenerationConfig {
            max_new_tokens: 3,
            max_context_tokens: 8,
            max_cache_mib: 1,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 123,
        };
        let mut generator = LlamaGenerator::new(123);
        let mut events = Vec::new();

        let result = generator.generate_ids_with_events(
            &model,
            &tokenizer,
            &backend,
            &[0],
            config,
            &cancellation,
            |event| {
                events.push(event);
                cancellation.cancel();
            },
        );

        assert!(matches!(result, Err(GenerationError::Cancelled)));
        assert_eq!(events.len(), 1);
        assert!(cancellation.is_cancelled());
    }
}

impl LlamaGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng_state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Generates new tokens from an already-tokenized prompt.
    ///
    /// This compatibility entry point runs without an externally-triggered cancellation signal.
    pub fn generate_ids<B: Backend>(
        &mut self,
        model: &LlamaModel<'_>,
        tokenizer: &GgufTokenizer,
        backend: &B,
        prompt: &[usize],
        config: GenerationConfig,
    ) -> Result<GenerationResult, GenerationError> {
        let cancellation = CancellationToken::new();
        self.generate_ids_with_cancellation(
            model,
            tokenizer,
            backend,
            prompt,
            config,
            &cancellation,
        )
    }

    /// Generates new tokens and checks the signal between complete token evaluations.
    ///
    /// Cancellation drops the request-local KV cache and returns GenerationError::Cancelled.
    pub fn generate_ids_with_cancellation<B: Backend>(
        &mut self,
        model: &LlamaModel<'_>,
        tokenizer: &GgufTokenizer,
        backend: &B,
        prompt: &[usize],
        config: GenerationConfig,
        cancellation: &CancellationToken,
    ) -> Result<GenerationResult, GenerationError> {
        self.generate_ids_internal(
            model,
            tokenizer,
            backend,
            prompt,
            config,
            cancellation,
            None::<fn(GenerationEvent)>,
        )
    }

    /// Generates tokens while emitting protocol-neutral events synchronously.
    ///
    /// The callback runs on the generation thread after each selected token.
    /// It can request cancellation through the shared CancellationToken.
    /// Implementations that bridge to async/HTTP transports should keep the
    /// callback bounded, for example by sending events to a channel.
    pub fn generate_ids_with_events<B, F>(
        &mut self,
        model: &LlamaModel<'_>,
        tokenizer: &GgufTokenizer,
        backend: &B,
        prompt: &[usize],
        config: GenerationConfig,
        cancellation: &CancellationToken,
        on_event: F,
    ) -> Result<GenerationResult, GenerationError>
    where
        B: Backend,
        F: FnMut(GenerationEvent),
    {
        self.generate_ids_internal(
            model,
            tokenizer,
            backend,
            prompt,
            config,
            cancellation,
            Some(on_event),
        )
    }

    fn generate_ids_internal<B, F>(
        &mut self,
        model: &LlamaModel<'_>,
        tokenizer: &GgufTokenizer,
        backend: &B,
        prompt: &[usize],
        config: GenerationConfig,
        cancellation: &CancellationToken,
        mut on_event: Option<F>,
    ) -> Result<GenerationResult, GenerationError>
    where
        B: Backend,
        F: FnMut(GenerationEvent),
    {
        config.validate()?;
        if prompt.is_empty() {
            return Err(GenerationError::EmptyPrompt);
        }
        config.validate_context(prompt.len())?;
        cancellation.check()?;
        let estimated_cache_bytes = estimate_llama_cache_bytes(model, config.max_context_tokens)?;
        config.validate_cache_budget(estimated_cache_bytes)?;
        cancellation.check()?;

        let mut cache =
            LlamaKvCache::with_context_capacity(&model.config, config.max_context_tokens)?;
        let mut generated = prompt.to_vec();
        let mut logits = Vec::new();
        let mut decoder = on_event.as_ref().map(|_| tokenizer.stream_decoder());

        for (position, &token_id) in prompt.iter().enumerate() {
            cancellation.check()?;
            logits = model.forward_logits(backend, token_id, position, &mut cache)?;
        }

        for _ in 0..config.max_new_tokens {
            cancellation.check()?;
            let next = self.sample(&logits, config)?;
            cancellation.check()?;
            if tokenizer.is_end_of_turn(next) {
                break;
            }

            generated.push(next);
            if let Some(callback) = on_event.as_mut() {
                let text_delta = decoder
                    .as_mut()
                    .expect("event callback requires a stream decoder")
                    .push(next)?;
                callback(GenerationEvent::Token {
                    token_id: next,
                    text_delta,
                });
                cancellation.check()?;
            }

            let position = cache.sequence_len();
            logits = model.forward_logits(backend, next, position, &mut cache)?;
        }

        if let Some(callback) = on_event.as_mut() {
            let final_delta = decoder
                .as_mut()
                .expect("event callback requires a stream decoder")
                .finish();
            if !final_delta.is_empty() {
                callback(GenerationEvent::TextDelta { text: final_delta });
            }
        }
        cancellation.check()?;

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
        let cancellation = CancellationToken::new();
        self.generate_with_cancellation(model, tokenizer, backend, prompt, config, &cancellation)
    }

    /// Tokenizes a prompt and runs generation with an external cancellation signal.
    pub fn generate_with_cancellation<B: Backend>(
        &mut self,
        model: &LlamaModel<'_>,
        tokenizer: &GgufTokenizer,
        backend: &B,
        prompt: &str,
        config: GenerationConfig,
        cancellation: &CancellationToken,
    ) -> Result<GenerationResult, GenerationError> {
        cancellation.check()?;
        let prompt_tokens = tokenizer.encode(prompt)?;
        self.generate_ids_with_cancellation(
            model,
            tokenizer,
            backend,
            &prompt_tokens,
            config,
            cancellation,
        )
    }

    /// Tokenizes a prompt and emits generated-token events as they become available.
    pub fn generate_with_events<B, F>(
        &mut self,
        model: &LlamaModel<'_>,
        tokenizer: &GgufTokenizer,
        backend: &B,
        prompt: &str,
        config: GenerationConfig,
        cancellation: &CancellationToken,
        on_event: F,
    ) -> Result<GenerationResult, GenerationError>
    where
        B: Backend,
        F: FnMut(GenerationEvent),
    {
        cancellation.check()?;
        let prompt_tokens = tokenizer.encode(prompt)?;
        self.generate_ids_with_events(
            model,
            tokenizer,
            backend,
            &prompt_tokens,
            config,
            cancellation,
            on_event,
        )
    }

    fn sample(
        &mut self,
        logits: &[f32],
        config: GenerationConfig,
    ) -> Result<usize, GenerationError> {
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

/// Autoregressive generator for one Qwen3.5 generation stream.
///
/// Model weights and the backend may be shared across requests. Each concurrent
/// stream should own its generator; every generation call allocates a fresh,
/// request-local recurrent/attention cache and scratch workspace.
pub struct Qwen35Generator {
    rng_state: u64,
}

impl Qwen35Generator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng_state: if seed == 0 { 1 } else { seed },
        }
    }

    pub fn generate_ids<B: Backend>(
        &mut self,
        model: &Qwen35Model<'_>,
        tokenizer: &GgufTokenizer,
        backend: &B,
        prompt: &[usize],
        config: GenerationConfig,
    ) -> Result<GenerationResult, GenerationError> {
        let cancellation = CancellationToken::new();
        self.generate_ids_with_cancellation(
            model,
            tokenizer,
            backend,
            prompt,
            config,
            &cancellation,
        )
    }

    /// Generates new tokens and checks the signal between complete token evaluations.
    ///
    /// Cancellation drops the request-local hybrid cache and returns GenerationError::Cancelled.
    pub fn generate_ids_with_cancellation<B: Backend>(
        &mut self,
        model: &Qwen35Model<'_>,
        tokenizer: &GgufTokenizer,
        backend: &B,
        prompt: &[usize],
        config: GenerationConfig,
        cancellation: &CancellationToken,
    ) -> Result<GenerationResult, GenerationError> {
        self.generate_ids_internal(
            model,
            tokenizer,
            backend,
            prompt,
            config,
            cancellation,
            None::<fn(GenerationEvent)>,
        )
    }

    /// Generates tokens while emitting protocol-neutral events synchronously.
    ///
    /// The callback runs on the generation thread after each selected token.
    /// It can request cancellation through the shared CancellationToken.
    /// Implementations that bridge to async/HTTP transports should keep the
    /// callback bounded, for example by sending events to a channel.
    pub fn generate_ids_with_events<B, F>(
        &mut self,
        model: &Qwen35Model<'_>,
        tokenizer: &GgufTokenizer,
        backend: &B,
        prompt: &[usize],
        config: GenerationConfig,
        cancellation: &CancellationToken,
        on_event: F,
    ) -> Result<GenerationResult, GenerationError>
    where
        B: Backend,
        F: FnMut(GenerationEvent),
    {
        self.generate_ids_internal(
            model,
            tokenizer,
            backend,
            prompt,
            config,
            cancellation,
            Some(on_event),
        )
    }

    fn generate_ids_internal<B, F>(
        &mut self,
        model: &Qwen35Model<'_>,
        tokenizer: &GgufTokenizer,
        backend: &B,
        prompt: &[usize],
        config: GenerationConfig,
        cancellation: &CancellationToken,
        mut on_event: Option<F>,
    ) -> Result<GenerationResult, GenerationError>
    where
        B: Backend,
        F: FnMut(GenerationEvent),
    {
        config.validate()?;
        if prompt.is_empty() {
            return Err(GenerationError::EmptyPrompt);
        }
        config.validate_context(prompt.len())?;
        cancellation.check()?;
        let estimated_cache_bytes = model
            .config
            .estimated_cache_bytes(config.max_context_tokens)
            .ok_or(GenerationError::InvalidConfig(
                "Qwen3.5 session cache size overflow or invalid layer map",
            ))?;
        config.validate_cache_budget(estimated_cache_bytes)?;
        cancellation.check()?;

        let mut cache =
            Qwen35Cache::with_context_capacity(&model.config, config.max_context_tokens)?;
        let mut generated = prompt.to_vec();
        let mut logits = Vec::new();
        let mut decoder = on_event.as_ref().map(|_| tokenizer.stream_decoder());
        let profiling = std::env::var_os("QARNYX_PROFILE").is_some();
        let mut prefill_profile = Qwen35Profile::default();
        let mut decode_profile = Qwen35Profile::default();

        // Profile model forward passes independently. Prefill and decode have
        // different shapes and bottlenecks, so merging their operator totals
        // makes the result difficult to use for kernel decisions.
        let prefill_started = profiling.then(Instant::now);
        for (position, &token_id) in prompt.iter().enumerate() {
            cancellation.check()?;
            if profiling {
                logits = model.forward_logits_profiled(
                    backend,
                    token_id,
                    position,
                    &mut cache,
                    &mut prefill_profile,
                )?;
            } else {
                logits = model.forward_logits(backend, token_id, position, &mut cache)?;
            }
        }
        let prefill_elapsed = prefill_started.map_or(Duration::ZERO, |start| start.elapsed());

        let mut decode_forward_elapsed = Duration::ZERO;
        let mut decode_forward_tokens = 0_usize;
        let decode_loop_started = profiling.then(Instant::now);

        for _ in 0..config.max_new_tokens {
            cancellation.check()?;
            let next = self.sample(&logits, config)?;
            cancellation.check()?;
            if tokenizer.is_end_of_turn(next) {
                break;
            }

            generated.push(next);
            if let Some(callback) = on_event.as_mut() {
                let text_delta = decoder
                    .as_mut()
                    .expect("event callback requires a stream decoder")
                    .push(next)?;
                callback(GenerationEvent::Token {
                    token_id: next,
                    text_delta,
                });
                cancellation.check()?;
            }

            let position = cache.sequence_len;
            let forward_started = profiling.then(Instant::now);
            if profiling {
                logits = model.forward_logits_profiled(
                    backend,
                    next,
                    position,
                    &mut cache,
                    &mut decode_profile,
                )?;
            } else {
                logits = model.forward_logits(backend, next, position, &mut cache)?;
            }
            if let Some(start) = forward_started {
                decode_forward_elapsed += start.elapsed();
                decode_forward_tokens += 1;
            }
        }

        let decode_loop_elapsed =
            decode_loop_started.map_or(Duration::ZERO, |start| start.elapsed());

        if let Some(callback) = on_event.as_mut() {
            let final_delta = decoder
                .as_mut()
                .expect("event callback requires a stream decoder")
                .finish();
            if !final_delta.is_empty() {
                callback(GenerationEvent::TextDelta { text: final_delta });
            }
        }
        cancellation.check()?;
        if profiling {
            let generated_tokens = generated.len().saturating_sub(prompt.len());
            let prefill_tokens_per_second = if prefill_elapsed.is_zero() {
                0.0
            } else {
                prompt.len() as f64 / prefill_elapsed.as_secs_f64()
            };
            let decode_forward_tokens_per_second = if decode_forward_elapsed.is_zero() {
                0.0
            } else {
                decode_forward_tokens as f64 / decode_forward_elapsed.as_secs_f64()
            };
            let decode_loop_tokens_per_second = if decode_loop_elapsed.is_zero() {
                0.0
            } else {
                generated_tokens as f64 / decode_loop_elapsed.as_secs_f64()
            };

            eprintln!("Qwen3.5 phase profile:");
            eprintln!(
                "  prefill: prompt_tokens={} forward_ms={:.3} tokens_per_sec={:.2}",
                prompt.len(),
                prefill_elapsed.as_secs_f64() * 1_000.0,
                prefill_tokens_per_second
            );
            eprintln!(
                "  decode: generated_tokens={} loop_ms={:.3} loop_tokens_per_sec={:.2}",
                generated_tokens,
                decode_loop_elapsed.as_secs_f64() * 1_000.0,
                decode_loop_tokens_per_second
            );
            eprintln!(
                "    forward_tokens={} forward_ms={:.3} forward_tokens_per_sec={:.2}",
                decode_forward_tokens,
                decode_forward_elapsed.as_secs_f64() * 1_000.0,
                decode_forward_tokens_per_second
            );
            eprintln!("{}", prefill_profile.report_for_phase("prefill"));
            eprintln!("{}", decode_profile.report_for_phase("decode"));
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
        let cancellation = CancellationToken::new();
        self.generate_with_cancellation(model, tokenizer, backend, prompt, config, &cancellation)
    }

    /// Tokenizes a prompt and runs generation with an external cancellation signal.
    pub fn generate_with_cancellation<B: Backend>(
        &mut self,
        model: &Qwen35Model<'_>,
        tokenizer: &GgufTokenizer,
        backend: &B,
        prompt: &str,
        config: GenerationConfig,
        cancellation: &CancellationToken,
    ) -> Result<GenerationResult, GenerationError> {
        cancellation.check()?;
        let ids = tokenizer.encode(prompt)?;
        self.generate_ids_with_cancellation(model, tokenizer, backend, &ids, config, cancellation)
    }

    /// Tokenizes a prompt and emits generated-token events as they become available.
    pub fn generate_with_events<B, F>(
        &mut self,
        model: &Qwen35Model<'_>,
        tokenizer: &GgufTokenizer,
        backend: &B,
        prompt: &str,
        config: GenerationConfig,
        cancellation: &CancellationToken,
        on_event: F,
    ) -> Result<GenerationResult, GenerationError>
    where
        B: Backend,
        F: FnMut(GenerationEvent),
    {
        cancellation.check()?;
        let ids = tokenizer.encode(prompt)?;
        self.generate_ids_with_events(
            model,
            tokenizer,
            backend,
            &ids,
            config,
            cancellation,
            on_event,
        )
    }

    fn sample(
        &mut self,
        logits: &[f32],
        config: GenerationConfig,
    ) -> Result<usize, GenerationError> {
        if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
            return Err(GenerationError::NonFiniteLogits);
        }
        if config.temperature == 0.0 {
            return logits
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(id, _)| id)
                .ok_or(GenerationError::NonFiniteLogits);
        }

        let mut candidates: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
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
        let mut probs: Vec<(usize, f64)> = candidates
            .iter()
            .zip(weights)
            .map(|(&(id, _), w)| (id, w / sum))
            .collect();
        if config.top_p < 1.0 {
            let mut cumulative = 0.0;
            let mut keep = 0;
            for &(_, p) in &probs {
                cumulative += p;
                keep += 1;
                if cumulative >= f64::from(config.top_p) {
                    break;
                }
            }
            probs.truncate(keep.max(1));
        }
        let total = probs.iter().map(|(_, p)| *p).sum::<f64>();
        let threshold = self.next_unit() * total;
        let mut cumulative = 0.0;
        for (id, p) in probs {
            cumulative += p;
            if cumulative >= threshold {
                return Ok(id);
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
        x as f64 / u64::MAX as f64
    }
}
