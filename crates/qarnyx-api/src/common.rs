//! Shared protocol-neutral API request and response contract.
//!
//! HTTP protocol adapters translate their wire formats into this contract;
//! only the shared preparation/generation path talks to qarnyx-runtime.

use qarnyx_runtime::{GenerationConfig, GenerationResult as RuntimeGenerationResult};

/// A vendor-neutral prompt and native Qarnyx generation configuration.
#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub prompt: String,
    pub config: GenerationConfig,
}

/// Generated output in a protocol-neutral representation.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationOutput {
    pub text: String,
    /// Generated token IDs only; prompt token IDs are intentionally excluded.
    pub token_ids: Vec<usize>,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

impl GenerationOutput {
    /// Converts the runtime's backward-compatible prompt+completion result into
    /// the common API response contract.
    pub fn from_runtime(
        text: String,
        result: &RuntimeGenerationResult,
        prompt_tokens: usize,
    ) -> Self {
        let generated = result.token_ids.get(prompt_tokens..).unwrap_or_default();
        Self {
            text,
            token_ids: generated.to_vec(),
            prompt_tokens,
            completion_tokens: generated.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_output_excludes_prompt_tokens() {
        let runtime = RuntimeGenerationResult {
            token_ids: vec![10, 11, 20, 21],
            text: "prompt + answer".into(),
        };
        let output = GenerationOutput::from_runtime("answer".into(), &runtime, 2);
        assert_eq!(output.token_ids, vec![20, 21]);
        assert_eq!(output.prompt_tokens, 2);
        assert_eq!(output.completion_tokens, 2);
        assert_eq!(output.text, "answer");
    }

    #[test]
    fn generation_output_handles_empty_or_short_runtime_results() {
        let runtime = RuntimeGenerationResult {
            token_ids: vec![10],
            text: "prompt".into(),
        };
        let output = GenerationOutput::from_runtime(String::new(), &runtime, 4);
        assert!(output.token_ids.is_empty());
        assert_eq!(output.completion_tokens, 0);
    }
}
