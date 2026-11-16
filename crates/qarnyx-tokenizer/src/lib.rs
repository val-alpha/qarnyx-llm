//! GGUF tokenizer metadata and token-id decoding.
//!
//! M5 keeps tokenization model-agnostic at the metadata boundary. GGUF
//! vocabularies, scores, merges, and special-token IDs are loaded without
//! touching model tensor payloads. Text-to-token encoding is added per
//! tokenizer family after the metadata contract is validated.

use qarnyx_gguf::{GgufFile, MetadataValue};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("missing GGUF tokenizer metadata key: {0}")]
    MissingMetadata(&'static str),

    #[error("invalid GGUF tokenizer metadata key {key}: expected {expected}")]
    InvalidMetadata {
        key: &'static str,
        expected: &'static str,
    },

    #[error("tokenizer vocabulary is empty")]
    EmptyVocabulary,

    #[error("token ID {0} is outside the vocabulary")]
    TokenIdOutOfRange(usize),
}

/// Tokenizer metadata embedded in a GGUF model.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufTokenizer {
    pub model: String,
    pub pre: Option<String>,
    pub tokens: Vec<String>,
    pub scores: Vec<f32>,
    pub merges: Vec<String>,
    pub bos_token_id: Option<usize>,
    pub eos_token_id: Option<usize>,
}

impl GgufTokenizer {
    /// Loads the GGUF tokenizer metadata and validates array lengths.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, TokenizerError> {
        let model = required_string(gguf, "tokenizer.ggml.model")?;
        let pre = optional_string(gguf, "tokenizer.ggml.pre")?;
        let tokens = required_string_array(gguf, "tokenizer.ggml.tokens")?;
        if tokens.is_empty() {
            return Err(TokenizerError::EmptyVocabulary);
        }

        let scores = optional_f32_array(gguf, "tokenizer.ggml.scores")?
            .unwrap_or_else(|| vec![0.0; tokens.len()]);
        if scores.len() != tokens.len() {
            return Err(TokenizerError::InvalidMetadata {
                key: "tokenizer.ggml.scores",
                expected: "FLOAT32 array with one score per token",
            });
        }

        let merges = optional_string_array(gguf, "tokenizer.ggml.merges")?
            .unwrap_or_default();

        Ok(Self {
            model,
            pre,
            tokens,
            scores,
            merges,
            bos_token_id: optional_usize(gguf, "tokenizer.ggml.bos_token_id")?,
            eos_token_id: optional_usize(gguf, "tokenizer.ggml.eos_token_id")?,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Converts generated token IDs back to their GGUF token strings.
    pub fn decode(&self, token_ids: &[usize]) -> Result<String, TokenizerError> {
        let mut output = String::new();
        for &token_id in token_ids {
            let token = self
                .tokens
                .get(token_id)
                .ok_or(TokenizerError::TokenIdOutOfRange(token_id))?;
            output.push_str(token);
        }
        Ok(output)
    }
}

fn required_string<'a>(gguf: &'a GgufFile, key: &'static str) -> Result<&'a str, TokenizerError> {
    gguf.metadata_value(key)
        .ok_or(TokenizerError::MissingMetadata(key))?
        .as_str()
        .ok_or(TokenizerError::InvalidMetadata {
            key,
            expected: "STRING",
        })
}

fn optional_string(gguf: &GgufFile, key: &'static str) -> Result<Option<String>, TokenizerError> {
    match gguf.metadata_value(key) {
        None => Ok(None),
        Some(MetadataValue::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(TokenizerError::InvalidMetadata {
            key,
            expected: "STRING",
        }),
    }
}

fn required_string_array(
    gguf: &GgufFile,
    key: &'static str,
) -> Result<Vec<String>, TokenizerError> {
    optional_string_array(gguf, key)?.ok_or(TokenizerError::MissingMetadata(key))
}

fn optional_string_array(
    gguf: &GgufFile,
    key: &'static str,
) -> Result<Option<Vec<String>>, TokenizerError> {
    let Some(value) = gguf.metadata_value(key) else {
        return Ok(None);
    };
    match value {
        MetadataValue::Array { values, .. } => values
            .iter()
            .map(|value| match value {
                MetadataValue::String(value) => Ok(value.clone()),
                _ => Err(TokenizerError::InvalidMetadata {
                    key,
                    expected: "STRING array",
                }),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        _ => Err(TokenizerError::InvalidMetadata {
            key,
            expected: "ARRAY",
        }),
    }
}

fn optional_f32_array(
    gguf: &GgufFile,
    key: &'static str,
) -> Result<Option<Vec<f32>>, TokenizerError> {
    let Some(value) = gguf.metadata_value(key) else {
        return Ok(None);
    };
    match value {
        MetadataValue::Array { values, .. } => values
            .iter()
            .map(|value| match value {
                MetadataValue::Float32(value) => Ok(*value),
                MetadataValue::Float64(value) if value.is_finite() => Ok(*value as f32),
                _ => Err(TokenizerError::InvalidMetadata {
                    key,
                    expected: "FLOAT32 array",
                }),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        _ => Err(TokenizerError::InvalidMetadata {
            key,
            expected: "ARRAY",
        }),
    }
}

fn optional_usize(gguf: &GgufFile, key: &'static str) -> Result<Option<usize>, TokenizerError> {
    let Some(value) = gguf.metadata_value(key) else {
        return Ok(None);
    };
    let raw = match value {
        MetadataValue::Uint32(value) => u64::from(*value),
        MetadataValue::Uint64(value) => *value,
        MetadataValue::Int32(value) if *value >= 0 => *value as u64,
        MetadataValue::Int64(value) if *value >= 0 => *value as u64,
        _ => {
            return Err(TokenizerError::InvalidMetadata {
                key,
                expected: "unsigned integer",
            })
        }
    };
    usize::try_from(raw).map(Some).map_err(|_| TokenizerError::InvalidMetadata {
        key,
        expected: "integer fitting usize",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use qarnyx_gguf::{MetadataEntry, MetadataValue};

    #[test]
    fn loads_vocab_and_special_tokens() {
        let gguf = GgufFile {
            version: 3,
            metadata: vec![
                MetadataEntry {
                    key: "tokenizer.ggml.model".into(),
                    value: MetadataValue::String("llama".into()),
                },
                MetadataEntry {
                    key: "tokenizer.ggml.tokens".into(),
                    value: MetadataValue::Array {
                        element_type: qarnyx_gguf::MetadataValueType::String,
                        values: vec![
                            MetadataValue::String("<unk>".into()),
                            MetadataValue::String("hello".into()),
                        ],
                    },
                },
                MetadataEntry {
                    key: "tokenizer.ggml.bos_token_id".into(),
                    value: MetadataValue::Uint32(1),
                },
                MetadataEntry {
                    key: "tokenizer.ggml.eos_token_id".into(),
                    value: MetadataValue::Uint32(0),
                },
            ],
            tensors: Vec::new(),
            alignment: 32,
            tensor_data_offset: 0,
            file_size: 0,
        };

        let tokenizer = GgufTokenizer::from_gguf(&gguf).unwrap();
        assert_eq!(tokenizer.model, "llama");
        assert_eq!(tokenizer.vocab_size(), 2);
        assert_eq!(tokenizer.bos_token_id, Some(1));
        assert_eq!(tokenizer.decode(&[1]).unwrap(), "hello");
    }
}
