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

    #[error("tokenizer model {0} is not supported for production encoding")]
    UnsupportedModel(String),

    #[error("chat prompt formatting is unavailable for architecture '{0}'")]
    UnsupportedChatArchitecture(String),

    #[error("chat prompt contains unsupported role '{0}'")]
    UnsupportedChatRole(String),

    #[error("tokenizer cannot encode input at byte offset {offset}")]
    Unencodable { offset: usize },
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

/// A chat message after protocol-specific JSON has been normalized.
///
/// This type deliberately contains no vendor protocol types. Tool calls are
/// represented as a neutral JSON string so the tokenizer can apply the model
/// chat template without depending on qarnyx-api.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatPromptMessage {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<String>,
    pub tool_call_id: Option<String>,
}

/// Stateful incremental text decoder for generated token streams.
///
/// Qwen3.5 uses GPT-2 byte-to-Unicode token spellings. A UTF-8 character may
/// span multiple tokens, so this decoder retains an incomplete trailing
/// sequence until a later token completes it.
#[derive(Debug, Clone)]
pub struct TokenizerStreamDecoder<'a> {
    tokenizer: &'a GgufTokenizer,
    pending_utf8: Vec<u8>,
    reverse_map: Option<std::collections::HashMap<u32, u8>>,
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
            model: model.to_owned(),
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

    /// Encodes text using the tokenizer family declared by GGUF metadata.
    ///
    /// Recognized control-token spellings present in the vocabulary are
    /// emitted as their dedicated token IDs, rather than being split into
    /// ordinary byte-level pieces.
    pub fn encode(&self, text: &str) -> Result<Vec<usize>, TokenizerError> {
        let mut special_tokens = self
            .tokens
            .iter()
            .enumerate()
            .filter(|(_, token)| is_special_token_spelling(token))
            .map(|(id, token)| (id, token.as_str()))
            .collect::<Vec<_>>();
        special_tokens.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

        let mut output = Vec::new();
        let mut cursor = 0;
        while cursor < text.len() {
            let next = special_tokens
                .iter()
                .filter_map(|&(id, marker)| {
                    text[cursor..].find(marker).map(|relative| {
                        (cursor + relative, id, marker)
                    })
                })
                .min_by(|a, b| {
                    a.0.cmp(&b.0)
                        .then_with(|| b.2.len().cmp(&a.2.len()))
                });

            let Some((position, token_id, marker)) = next else {
                output.extend(self.encode_plain(&text[cursor..])?);
                break;
            };

            output.extend(self.encode_plain(&text[cursor..position])?);
            output.push(token_id);
            cursor = position + marker.len();
        }

        Ok(output)
    }

    fn encode_plain(&self, text: &str) -> Result<Vec<usize>, TokenizerError> {
        if self.merges.is_empty() {
            self.encode_vocab_pieces(text)
        } else {
            self.encode_bpe(text)
        }
    }

    /// Formats normalized chat messages using the selected model family template.
    ///
    /// Vendor request parsing and validation stay in qarnyx-api. This method
    /// owns model-specific role syntax, control tokens, and assistant prefix.
    pub fn format_chat_prompt(
        &self,
        architecture: &str,
        messages: &[ChatPromptMessage],
        tool_instruction: Option<&str>,
    ) -> Result<String, TokenizerError> {
        let mut prompt = String::new();
        match architecture {
            "qwen35" => {
                if self.model != "gpt2" || self.pre.as_deref() != Some("qwen35") {
                    return Err(TokenizerError::UnsupportedModel(self.model.clone()));
                }
                for message in messages {
                    let role = match message.role.as_str() {
                        "system" | "developer" => "system",
                        "user" => "user",
                        "assistant" => "assistant",
                        "tool" => "tool",
                        other => return Err(TokenizerError::UnsupportedChatRole(other.to_owned())),
                    };
                    prompt.push_str("<|im_start|>");
                    prompt.push_str(role);
                    prompt.push('\n');
                    if let Some(content) = &message.content {
                        prompt.push_str(content);
                    }
                    if let Some(tool_calls) = &message.tool_calls {
                        prompt.push_str(tool_calls);
                    }
                    if message.role == "tool" {
                        prompt.push_str(" [tool_call_id=");
                        prompt.push_str(message.tool_call_id.as_deref().unwrap_or(""));
                        prompt.push(']');
                    }
                    prompt.push_str("<|im_end|>\n");
                }
                if let Some(instruction) = tool_instruction {
                    prompt.push_str("<|im_start|>system\n");
                    prompt.push_str(instruction);
                    prompt.push_str("<|im_end|>\n");
                }
                prompt.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
            }
            "llama" => {
                for message in messages {
                    let role = match message.role.as_str() {
                        "system" | "developer" => "SYSTEM",
                        "user" => "USER",
                        "assistant" => "ASSISTANT",
                        "tool" => "TOOL",
                        other => return Err(TokenizerError::UnsupportedChatRole(other.to_owned())),
                    };
                    prompt.push_str(role);
                    prompt.push_str(":\n");
                    if let Some(content) = &message.content {
                        prompt.push_str(content);
                    }
                    if let Some(tool_calls) = &message.tool_calls {
                        prompt.push_str(tool_calls);
                    }
                    if message.role == "tool" {
                        prompt.push_str(" [tool_call_id=");
                        prompt.push_str(message.tool_call_id.as_deref().unwrap_or(""));
                        prompt.push(']');
                    }
                    prompt.push_str("\n\n");
                }
                if let Some(instruction) = tool_instruction {
                    prompt.push_str("SYSTEM:\n");
                    prompt.push_str(instruction);
                    prompt.push_str("\n\n");
                }
                prompt.push_str("ASSISTANT:\n");
            }
            other => return Err(TokenizerError::UnsupportedChatArchitecture(other.to_owned())),
        }
        Ok(prompt)
    }

    /// Formats a single-turn, text-only Qwen3.5 chat prompt.
    ///
    /// Thinking mode leaves the assistant inside its reasoning block.
    /// Non-thinking mode prefills an empty reasoning block before the answer.
    pub fn encode_qwen35_chat_prompt(
        &self,
        prompt: &str,
        enable_thinking: bool,
    ) -> Result<Vec<usize>, TokenizerError> {
        if self.model != "gpt2" || self.pre.as_deref() != Some("qwen35") {
            return Err(TokenizerError::UnsupportedModel(self.model.clone()));
        }

        let assistant_prefix = if enable_thinking {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n"
        };
        let formatted = format!(
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{assistant_prefix}"
        );
        self.encode(&formatted)
    }

    /// Returns whether a token terminates an assistant turn.
    ///
    /// Qwen3.5 GGUFs may expose the turn-ending marker separately from the
    /// tokenizer's single EOS ID, so recognize known end markers by spelling.
    pub fn is_end_of_turn(&self, token_id: usize) -> bool {
        self.eos_token_id == Some(token_id)
            || matches!(
                self.tokens.get(token_id).map(String::as_str),
                Some("<|im_end|>" | "<|fim_suffix|>" | "<|endoftext|>")
            )
    }

    fn encode_vocab_pieces(&self, text: &str) -> Result<Vec<usize>, TokenizerError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let pieces = self.pre_tokenize(text);
        let mut tokens = Vec::new();

        for (offset, piece) in pieces {
            if let Some(token_id) = self.tokens.iter().position(|token| token == &piece) {
                tokens.push(token_id);
                continue;
            }

            // Some simple vocabularies contain raw text pieces rather than
            // SentencePiece-style whitespace markers. Preserve the original
            // longest-prefix fallback for those vocabularies.
            let raw = piece.strip_prefix('▁').unwrap_or(&piece);
            let mut cursor = 0;
            while cursor < raw.len() {
                let tail = &raw[cursor..];
                let mut best: Option<(usize, usize)> = None;

                for (token_id, token) in self.tokens.iter().enumerate() {
                    if token.is_empty() || !tail.starts_with(token) {
                        continue;
                    }
                    let token_len = token.len();
                    if best.map(|(_, len)| token_len > len).unwrap_or(true) {
                        best = Some((token_id, token_len));
                    }
                }

                let Some((token_id, token_len)) = best else {
                    return Err(TokenizerError::Unencodable { offset });
                };

                tokens.push(token_id);
                cursor += token_len;
            }
        }

        Ok(tokens)
    }

    fn encode_bpe(&self, text: &str) -> Result<Vec<usize>, TokenizerError> {
        let mut merge_rank = std::collections::HashMap::with_capacity(self.merges.len());
        for (rank, merge) in self.merges.iter().enumerate() {
            let mut parts = merge.split_whitespace();
            let (Some(left), Some(right)) = (parts.next(), parts.next()) else {
                continue;
            };
            merge_rank.insert((left.to_owned(), right.to_owned()), rank);
        }

        let byte_level = self.model == "gpt2" && self.pre.as_deref() == Some("qwen35");
        let pieces = self.pre_tokenize(text);
        let mut output = Vec::new();

        for (offset, piece) in pieces {
            let symbols = if byte_level {
                byte_to_unicode(&piece)
                    .chars()
                    .map(|ch| ch.to_string())
                    .collect::<Vec<_>>()
            } else {
                piece.chars().map(|ch| ch.to_string()).collect::<Vec<_>>()
            };

            if symbols.is_empty() {
                continue;
            }

            let mut symbols = symbols;
            loop {
                let mut best: Option<(usize, usize)> = None;
                for i in 0..symbols.len().saturating_sub(1) {
                    if let Some(&rank) =
                        merge_rank.get(&(symbols[i].clone(), symbols[i + 1].clone()))
                        && best.map(|(_, current)| rank < current).unwrap_or(true)
                    {
                        best = Some((i, rank));
                    }
                }
                let Some((index, _)) = best else { break };
                let merged = format!("{}{}", symbols[index], symbols[index + 1]);
                symbols.splice(index..=index + 1, [merged]);
            }

            for symbol in symbols {
                let Some(id) = self.tokens.iter().position(|token| token == &symbol) else {
                    return Err(TokenizerError::Unencodable { offset });
                };
                output.push(id);
            }
        }

        Ok(output)
    }

    fn pre_tokenize(&self, text: &str) -> Vec<(usize, String)> {
        if self.model == "gpt2" && self.pre.as_deref() == Some("qwen35") {
            return qwen35_pre_tokenize(text);
        }

        let mut pieces = Vec::new();
        match self.pre.as_deref() {
            Some(pre) if pre.contains("llama") || pre.contains("sentencepiece") => {
                for (index, word) in text.split_whitespace().enumerate() {
                    let offset = text.find(word).unwrap_or(index);
                    pieces.push((offset, format!("▁{word}")));
                }
            }
            _ => {
                let mut search_from = 0;
                for (index, word) in text.split_whitespace().enumerate() {
                    let relative = text[search_from..].find(word).unwrap_or(0);
                    let offset = search_from + relative;
                    let prefix = if index == 0 { "" } else { "Ġ" };
                    pieces.push((offset, format!("{prefix}{word}")));
                    search_from = offset + word.len();
                }
            }
        }
        pieces
    }

    /// Converts generated token IDs back to text.
    pub fn decode(&self, token_ids: &[usize]) -> Result<String, TokenizerError> {
        if self.model != "gpt2" || self.pre.as_deref() != Some("qwen35") {
            let mut output = String::new();
            for &token_id in token_ids {
                output.push_str(
                    self.tokens
                        .get(token_id)
                        .ok_or(TokenizerError::TokenIdOutOfRange(token_id))?,
                );
            }
            return Ok(output);
        }

        let reverse_map = gpt2_reverse_map();
        let mut bytes = Vec::new();
        for &token_id in token_ids {
            let token = self
                .tokens
                .get(token_id)
                .ok_or(TokenizerError::TokenIdOutOfRange(token_id))?;
            bytes.extend_from_slice(&unicode_to_bytes_with_map(token, &reverse_map));
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Creates a decoder that yields text deltas as generated tokens arrive.
    pub fn stream_decoder(&self) -> TokenizerStreamDecoder<'_> {
        let is_qwen35_byte_level = self.model == "gpt2" && self.pre.as_deref() == Some("qwen35");
        TokenizerStreamDecoder {
            tokenizer: self,
            pending_utf8: Vec::new(),
            reverse_map: is_qwen35_byte_level.then(gpt2_reverse_map),
        }
    }
}

impl<'a> TokenizerStreamDecoder<'a> {
    /// Decodes one generated token into the next available valid text delta.
    ///
    /// The returned string can be empty when a token ends in an incomplete
    /// UTF-8 sequence. Call finish when generation ends to flush any
    /// remaining bytes using the same lossy policy as GgufTokenizer::decode.
    pub fn push(&mut self, token_id: usize) -> Result<String, TokenizerError> {
        let token = self
            .tokenizer
            .tokens
            .get(token_id)
            .ok_or(TokenizerError::TokenIdOutOfRange(token_id))?;

        if self.tokenizer.model != "gpt2" || self.tokenizer.pre.as_deref() != Some("qwen35") {
            return Ok(token.clone());
        }

        let reverse_map = self
            .reverse_map
            .as_ref()
            .expect("Qwen3.5 byte-level decoder must have a reverse map");
        self.pending_utf8
            .extend_from_slice(&unicode_to_bytes_with_map(token, reverse_map));
        Ok(self.drain_valid_utf8())
    }

    /// Flushes an incomplete trailing byte sequence at end of stream.
    pub fn finish(&mut self) -> String {
        String::from_utf8_lossy(&std::mem::take(&mut self.pending_utf8)).into_owned()
    }

    fn drain_valid_utf8(&mut self) -> String {
        let mut output = String::new();

        loop {
            let (valid_up_to, error_len, is_valid) = match std::str::from_utf8(&self.pending_utf8) {
                Ok(_) => (self.pending_utf8.len(), None, true),
                Err(error) => (error.valid_up_to(), error.error_len(), false),
            };

            if is_valid {
                let bytes = std::mem::take(&mut self.pending_utf8);
                output.push_str(
                    &String::from_utf8(bytes).expect("validated UTF-8 bytes must decode"),
                );
                return output;
            }

            if valid_up_to > 0 {
                let prefix = std::str::from_utf8(&self.pending_utf8[..valid_up_to])
                    .expect("UTF-8 valid_up_to must identify a valid prefix");
                output.push_str(prefix);
                self.pending_utf8.drain(..valid_up_to);
            }

            match error_len {
                Some(invalid_len) => {
                    output.push('\u{FFFD}');
                    self.pending_utf8.drain(..invalid_len);
                }
                None => return output,
            }
        }
    }
}

fn is_special_token_spelling(token: &str) -> bool {
    (token.starts_with("<|") && token.ends_with("|>"))
        || matches!(
            token,
            "<think>" | "</think>" | "<tool_call>" | "</tool_call>"
        )
}

fn byte_to_unicode(text: &str) -> String {
    let map = gpt2_byte_map();
    text.as_bytes()
        .iter()
        .map(|byte| char::from_u32(map[*byte as usize]).expect("GPT-2 byte map is Unicode"))
        .collect()
}

fn unicode_to_bytes_with_map(token: &str, reverse: &std::collections::HashMap<u32, u8>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(token.len());
    for ch in token.chars() {
        if let Some(&byte) = reverse.get(&(ch as u32)) {
            bytes.push(byte);
        } else {
            let mut encoded = [0_u8; 4];
            bytes.extend_from_slice(ch.encode_utf8(&mut encoded).as_bytes());
        }
    }
    bytes
}

fn gpt2_byte_map() -> [u32; 256] {
    let mut map = [0u32; 256];
    let mut bs = Vec::new();
    for b in 33u8..=126 { bs.push(b); }
    for b in 161u8..=172 { bs.push(b); }
    for b in 174u8..=255 { bs.push(b); }
    let mut n = 0u32;
    for b in 0u16..=255 {
        let byte = b as u8;
        if bs.contains(&byte) {
            map[b as usize] = b as u32;
        } else {
            map[b as usize] = 256 + n;
            n += 1;
        }
    }
    map
}

fn gpt2_reverse_map() -> std::collections::HashMap<u32, u8> {
    let map = gpt2_byte_map();
    let mut reverse = std::collections::HashMap::with_capacity(256);
    for (byte, codepoint) in map.into_iter().enumerate() {
        reverse.insert(codepoint, byte as u8);
    }
    reverse
}

fn qwen35_pre_tokenize(text: &str) -> Vec<(usize, String)> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut pieces = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        let start = chars[i].0;
        let mut j = i;
        let mut piece = String::new();

        if chars[j].1 == '\n' || chars[j].1 == '\r' {
            while j < chars.len() && (chars[j].1 == '\n' || chars[j].1 == '\r') {
                piece.push(chars[j].1);
                j += 1;
            }
            pieces.push((start, piece));
            i = j;
            continue;
        }

        if chars[j].1 == ' ' && j + 1 < chars.len() {
            let next = chars[j + 1].1;
            if next.is_alphabetic() {
                piece.push(' ');
                j += 1;
            } else if !next.is_whitespace() && !next.is_alphabetic() && !next.is_numeric() {
                piece.push(' ');
                j += 1;
            }
        }

        if j < chars.len() && (chars[j].1.is_alphabetic()) {
            while j < chars.len() && chars[j].1.is_alphabetic() {
                piece.push(chars[j].1);
                j += 1;
            }
        } else if j < chars.len() && chars[j].1.is_numeric() {
            piece.push(chars[j].1);
            j += 1;
        } else if j < chars.len() && !chars[j].1.is_whitespace() {
            while j < chars.len() {
                let c = chars[j].1;
                if c.is_whitespace() || c.is_alphabetic() || c.is_numeric() {
                    break;
                }
                piece.push(c);
                j += 1;
            }
        } else if j < chars.len() {
            while j < chars.len() && chars[j].1.is_whitespace() {
                piece.push(chars[j].1);
                j += 1;
            }
        }

        if !piece.is_empty() {
            pieces.push((start, piece));
        }
        i = j.max(i + 1);
    }

    pieces
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
    fn encodes_merge_based_vocabulary() {
        let tokenizer = GgufTokenizer {
            model: "test-bpe".into(),
            pre: None,
            tokens: vec![
                "h".into(),
                "e".into(),
                "l".into(),
                "o".into(),
                "he".into(),
                "ll".into(),
                "hello".into(),
            ],
            scores: vec![0.0; 7],
            merges: vec!["h e".into(), "he l".into(), "l o".into(), "hel lo".into()],
            bos_token_id: None,
            eos_token_id: None,
        };

        assert_eq!(tokenizer.encode("hello").unwrap(), [6]);
    }

    #[test]
    fn encodes_chat_control_tokens_as_atomic_vocabulary_entries() {
        let tokenizer = GgufTokenizer {
            model: "gpt2".into(),
            pre: Some("qwen35".into()),
            tokens: vec![
                "h".into(),
                "i".into(),
                "<|im_start|>".into(),
                "<|im_end|>".into(),
            ],
            scores: vec![0.0; 4],
            merges: Vec::new(),
            bos_token_id: None,
            eos_token_id: None,
        };

        assert_eq!(
            tokenizer.encode("<|im_start|>hi<|im_end|>").unwrap(),
            [2, 0, 1, 3]
        );
    }

    #[test]
    fn formats_qwen35_multi_turn_chat_and_tool_history() {
        let tokenizer = GgufTokenizer {
            model: "gpt2".into(),
            pre: Some("qwen35".into()),
            tokens: vec!["x".into()],
            scores: vec![0.0],
            merges: Vec::new(),
            bos_token_id: None,
            eos_token_id: None,
        };
        let messages = vec![
            ChatPromptMessage {
                role: "system".into(), content: Some("Be concise.".into()),
                tool_calls: None, tool_call_id: None
            },
            ChatPromptMessage {
                role: "assistant".into(), content: None,
                tool_calls: Some(r#"{"tool_calls":[{"id":"toolu_1"}]}"#.into()),
                tool_call_id: None
            },
            ChatPromptMessage {
                role: "tool".into(), content: Some("Sunny".into()),
                tool_calls: None, tool_call_id: Some("toolu_1".into())
            },
        ];
        let prompt = tokenizer.format_chat_prompt("qwen35", &messages, Some("Call tools only when needed.")).unwrap();
        assert!(prompt.starts_with("<|im_start|>system\nBe concise.<|im_end|>"));
        assert!(prompt.contains(r#"{"tool_calls":[{"id":"toolu_1"}]}"#));
        assert!(!prompt.contains("TOOL"));
        assert!(prompt.contains(" [tool_call_id=toolu_1]"));
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn formats_llama_chat_and_rejects_unknown_architectures() {
        let tokenizer = GgufTokenizer {
            model: "llama".into(),
            pre: None,
            tokens: vec!["x".into()],
            scores: vec![0.0],
            merges: Vec::new(),
            bos_token_id: None,
            eos_token_id: None,
        };
        let messages = vec![ChatPromptMessage {
            role: "user".into(), content: Some("Hello".into()),
            tool_calls: None, tool_call_id: None
        }];
        let prompt = tokenizer.format_chat_prompt("llama", &messages, None).unwrap();
        assert_eq!(prompt, "USER:\nHello\n\nASSISTANT:\n");
        assert!(matches!(
            tokenizer.format_chat_prompt("unknown", &messages, None),
            Err(TokenizerError::UnsupportedChatArchitecture(_))
        ));
    }

    #[test]
    fn qwen35_chat_prompt_uses_non_thinking_assistant_prefix() {
        let tokenizer = GgufTokenizer {
            model: "gpt2".into(),
            pre: Some("qwen35".into()),
            tokens: vec![
                "<|im_start|>".into(),
                "<|im_end|>".into(),
                "<think>".into(),
                "</think>".into(),
                "user".into(),
                "\n".into(),
                "hi".into(),
                "assistant".into(),
                "\n\n".into(),
            ],
            scores: vec![0.0; 9],
            merges: Vec::new(),
            bos_token_id: None,
            eos_token_id: None,
        };

        assert_eq!(
            tokenizer.encode_qwen35_chat_prompt("hi", false).unwrap(),
            [0, 4, 5, 6, 1, 5, 0, 7, 5, 2, 8, 3, 8]
        );
    }

    #[test]
    fn qwen35_chat_prompt_leaves_reasoning_open_when_enabled() {
        let tokenizer = GgufTokenizer {
            model: "gpt2".into(),
            pre: Some("qwen35".into()),
            tokens: vec![
                "<|im_start|>".into(),
                "<|im_end|>".into(),
                "<think>".into(),
                "</think>".into(),
                "user".into(),
                "\n".into(),
                "hi".into(),
                "assistant".into(),
                "\n\n".into(),
            ],
            scores: vec![0.0; 9],
            merges: Vec::new(),
            bos_token_id: None,
            eos_token_id: None,
        };

        assert_eq!(
            tokenizer.encode_qwen35_chat_prompt("hi", true).unwrap(),
            [0, 4, 5, 6, 1, 5, 0, 7, 5, 2, 5]
        );
    }

    #[test]
    fn recognizes_qwen35_turn_end_markers() {
        let tokenizer = GgufTokenizer {
            model: "gpt2".into(),
            pre: Some("qwen35".into()),
            tokens: vec![
                "<|im_end|>".into(),
                "<|fim_suffix|>".into(),
                "ordinary".into(),
                "<|endoftext|>".into(),
            ],
            scores: vec![0.0; 4],
            merges: Vec::new(),
            bos_token_id: None,
            eos_token_id: Some(2),
        };

        assert!(tokenizer.is_end_of_turn(0));
        assert!(tokenizer.is_end_of_turn(1));
        assert!(tokenizer.is_end_of_turn(2));
        assert!(tokenizer.is_end_of_turn(3));
    }

    #[test]
    fn pre_tokenizes_sentencepiece_words() {
        let tokenizer = GgufTokenizer {
            model: "test-bpe".into(),
            pre: Some("sentencepiece".into()),
            tokens: vec!["▁hello".into(), "▁world".into()],
            scores: vec![0.0; 2],
            merges: Vec::new(),
            bos_token_id: None,
            eos_token_id: None,
        };

        assert_eq!(tokenizer.encode("hello world").unwrap(), [0, 1]);
    }

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
        assert_eq!(tokenizer.encode("hello").unwrap(), [1]);
    }
    #[test]
    fn stream_decoder_preserves_plain_token_text() {
        let tokenizer = GgufTokenizer {
            model: "test".into(),
            pre: None,
            tokens: vec!["hello".into(), " world".into()],
            scores: vec![0.0; 2],
            merges: Vec::new(),
            bos_token_id: None,
            eos_token_id: None,
        };
        let mut decoder = tokenizer.stream_decoder();

        assert_eq!(decoder.push(0).unwrap(), "hello");
        assert_eq!(decoder.push(1).unwrap(), " world");
        assert_eq!(decoder.finish(), "");
    }

    #[test]
    fn qwen35_stream_decoder_buffers_split_utf8_sequences() {
        let encoded = byte_to_unicode("é");
        let mut chars = encoded.chars();
        let tokenizer = GgufTokenizer {
            model: "gpt2".into(),
            pre: Some("qwen35".into()),
            tokens: vec![
                chars.next().unwrap().to_string(),
                chars.next().unwrap().to_string(),
            ],
            scores: vec![0.0; 2],
            merges: Vec::new(),
            bos_token_id: None,
            eos_token_id: None,
        };
        let mut decoder = tokenizer.stream_decoder();

        assert_eq!(decoder.push(0).unwrap(), "");
        assert_eq!(decoder.push(1).unwrap(), "é");
        assert_eq!(decoder.finish(), "");
    }

    #[test]
    fn qwen35_stream_decoder_flushes_incomplete_utf8_lossily() {
        let encoded = byte_to_unicode("é");
        let first_byte = encoded.chars().next().unwrap().to_string();
        let tokenizer = GgufTokenizer {
            model: "gpt2".into(),
            pre: Some("qwen35".into()),
            tokens: vec![first_byte],
            scores: vec![0.0],
            merges: Vec::new(),
            bos_token_id: None,
            eos_token_id: None,
        };
        let mut decoder = tokenizer.stream_decoder();

        assert_eq!(decoder.push(0).unwrap(), "");
        assert_eq!(decoder.finish(), "�");
        assert_eq!(decoder.finish(), "");
    }

    #[test]
    fn stream_decoder_rejects_out_of_range_token_ids() {
        let tokenizer = GgufTokenizer {
            model: "test".into(),
            pre: None,
            tokens: vec!["hello".into()],
            scores: vec![0.0],
            merges: Vec::new(),
            bos_token_id: None,
            eos_token_id: None,
        };
        let mut decoder = tokenizer.stream_decoder();

        assert!(matches!(
            decoder.push(1),
            Err(TokenizerError::TokenIdOutOfRange(1))
        ));
    }
}
