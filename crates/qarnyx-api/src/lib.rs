//! OpenAI Chat Completions-compatible HTTP adapter for Qarnyx.
//!
//! The adapter owns HTTP/JSON/SSE concerns and delegates all model execution to
//! qarnyx-runtime. Model weights are shared read-only; each request creates its
//! own generator and request-local cache.

use axum::{
    body::Body,
    extract::{rejection::JsonRejection, State},
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use bytes::Bytes;
use qarnyx_cpu::{CpuBackend, CpuBackendError};
use qarnyx_gguf::{GgufError, GgufFile};
use qarnyx_models::{LlamaModel, Qwen35Model};
use qarnyx_runtime::{
    CancellationToken, GenerationConfig, GenerationError, GenerationEvent, GenerationResult,
    LlamaGenerator, Qwen35Generator,
};
use qarnyx_tokenizer::{GgufTokenizer, TokenizerError};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    convert::Infallible,
    fs::File,
    io::{self, Read},
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{SystemTime, UNIX_EPOCH},
};
use thiserror::Error;
use tokio::sync::{mpsc, Semaphore};
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

const DEFAULT_SEED: u64 = 0x51_4e_52_59_58;
static NEXT_COMPLETION_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Error)]
pub enum ApiLoadError {
    #[error("model file I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("GGUF parsing failed: {0}")]
    Gguf(#[from] GgufError),
    #[error("tokenizer loading failed: {0}")]
    Tokenizer(#[from] TokenizerError),
    #[error("CPU backend construction failed: {0}")]
    Cpu(#[from] CpuBackendError),
    #[error("model loading failed: {0}")]
    Model(String),
    #[error("unsupported model architecture: {0}")]
    UnsupportedArchitecture(String),
    #[error("invalid memory budget: {0}")]
    MemoryBudget(String),
}

/// Shared immutable model assets and backend configuration.
///
/// Tensor payloads stay in one shared byte buffer. A lightweight model view is
/// reconstructed per request so no self-referential model/byte-buffer owner is
/// required and mutable generation state is never shared between requests.
#[derive(Clone)]
pub struct ApiState {
    assets: Arc<ModelAssets>,
}

struct ModelAssets {
    model_data: Arc<[u8]>,
    gguf: GgufFile,
    tokenizer: GgufTokenizer,
    backend: CpuBackend,
    architecture: String,
    model_id: String,
    max_context_tokens: usize,
    max_cache_mib: usize,
    request_slots: Arc<Semaphore>,
}

impl ApiState {
    /// Loads and validates a single GGUF model for HTTP serving.
    pub fn load(
        path: impl AsRef<Path>,
        workers: usize,
        max_memory_mib: usize,
        max_context_tokens: usize,
        max_cache_mib: usize,
        max_concurrent_requests: usize,
    ) -> Result<Self, ApiLoadError> {
        if max_context_tokens == 0 || max_concurrent_requests == 0 {
            return Err(ApiLoadError::MemoryBudget(
                "max_context_tokens and max_concurrent_requests must be non-zero".into(),
            ));
        }
        let model_data = read_model_with_budget(
            path.as_ref(),
            max_memory_mib,
            max_cache_mib,
            max_concurrent_requests,
        )?;
        let gguf = GgufFile::parse(&model_data)?;
        let architecture = gguf
            .metadata_string("general.architecture")
            .ok_or_else(|| ApiLoadError::Model("GGUF is missing general.architecture".into()))?
            .to_owned();
        let model_id = gguf
            .metadata_string("general.name")
            .filter(|name| !name.trim().is_empty())
            .map(str::to_owned)
            .or_else(|| {
                path.as_ref()
                    .file_stem()
                    .map(|name| name.to_string_lossy().into_owned())
            })
            .unwrap_or_else(|| "qarnyx-model".into());
        let tokenizer = GgufTokenizer::from_gguf(&gguf)?;

        // Validate the architecture and tensor mapping before opening the
        // listener. The temporary view borrows model_data only for this check.
        match architecture.as_str() {
            "llama" => {
                let _ = LlamaModel::from_gguf(&gguf, &model_data)
                    .map_err(|error| ApiLoadError::Model(error.to_string()))?;
            }
            "qwen35" => {
                let _ = Qwen35Model::from_gguf(&gguf, &model_data)
                    .map_err(|error| ApiLoadError::Model(error.to_string()))?;
            }
            other => return Err(ApiLoadError::UnsupportedArchitecture(other.to_owned())),
        }

        let backend = CpuBackend::new(workers)?;
        Ok(Self {
            assets: Arc::new(ModelAssets {
                model_data: Arc::from(model_data),
                gguf,
                tokenizer,
                backend,
                architecture,
                model_id,
                max_context_tokens,
                max_cache_mib,
                request_slots: Arc::new(Semaphore::new(max_concurrent_requests)),
            }),
        })
    }

    pub fn model_id(&self) -> &str {
        &self.assets.model_id
    }

    pub fn architecture(&self) -> &str {
        &self.assets.architecture
    }
}

fn read_model_with_budget(
    path: &Path,
    max_memory_mib: usize,
    max_cache_mib: usize,
    max_concurrent_requests: usize,
) -> Result<Vec<u8>, ApiLoadError> {
    const MIB: usize = 1024 * 1024;
    if max_memory_mib == 0 || max_cache_mib == 0 {
        return Err(ApiLoadError::MemoryBudget(
            "max_memory_mib and max_cache_mib must be non-zero".into(),
        ));
    }
    let total = max_memory_mib
        .checked_mul(MIB)
        .ok_or_else(|| ApiLoadError::MemoryBudget("max_memory_mib overflows bytes".into()))?;
    let cache = max_cache_mib
        .checked_mul(max_concurrent_requests)
        .and_then(|value| value.checked_mul(MIB))
        .ok_or_else(|| {
            ApiLoadError::MemoryBudget(
                "max_cache_mib multiplied by max_concurrent_requests overflows bytes".into(),
            )
        })?;
    let model_limit = total.checked_sub(cache).filter(|value| *value > 0).ok_or_else(|| {
        ApiLoadError::MemoryBudget(
            "max_memory_mib must exceed the aggregate cache reservation, leaving space for model weights".into(),
        )
    })?;

    let mut file = File::open(path)?;
    let file_size = usize::try_from(file.metadata()?.len())
        .map_err(|_| ApiLoadError::MemoryBudget("model file size does not fit usize".into()))?;
    if file_size > model_limit {
        return Err(ApiLoadError::MemoryBudget(format!(
            "model file is {file_size} bytes but the model budget is {model_limit} bytes"
        )));
    }

    let mut bytes = Vec::new();
    bytes.try_reserve_exact(file_size).map_err(|error| {
        ApiLoadError::MemoryBudget(format!("could not reserve model buffer: {error}"))
    })?;
    bytes.resize(file_size, 0);
    file.read_exact(&mut bytes)?;
    let mut extra = [0_u8; 1];
    if file.read(&mut extra)? != 0 {
        return Err(ApiLoadError::MemoryBudget(
            "model file changed size while being read".into(),
        ));
    }
    Ok(bytes)
}

pub fn router(state: ApiState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(create_chat_completion))
        .with_state(state)
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn list_models(State(state): State<ApiState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.assets.model_id,
            "object": "model",
            "created": 0,
            "owned_by": "qarnyx"
        }]
    }))
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    n: Option<usize>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    stop: Option<Value>,
    logprobs: Option<bool>,
    top_logprobs: Option<usize>,
    logit_bias: Option<HashMap<String, Value>>,
    tools: Option<Value>,
    tool_choice: Option<Value>,
    functions: Option<Value>,
    function_call: Option<Value>,
    response_format: Option<ResponseFormat>,
    stream_options: Option<StreamOptions>,
    #[serde(rename = "user")]
    _user: Option<String>,
    store: Option<bool>,
    #[serde(default)]
    max_context_tokens: Option<usize>,
    max_cache_mib: Option<usize>,
    #[serde(flatten)]
    unsupported: HashMap<String, Value>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
    #[serde(flatten)]
    unsupported: HashMap<String, Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Debug, Deserialize, Default)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
    include_obfuscation: Option<bool>,
    #[serde(flatten)]
    unsupported: HashMap<String, Value>,
}

struct PreparedRequest {
    prompt_ids: Vec<usize>,
    config: GenerationConfig,
    seed: u64,
    include_usage: bool,
}

impl ApiState {
    fn prepare(&self, request: ChatCompletionRequest) -> Result<PreparedRequest, ApiError> {
        if request.model != self.assets.model_id && request.model != "qarnyx" {
            return Err(ApiError::not_found(format!(
                "model '{}' is not loaded; available model is '{}'",
                request.model, self.assets.model_id
            )));
        }
        if request.messages.is_empty() {
            return Err(ApiError::bad_request("messages must contain at least one message"));
        }
        for message in &request.messages {
            if !message.unsupported.is_empty() {
                return Err(ApiError::bad_request(
                    "message fields other than role and text content are not supported",
                ));
            }
        }
        if request.n.unwrap_or(1) != 1 {
            return Err(ApiError::bad_request("only n=1 is supported"));
        }
        for (name, value) in [
            ("frequency_penalty", request.frequency_penalty),
            ("presence_penalty", request.presence_penalty),
        ] {
            if value.is_some_and(|value| value != 0.0) {
                return Err(ApiError::bad_request(format!(
                    "{name} is not supported; omit it or set it to 0"
                )));
            }
        }
        if request.stop.as_ref().is_some_and(|value| !value.is_null()) {
            return Err(ApiError::bad_request(
                "stop sequences are not supported by this runtime adapter yet",
            ));
        }
        if request.logprobs.unwrap_or(false) || request.top_logprobs.unwrap_or(0) > 0 {
            return Err(ApiError::bad_request("logprobs are not supported"));
        }
        if request.logit_bias.as_ref().is_some_and(|value| !value.is_empty()) {
            return Err(ApiError::bad_request("logit_bias is not supported"));
        }
        if request.tools.is_some()
            || request.tool_choice.is_some()
            || request.functions.is_some()
            || request.function_call.is_some()
        {
            return Err(ApiError::bad_request("tool and function calling are not supported"));
        }
        if let Some(format) = request.response_format {
            if format.kind != "text" {
                return Err(ApiError::bad_request(
                    "only response_format.type='text' is supported",
                ));
            }
        }
        if request.store.unwrap_or(false) {
            return Err(ApiError::bad_request(
                "response storage is not supported; set store=false",
            ));
        }
        if !request.unsupported.is_empty() {
            let mut fields = request.unsupported.keys().cloned().collect::<Vec<_>>();
            fields.sort();
            return Err(ApiError::bad_request(format!(
                "unsupported request fields: {}",
                fields.join(", ")
            )));
        }

        let max_new_tokens = match (request.max_tokens, request.max_completion_tokens) {
            (Some(_), Some(_)) => {
                return Err(ApiError::bad_request(
                    "provide max_tokens or max_completion_tokens, not both",
                ));
            }
            (Some(value), None) | (None, Some(value)) => value,
            (None, None) => GenerationConfig::default().max_new_tokens,
        };
        let max_context_tokens = request
            .max_context_tokens
            .unwrap_or(self.assets.max_context_tokens);
        let max_cache_mib = request.max_cache_mib.unwrap_or(self.assets.max_cache_mib);

        if max_context_tokens == 0 || max_cache_mib == 0 {
            return Err(ApiError::bad_request(
                "max_context_tokens and max_cache_mib must be non-zero",
            ));
        }
        if max_context_tokens > self.assets.max_context_tokens {
            return Err(ApiError::bad_request(format!(
                "max_context_tokens exceeds the server limit of {}",
                self.assets.max_context_tokens
            )));
        }
        if max_cache_mib > self.assets.max_cache_mib {
            return Err(ApiError::bad_request(format!(
                "max_cache_mib exceeds the server limit of {}",
                self.assets.max_cache_mib
            )));
        }

        let stream_options = request.stream_options.unwrap_or_default();
        if stream_options.include_obfuscation.unwrap_or(false) {
            return Err(ApiError::bad_request(
                "stream obfuscation is not supported",
            ));
        }
        if !stream_options.unsupported.is_empty() {
            return Err(ApiError::bad_request(
                "unsupported stream_options fields were provided",
            ));
        }

        let prompt = format_chat_prompt(&self.assets.architecture, &request.messages)?;
        let prompt_ids = self
            .assets
            .tokenizer
            .encode(&prompt)
            .map_err(|error| ApiError::bad_request(error.to_string()))?;
        let seed = request.seed.unwrap_or(DEFAULT_SEED);
        let config = GenerationConfig {
            max_new_tokens,
            max_context_tokens,
            max_cache_mib,
            temperature: request.temperature.unwrap_or(0.8),
            top_k: GenerationConfig::default().top_k,
            top_p: request.top_p.unwrap_or(0.95),
            seed,
        };
        Ok(PreparedRequest {
            prompt_ids,
            config,
            seed,
            include_usage: stream_options.include_usage,
        })
    }

    fn generate<F>(
        &self,
        prepared: &PreparedRequest,
        cancellation: &CancellationToken,
        on_event: F,
    ) -> Result<GenerationResult, RunError>
    where
        F: FnMut(GenerationEvent),
    {
        match self.assets.architecture.as_str() {
            "llama" => {
                let model = LlamaModel::from_gguf(
                    &self.assets.gguf,
                    self.assets.model_data.as_ref(),
                )
                .map_err(|error| RunError::Model(error.to_string()))?;
                let mut generator = LlamaGenerator::new(prepared.seed);
                generator
                    .generate_ids_with_events(
                        &model,
                        &self.assets.tokenizer,
                        &self.assets.backend,
                        &prepared.prompt_ids,
                        prepared.config,
                        cancellation,
                        on_event,
                    )
                    .map_err(RunError::Generation)
            }
            "qwen35" => {
                let model = Qwen35Model::from_gguf(
                    &self.assets.gguf,
                    self.assets.model_data.as_ref(),
                )
                .map_err(|error| RunError::Model(error.to_string()))?;
                let mut generator = Qwen35Generator::new(prepared.seed);
                generator
                    .generate_ids_with_events(
                        &model,
                        &self.assets.tokenizer,
                        &self.assets.backend,
                        &prepared.prompt_ids,
                        prepared.config,
                        cancellation,
                        on_event,
                    )
                    .map_err(RunError::Generation)
            }
            other => Err(RunError::Model(format!(
                "unsupported model architecture: {other}"
            ))),
        }
    }
}

fn format_chat_prompt(architecture: &str, messages: &[ChatMessage]) -> Result<String, ApiError> {
    let mut prompt = String::new();
    match architecture {
        "qwen35" => {
            for message in messages {
                let role = match message.role.as_str() {
                    "system" | "developer" => "system",
                    "user" => "user",
                    "assistant" => "assistant",
                    other => {
                        return Err(ApiError::bad_request(format!(
                            "unsupported message role '{other}'"
                        )));
                    }
                };
                prompt.push_str("<|im_start|>");
                prompt.push_str(role);
                prompt.push('\n');
                prompt.push_str(&message.content);
                prompt.push_str("<|im_end|>\n");
            }
            // Match the runtime's default non-thinking Qwen3.5 chat prefix.
            prompt.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
        }
        "llama" => {
            for message in messages {
                let role = match message.role.as_str() {
                    "system" | "developer" => "SYSTEM",
                    "user" => "USER",
                    "assistant" => "ASSISTANT",
                    other => {
                        return Err(ApiError::bad_request(format!(
                            "unsupported message role '{other}'"
                        )));
                    }
                };
                prompt.push_str(role);
                prompt.push_str(":\n");
                prompt.push_str(&message.content);
                prompt.push_str("\n\n");
            }
            prompt.push_str("ASSISTANT:\n");
        }
        other => {
            return Err(ApiError::bad_request(format!(
                "chat prompt formatting is unavailable for architecture '{other}'"
            )));
        }
    }
    Ok(prompt)
}

#[derive(Debug, Error)]
enum RunError {
    #[error("generation failed: {0}")]
    Generation(#[from] GenerationError),
    #[error("model error: {0}")]
    Model(String),
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    kind: &'static str,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            message: message.into(),
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            kind: "invalid_request_error",
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            kind: "server_error",
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({
                "error": {
                    "message": self.message,
                    "type": self.kind,
                    "param": Value::Null,
                    "code": Value::Null
                }
            })),
        )
            .into_response()
    }
}

impl From<RunError> for ApiError {
    fn from(error: RunError) -> Self {
        match error {
            RunError::Generation(
                error @ (GenerationError::EmptyPrompt
                | GenerationError::InvalidConfig(_)
                | GenerationError::ContextLimitExceeded { .. }
                | GenerationError::CacheBudgetExceeded { .. }),
            ) => Self::bad_request(error.to_string()),
            RunError::Generation(GenerationError::Cancelled) => {
                Self::bad_request("generation was cancelled")
            }
            RunError::Generation(error) => Self::internal(error.to_string()),
            RunError::Model(error) => Self::internal(error),
        }
    }
}

async fn create_chat_completion(
    State(state): State<ApiState>,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) =
        payload.map_err(|error| ApiError::bad_request(format!("invalid JSON request: {error}")))?;
    let stream = request.stream;
    let prepared = state.prepare(request)?;
    let prompt_tokens = prepared.prompt_ids.len();
    let max_new_tokens = prepared.config.max_new_tokens;

    if stream {
        return stream_chat_completion(
            state,
            prepared,
            prompt_tokens,
            max_new_tokens,
        )
        .await
        .map(|response| response);
    }

    let permit = state
        .assets
        .request_slots
        .clone()
        .acquire_owned()
        .await
        .map_err(|error| ApiError::internal(format!("request admission failed: {error}")))?;
    let cancellation = CancellationToken::new();
    let state_for_task = state.clone();
    let completion = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut content = String::new();
        let result = state_for_task.generate(&prepared, &cancellation, |event| match event {
            GenerationEvent::Token { text_delta, .. } => content.push_str(&text_delta),
            GenerationEvent::TextDelta { text } => content.push_str(&text),
        })?;
        Ok::<_, RunError>((content, result))
    })
    .await
    .map_err(|error| ApiError::internal(format!("generation task failed: {error}")))?
    .map_err(ApiError::from)?;

    let completion_tokens = completion
        .1
        .token_ids
        .len()
        .saturating_sub(prompt_tokens);
    let finish_reason = if completion_tokens >= max_new_tokens {
        "length"
    } else {
        "stop"
    };
    let (id, created) = next_completion_identity();
    Ok(Json(json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": state.assets.model_id,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": completion.0 },
            "logprobs": null,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }))
    .into_response())
}

async fn stream_chat_completion(
    state: ApiState,
    prepared: PreparedRequest,
    prompt_tokens: usize,
    max_new_tokens: usize,
) -> Result<Response, ApiError> {
    let permit = state
        .assets
        .request_slots
        .clone()
        .acquire_owned()
        .await
        .map_err(|error| ApiError::internal(format!("request admission failed: {error}")))?;
    let include_usage = prepared.include_usage;
    let (id, created) = next_completion_identity();
    let model_id = state.assets.model_id.clone();
    let cancellation = CancellationToken::new();
    let task_cancellation = cancellation.clone();
    let (sender, receiver) = mpsc::channel::<Bytes>(8);

    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let send = |value: Value| -> bool {
            let bytes = Bytes::from(format!("data: {}\n\n", value));
            if sender.blocking_send(bytes).is_err() {
                task_cancellation.cancel();
                false
            } else {
                true
            }
        };

        if !send(stream_chunk(
            &id,
            created,
            &model_id,
            json!({"role": "assistant", "content": ""}),
            Value::Null,
            include_usage,
        )) {
            return;
        }

        let mut completion_tokens = 0usize;
        let mut disconnected = false;
        let result = state.generate(&prepared, &task_cancellation, |event| {
            let delta = match event {
                GenerationEvent::Token { token_id: _, text_delta } => {
                    completion_tokens += 1;
                    text_delta
                }
                GenerationEvent::TextDelta { text } => text,
            };
            if delta.is_empty() {
                return;
            }
            if !send(stream_chunk(
                &id,
                created,
                &model_id,
                json!({"content": delta}),
                Value::Null,
                include_usage,
            )) {
                disconnected = true;
                task_cancellation.cancel();
            }
        });

        if disconnected || task_cancellation.is_cancelled() {
            return;
        }

        match result {
            Ok(generation) => {
                completion_tokens = generation
                    .token_ids
                    .len()
                    .saturating_sub(prompt_tokens)
                    .max(completion_tokens);
                let finish_reason = if completion_tokens >= max_new_tokens {
                    "length"
                } else {
                    "stop"
                };
                if !send(stream_chunk(
                    &id,
                    created,
                    &model_id,
                    json!({}),
                    json!(finish_reason),
                    include_usage,
                )) {
                    return;
                }
                if include_usage {
                    if !send(json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens
                        }
                    })) {
                        return;
                    }
                }
                let _ = sender.blocking_send(Bytes::from_static(b"data: [DONE]\n\n"));
            }
            Err(error) => {
                let _ = send(json!({
                    "error": {
                        "message": error.to_string(),
                        "type": "server_error",
                        "param": null,
                        "code": null
                    }
                }));
                let _ = sender.blocking_send(Bytes::from_static(b"data: [DONE]\n\n"));
            }
        }
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"))
        .header(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"))
        .header(header::CONNECTION, HeaderValue::from_static("keep-alive"))
        .body(Body::from_stream(ReceiverStream::new(receiver).map(Ok::<_, Infallible>)))
        .map_err(|error| ApiError::internal(format!("could not construct SSE response: {error}")))
}

fn stream_chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: Value,
    finish_reason: Value,
    include_usage: bool,
) -> Value {
    let mut chunk = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "logprobs": null,
            "finish_reason": finish_reason
        }]
    });
    if include_usage {
        chunk["usage"] = Value::Null;
    }
    chunk
}

fn next_completion_identity() -> (String, u64) {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let sequence = NEXT_COMPLETION_ID.fetch_add(1, Ordering::Relaxed);
    (format!("chatcmpl-qarnyx-{created}-{sequence}"), created)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.into(),
            unsupported: HashMap::new(),
        }
    }

    #[test]
    fn qwen35_chat_prompt_preserves_roles_and_non_thinking_prefix() {
        let prompt = format_chat_prompt(
            "qwen35",
            &[
                message("system", "Be concise."),
                message("user", "Hello"),
            ],
        )
        .unwrap();

        assert_eq!(
            prompt,
            "<|im_start|>system\nBe concise.<|im_end|>\n\
             <|im_start|>user\nHello<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn llama_chat_prompt_preserves_roles() {
        let prompt = format_chat_prompt(
            "llama",
            &[message("user", "Hello"), message("assistant", "Hi")],
        )
        .unwrap();
        assert_eq!(prompt, "USER:\nHello\n\nASSISTANT:\nHi\n\nASSISTANT:\n");
    }

    #[test]
    fn unsupported_chat_roles_are_rejected() {
        assert!(format_chat_prompt("qwen35", &[message("tool", "result")]).is_err());
    }

    #[test]
    fn stream_chunks_use_openai_chunk_shape() {
        let chunk = stream_chunk(
            "chatcmpl-test",
            123,
            "qarnyx",
            json!({"content": "hello"}),
            Value::Null,
            true,
        );
        assert_eq!(chunk["object"], "chat.completion.chunk");
        assert_eq!(chunk["choices"][0]["delta"]["content"], "hello");
        assert!(chunk["usage"].is_null());
    }

    #[test]
    fn completion_ids_are_unique() {
        let first = next_completion_identity().0;
        let second = next_completion_identity().0;
        assert_ne!(first, second);
    }

    #[test]
    fn request_defaults_are_applied_without_streaming() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "qarnyx",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
        assert!(!request.stream);
        assert_eq!(request.n.unwrap_or(1), 1);
        assert_eq!(request.max_tokens, None);
    }
}

