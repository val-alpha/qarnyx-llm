//! Native Qarnyx generation endpoint.
//!
//! Unlike vendor chat protocols, this adapter accepts a raw prompt and the
//! runtime's own generation controls. It still uses the shared request,
//! preparation, cancellation, admission, and generation path.

use super::*;
use axum::{
    Json,
    body::Body,
    extract::{State, rejection::JsonRejection},
    http::{HeaderValue, header},
    response::Response,
};
use bytes::Bytes;
use serde::Deserialize;
use serde_json::{Value, json};
use std::convert::Infallible;
use tokio::sync::mpsc;
use tokio_stream::{StreamExt, wrappers::ReceiverStream};

#[derive(Debug)]
struct NativeApiError(ApiError);

impl From<ApiError> for NativeApiError {
    fn from(error: ApiError) -> Self { Self(error) }
}

impl IntoResponse for NativeApiError {
    fn into_response(self) -> Response {
        let error_type = match self.0.status {
            StatusCode::NOT_FOUND => "not_found",
            StatusCode::INTERNAL_SERVER_ERROR => "internal_error",
            _ => "invalid_request",
        };
        (
            self.0.status,
            Json(json!({
                "object":"qarnyx.error",
                "error":{"type":error_type,"message":self.0.message}
            })),
        ).into_response()
    }
}

#[derive(Debug, Deserialize)]
struct NativeGenerateRequest {
    model: String,
    prompt: String,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    config: NativeGenerationConfig,
    #[serde(flatten)]
    unsupported: std::collections::HashMap<String, Value>,
}

/// Native runtime configuration. Omitted values use GenerationConfig defaults.
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct NativeGenerationConfig {
    max_new_tokens: Option<usize>,
    max_context_tokens: Option<usize>,
    max_cache_mib: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    seed: Option<u64>,
}

fn prepare_request(
    state: &ApiState,
    request: NativeGenerateRequest,
) -> Result<PreparedRequest, ApiError> {
    if request.model != state.assets.model_id && request.model != "qarnyx" {
        return Err(ApiError::not_found(format!(
            "model '{}' is not loaded; available model is '{}'",
            request.model, state.assets.model_id
        )));
    }
    if request.prompt.is_empty() {
        return Err(ApiError::bad_request("prompt must not be empty"));
    }
    if !request.unsupported.is_empty() {
        let mut fields = request.unsupported.keys().cloned().collect::<Vec<_>>();
        fields.sort();
        return Err(ApiError::bad_request(format!(
            "unsupported native request fields: {}", fields.join(", ")
        )));
    }

    let defaults = GenerationConfig::default();
    let options = request.config;
    let config = GenerationConfig {
        max_new_tokens: options.max_new_tokens.unwrap_or(defaults.max_new_tokens),
        max_context_tokens: options.max_context_tokens.unwrap_or(state.assets.max_context_tokens),
        max_cache_mib: options.max_cache_mib.unwrap_or(state.assets.max_cache_mib),
        temperature: options.temperature.unwrap_or(defaults.temperature),
        top_k: options.top_k.unwrap_or(defaults.top_k),
        top_p: options.top_p.unwrap_or(defaults.top_p),
        seed: options.seed.unwrap_or(defaults.seed),
    };
    state.prepare_generation(
        GenerationRequest { prompt: request.prompt, config },
        false,
        Vec::new(),
        ToolChoice::None,
    )
}

fn finish_reason(output: &GenerationOutput, max_new_tokens: usize) -> &'static str {
    if output.completion_tokens >= max_new_tokens { "length" } else { "stop" }
}

pub(super) async fn generate(
    State(state): State<ApiState>,
    payload: Result<Json<NativeGenerateRequest>, JsonRejection>,
) -> Result<Response, NativeApiError> {
    let Json(request) = payload.map_err(|error| {
        ApiError::bad_request(format!("invalid JSON request: {error}"))
    })?;
    let stream = request.stream;
    let prepared = prepare_request(&state, request)?;
    let prompt_tokens = prepared.prompt_ids.len();
    let max_new_tokens = prepared.config.max_new_tokens;
    if stream {
        return stream_generation(state, prepared, prompt_tokens, max_new_tokens).await;
    }

    let permit = state.assets.request_slots.clone().acquire_owned().await
        .map_err(|error| ApiError::internal(format!("request admission failed: {error}")))?;
    let cancellation = CancellationToken::new();
    let state_task = state.clone();
    let (text, result) = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut text = String::new();
        let result = state_task.generate(&prepared, &cancellation, |event| match event {
            GenerationEvent::Token { text_delta, .. } => text.push_str(&text_delta),
            GenerationEvent::TextDelta { text: delta } => text.push_str(&delta),
        })?;
        Ok::<_, RunError>((text, result))
    }).await.map_err(|error| ApiError::internal(format!("generation task failed: {error}")))?
      .map_err(ApiError::from)?;

    let output = GenerationOutput::from_runtime(text, &result, prompt_tokens);
    Ok(Json(json!({
        "object": "qarnyx.generation",
        "model": state.assets.model_id,
        "text": output.text,
        "token_ids": output.token_ids,
        "usage": {
            "prompt_tokens": output.prompt_tokens,
            "completion_tokens": output.completion_tokens,
            "total_tokens": output.prompt_tokens + output.completion_tokens
        },
        "finish_reason": finish_reason(&output, max_new_tokens)
    })).into_response())
}

async fn stream_generation(
    state: ApiState,
    prepared: PreparedRequest,
    prompt_tokens: usize,
    max_new_tokens: usize,
) -> Result<Response, NativeApiError> {
    let permit = state.assets.request_slots.clone().acquire_owned().await
        .map_err(|error| ApiError::internal(format!("request admission failed: {error}")))?;
    let model = state.assets.model_id.clone();
    let cancellation = CancellationToken::new();
    let task_cancellation = cancellation.clone();
    let (sender, receiver) = mpsc::channel::<Bytes>(8);

    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let send = |event_name: &str, value: Value| -> bool {
            let bytes = Bytes::from(format!("event: {event_name}\ndata: {value}\n\n"));
            if sender.blocking_send(bytes).is_err() {
                task_cancellation.cancel();
                false
            } else { true }
        };
        let mut text = String::new();
        let mut generated_ids = Vec::new();
        let result = state.generate(&prepared, &task_cancellation, |event| {
            match event {
                GenerationEvent::Token { token_id, text_delta } => {
                    generated_ids.push(token_id);
                    text.push_str(&text_delta);
                    if !send("token", json!({
                        "type":"token","token_id":token_id,"text_delta":text_delta
                    })) {
                        task_cancellation.cancel();
                    }
                }
                GenerationEvent::TextDelta { text: delta } => {
                    text.push_str(&delta);
                    if !send("text_delta", json!({"type":"text_delta","text":delta})) {
                        task_cancellation.cancel();
                    }
                }
            }
        });
        if task_cancellation.is_cancelled() { return; }
        match result {
            Ok(generation) => {
                let output = GenerationOutput::from_runtime(text, &generation, prompt_tokens);
                let _ = send("completed", json!({
                    "type":"completed",
                    "model":model,
                    "text":output.text,
                    "token_ids":if output.token_ids.is_empty() { generated_ids } else { output.token_ids },
                    "usage":{
                        "prompt_tokens":output.prompt_tokens,
                        "completion_tokens":output.completion_tokens,
                        "total_tokens":output.prompt_tokens + output.completion_tokens
                    },
                    "finish_reason":finish_reason(&output, max_new_tokens)
                }));
                let _ = send("done", json!({"type":"done"}));
            }
            Err(error) => {
                let _ = send("error", json!({"type":"error","message":error.to_string()}));
            }
        }
    });

    let response = Response::builder()
        .header(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"))
        .header(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"))
        .body(Body::from_stream(ReceiverStream::new(receiver).map(Ok::<_, Infallible>)))
        .map_err(|error| ApiError::internal(format!("could not construct SSE response: {error}")))?;
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_config_deserializes_runtime_controls() {
        let request: NativeGenerateRequest = serde_json::from_value(json!({
            "model":"qarnyx",
            "prompt":"Once upon a time",
            "stream":true,
            "config":{
                "max_new_tokens":64,
                "max_context_tokens":1024,
                "max_cache_mib":128,
                "temperature":0.2,
                "top_k":20,
                "top_p":0.9,
                "seed":7
            }
        })).unwrap();
        assert!(request.stream);
        assert_eq!(request.config.max_new_tokens, Some(64));
        assert_eq!(request.config.top_k, Some(20));
        assert_eq!(request.config.seed, Some(7));
    }

    #[test]
    fn native_request_rejects_unknown_config_fields() {
        let request = serde_json::from_value::<NativeGenerateRequest>(json!({
            "model":"qarnyx",
            "prompt":"hello",
            "config":{"made_up_option":true}
        }));
        assert!(request.is_err());
    }

    #[test]
    fn native_finish_reason_observes_generation_limit() {
        let output = GenerationOutput {
            text: "done".into(), token_ids: vec![1,2], prompt_tokens: 4, completion_tokens: 2
        };
        assert_eq!(finish_reason(&output, 2), "length");
        assert_eq!(finish_reason(&output, 3), "stop");
    }
}
