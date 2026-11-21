//! Anthropic Messages-compatible HTTP adapter.
//!
//! Wire-format parsing/serialization lives here. Requests are normalized into
//! the shared ChatMessage/GenerationRequest path and use the same runtime event
//! source as the OpenAI adapter.

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
use std::{
    convert::Infallible,
    sync::atomic::{AtomicU64, Ordering},
};
use tokio::sync::mpsc;
use tokio_stream::{StreamExt, wrappers::ReceiverStream};

static NEXT_MESSAGE_ID: AtomicU64 = AtomicU64::new(1);

static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

fn next_request_id() -> String {
    format!("req_qarnyx_{}", NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed))
}

fn with_request_id(mut response: Response) -> Response {
    if let Ok(value) = HeaderValue::try_from(next_request_id().as_str()) {
        response.headers_mut().insert("request-id", value);
    }
    response
}


#[derive(Debug)]
struct AnthropicApiError(ApiError);

impl From<ApiError> for AnthropicApiError {
    fn from(error: ApiError) -> Self { Self(error) }
}

impl IntoResponse for AnthropicApiError {
    fn into_response(self) -> Response {
        let error_type = match self.0.status {
            StatusCode::NOT_FOUND => "not_found_error",
            StatusCode::INTERNAL_SERVER_ERROR => "api_error",
            _ => "invalid_request_error",
        };
        let request_id = next_request_id();
        let mut response = (
            self.0.status,
            Json(json!({
                "type":"error",
                "error":{"type":error_type,"message":self.0.message},
                "request_id":request_id.clone()
            })),
        ).into_response();
        if let Ok(value) = HeaderValue::try_from(request_id.as_str()) {
            response.headers_mut().insert("request-id", value);
        }
        response
    }
}

#[derive(Debug, Deserialize)]
struct MessagesRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: usize,
    #[serde(default)]
    stream: bool,
    system: Option<AnthropicContent>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
    tools: Option<Vec<AnthropicTool>>,
    tool_choice: Option<Value>,
    stop_sequences: Option<Vec<String>>,
    #[serde(flatten)]
    unsupported: std::collections::HashMap<String, Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum AnthropicBlock {
    Text { text: String },
    ToolUse { id: String, name: String, input: Value },
    ToolResult {
        tool_use_id: String,
        content: Option<Value>,
        #[serde(default)]
        is_error: bool,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AnthropicTool {
    name: String,
    description: Option<String>,
    input_schema: Value,
}

fn content_to_text(content: AnthropicContent) -> Result<String, ApiError> {
    match content {
        AnthropicContent::Text(text) => Ok(text),
        AnthropicContent::Blocks(blocks) => {
            let mut text = String::new();
            for block in blocks {
                match block {
                    AnthropicBlock::Text { text: part } => text.push_str(&part),
                    _ => return Err(ApiError::bad_request(
                        "system and tool-result content may contain only text blocks",
                    )),
                }
            }
            Ok(text)
        }
    }
}

fn tool_result_text(value: Option<Value>) -> Result<String, ApiError> {
    match value {
        Some(Value::String(text)) => Ok(text),
        Some(Value::Array(blocks)) => {
            let mut text = String::new();
            for block in blocks {
                let part = block.as_object().ok_or_else(|| ApiError::bad_request(
                    "tool_result content blocks must be objects",
                ))?;
                if part.get("type").and_then(Value::as_str) != Some("text") {
                    return Err(ApiError::bad_request(
                        "only text blocks are supported in tool_result content",
                    ));
                }
                let value = part.get("text").and_then(Value::as_str).ok_or_else(|| {
                    ApiError::bad_request("tool_result text block requires string text")
                })?;
                text.push_str(value);
            }
            Ok(text)
        }
        _ => Err(ApiError::bad_request("tool_result requires string or text-block content")),
    }
}

fn normalize_messages(
    system: Option<AnthropicContent>,
    messages: Vec<AnthropicMessage>,
) -> Result<Vec<ChatMessage>, ApiError> {
    let mut normalized = Vec::new();
    if let Some(system) = system {
        let text = content_to_text(system)?;
        if !text.is_empty() {
            normalized.push(ChatMessage {
                role: "system".into(),
                content: Some(text),
                tool_calls: None,
                tool_call_id: None,
                name: None,
                unsupported: Default::default(),
            });
        }
    }

    for message in messages {
        if message.role != "user" && message.role != "assistant" {
            return Err(ApiError::bad_request(
                "Anthropic message roles must be 'user' or 'assistant'",
            ));
        }
        match message.content {
            AnthropicContent::Text(text) => normalized.push(ChatMessage {
                role: message.role,
                content: Some(text),
                tool_calls: None,
                tool_call_id: None,
                name: None,
                unsupported: Default::default(),
            }),
            AnthropicContent::Blocks(blocks) => {
                let mut text = String::new();
                let mut calls = Vec::new();
                let mut results = Vec::new();
                for block in blocks {
                    match block {
                        AnthropicBlock::Text { text: part } => text.push_str(&part),
                        AnthropicBlock::ToolUse { id, name, input } => {
                            if message.role != "assistant" {
                                return Err(ApiError::bad_request(
                                    "tool_use blocks are only valid in assistant messages",
                                ));
                            }
                            if id.trim().is_empty() || name.trim().is_empty() || !input.is_object() {
                                return Err(ApiError::bad_request(
                                    "tool_use requires non-empty id/name and object input",
                                ));
                            }
                            calls.push(OpenAiToolCall {
                                id,
                                kind: "function".into(),
                                function: tool_call::OpenAiFunctionCall {
                                    name,
                                    arguments: input.to_string(),
                                },
                            });
                        }
                        AnthropicBlock::ToolResult { tool_use_id, content, is_error } => {
                            if message.role != "user" {
                                return Err(ApiError::bad_request(
                                    "tool_result blocks are only valid in user messages",
                                ));
                            }
                            if tool_use_id.trim().is_empty() {
                                return Err(ApiError::bad_request("tool_result requires tool_use_id"));
                            }
                            let mut result = tool_result_text(content)?;
                            if is_error {
                                result = format!("Tool execution error: {result}");
                            }
                            results.push(ChatMessage {
                                role: "tool".into(),
                                content: Some(result),
                                tool_calls: None,
                                tool_call_id: Some(tool_use_id),
                                name: None,
                                unsupported: Default::default(),
                            });
                        }
                    }
                }
                if !text.is_empty() || !calls.is_empty() {
                    normalized.push(ChatMessage {
                        role: message.role,
                        content: if text.is_empty() { None } else { Some(text) },
                        tool_calls: if calls.is_empty() { None } else { Some(calls) },
                        tool_call_id: None,
                        name: None,
                        unsupported: Default::default(),
                    });
                }
                normalized.extend(results);
            }
        }
    }
    if normalized.is_empty() {
        return Err(ApiError::bad_request("messages must contain at least one text or tool-result block"));
    }
    Ok(normalized)
}

fn normalize_tools(tools: Option<Vec<AnthropicTool>>) -> Result<Vec<OpenAiTool>, ApiError> {
    let tools = tools.unwrap_or_default();
    let normalized = tools.into_iter().map(|tool| OpenAiTool {
        kind: "function".into(),
        function: tool_call::FunctionDefinition {
            name: tool.name,
            description: tool.description,
            parameters: Some(tool.input_schema),
        },
    }).collect::<Vec<_>>();
    if !normalized.is_empty() {
        validate_tools(&normalized).map_err(ApiError::bad_request)?;
        for tool in &normalized {
            if tool.function.parameters.as_ref().map_or(true, |schema| {
                schema.get("type").and_then(Value::as_str) != Some("object")
            }) {
                return Err(ApiError::bad_request(
                    "Anthropic tool input_schema must be a JSON object schema",
                ));
            }
        }
    }
    Ok(normalized)
}

fn normalize_tool_choice(value: Option<Value>) -> Result<Value, ApiError> {
    let Some(value) = value else { return Ok(json!("auto")); };
    let object = value.as_object().ok_or_else(|| {
        ApiError::bad_request("tool_choice must be an object")
    })?;
    let kind = object.get("type").and_then(Value::as_str).ok_or_else(|| {
        ApiError::bad_request("tool_choice requires a string type")
    })?;
    let allowed = match kind {
        "auto" | "any" => ["type", "disable_parallel_tool_use", "name"].as_slice(),
        "tool" => ["type", "name", "disable_parallel_tool_use"].as_slice(),
        _ => return Err(ApiError::bad_request(
            "tool_choice type must be 'auto', 'any', or 'tool'",
        )),
    };
    if object.keys().any(|key| !allowed.contains(&key.as_str())) {
        return Err(ApiError::bad_request("tool_choice contains unsupported fields"));
    }
    if object.get("disable_parallel_tool_use").and_then(Value::as_bool) == Some(true) {
        return Err(ApiError::bad_request("disable_parallel_tool_use=true is not supported"));
    }
    match kind {
        "auto" => Ok(json!("auto")),
        "any" => Ok(json!("required")),
        "tool" => {
            let name = object.get("name").and_then(Value::as_str)
                .filter(|name| !name.trim().is_empty())
                .ok_or_else(|| ApiError::bad_request("tool_choice type='tool' requires name"))?;
            Ok(json!({"type":"function","function":{"name":name}}))
        }
        _ => unreachable!("tool_choice type was validated above"),
    }
}

fn prepare_request(state: &ApiState, request: MessagesRequest) -> Result<PreparedRequest, ApiError> {
    if request.model != state.assets.model_id && request.model != "qarnyx" {
        return Err(ApiError::not_found(format!(
            "model '{}' is not loaded; available model is '{}'",
            request.model, state.assets.model_id
        )));
    }
    if request.max_tokens == 0 {
        return Err(ApiError::bad_request("max_tokens must be non-zero"));
    }
    if request.stop_sequences.as_ref().is_some_and(|values| !values.is_empty()) {
        return Err(ApiError::bad_request("stop_sequences are not supported by the runtime yet"));
    }
    if !request.unsupported.is_empty() {
        let mut fields = request.unsupported.keys().cloned().collect::<Vec<_>>();
        fields.sort();
        return Err(ApiError::bad_request(format!(
            "unsupported Anthropic request fields: {}", fields.join(", ")
        )));
    }

    let messages = normalize_messages(request.system, request.messages)?;
    validate_chat_messages(&messages)?;
    let tools = normalize_tools(request.tools)?;
    let choice_value = normalize_tool_choice(request.tool_choice)?;
    let tool_choice = parse_tool_choice(Some(&choice_value)).map_err(ApiError::bad_request)?;
    if tools.is_empty() && !matches!(tool_choice, ToolChoice::Auto | ToolChoice::None) {
        return Err(ApiError::bad_request("tool_choice requires at least one tool"));
    }
    validate_choice(&tool_choice, &tools).map_err(ApiError::bad_request)?;

    let prompt = format_chat_prompt(&state.assets.architecture, &messages, &tools, &tool_choice)?;
    let defaults = GenerationConfig::default();
    let config = GenerationConfig {
        max_new_tokens: request.max_tokens,
        max_context_tokens: state.assets.max_context_tokens,
        max_cache_mib: state.assets.max_cache_mib,
        temperature: request.temperature.unwrap_or(defaults.temperature),
        top_k: request.top_k.unwrap_or(defaults.top_k),
        top_p: request.top_p.unwrap_or(defaults.top_p),
        seed: request.seed.unwrap_or(defaults.seed),
    };
    state.prepare_generation(
        GenerationRequest { prompt, config },
        false,
        tools,
        tool_choice,
    )
}

fn stop_reason(parsed: &tool_call::ParsedAssistant, completion_tokens: usize, max_tokens: usize) -> &'static str {
    if !parsed.tool_calls.is_empty() { "tool_use" }
    else if completion_tokens >= max_tokens { "max_tokens" }
    else { "end_turn" }
}

fn anthropic_tool_id(id: &str) -> String {
    format!("toolu_{}", id.strip_prefix("call_qarnyx_").unwrap_or(id))
}

fn content_blocks(parsed: &tool_call::ParsedAssistant) -> Vec<Value> {
    let mut blocks = Vec::new();
    if let Some(text) = &parsed.content {
        if !text.is_empty() {
            blocks.push(json!({"type":"text","text":text}));
        }
    }
    for call in &parsed.tool_calls {
        blocks.push(json!({
            "type":"tool_use",
            "id":anthropic_tool_id(&call.id),
            "name":call.name,
            "input":call.arguments
        }));
    }
    blocks
}

pub(super) async fn create_messages(
    State(state): State<ApiState>,
    payload: Result<Json<MessagesRequest>, JsonRejection>,
) -> Result<Response, AnthropicApiError> {
    let Json(request) = payload.map_err(|error| {
        ApiError::bad_request(format!("invalid JSON request: {error}"))
    })?;
    let stream = request.stream;
    let prepared = prepare_request(&state, request)?;
    let prompt_tokens = prepared.prompt_ids.len();
    let max_tokens = prepared.config.max_new_tokens;
    if stream {
        return stream_messages(state, prepared, prompt_tokens, max_tokens).await;
    }

    let permit = state.assets.request_slots.clone().acquire_owned().await
        .map_err(|error| ApiError::internal(format!("request admission failed: {error}")))?;
    let cancellation = CancellationToken::new();
    let response_tools = prepared.tools.clone();
    let response_tool_choice = prepared.tool_choice.clone();
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
    let parsed = parse_model_output(&output.text, &response_tools, &response_tool_choice)
        .map_err(ApiError::bad_request)?;
    let reason = stop_reason(&parsed, output.completion_tokens, max_tokens);
    let id = format!("msg_{}", NEXT_MESSAGE_ID.fetch_add(1, Ordering::Relaxed));
    Ok(with_request_id(Json(json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": state.assets.model_id,
        "content": content_blocks(&parsed),
        "stop_reason": reason,
        "stop_sequence": null,
        "usage": {
            "input_tokens": output.prompt_tokens,
            "output_tokens": output.completion_tokens
        }
    })).into_response()))
}

async fn stream_messages(
    state: ApiState,
    prepared: PreparedRequest,
    prompt_tokens: usize,
    max_tokens: usize,
) -> Result<Response, AnthropicApiError> {
    let permit = state.assets.request_slots.clone().acquire_owned().await
        .map_err(|error| ApiError::internal(format!("request admission failed: {error}")))?;
    let model = state.assets.model_id.clone();
    let id = format!("msg_{}", NEXT_MESSAGE_ID.fetch_add(1, Ordering::Relaxed));
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
        if !send("message_start", json!({
            "type":"message_start",
            "message":{
                "id":id,"type":"message","role":"assistant","model":model,
                "content":[],"stop_reason":null,"stop_sequence":null,
                "usage":{"input_tokens":prompt_tokens,"output_tokens":0}
            }
        })) { return; }

        let buffer_tools = !prepared.tools.is_empty();
        if !buffer_tools && !send("content_block_start", json!({
            "type":"content_block_start","index":0,
            "content_block":{"type":"text","text":""}
        })) { return; }
        let mut raw = String::new();
        let mut completion_tokens = 0usize;
        let result = state.generate(&prepared, &task_cancellation, |event| {
            match event {
                GenerationEvent::Token { token_id, text_delta } => {
                    completion_tokens += 1;
                    if buffer_tools {
                        raw.push_str(&text_delta);
                    } else if !text_delta.is_empty() {
                        let _ = send("content_block_delta", json!({
                            "type":"content_block_delta","index":0,
                            "delta":{"type":"text_delta","text":text_delta}
                        }));
                    }
                    let _ = token_id;
                }
                GenerationEvent::TextDelta { text } => {
                    if buffer_tools { raw.push_str(&text); }
                    else if !text.is_empty() {
                        let _ = send("content_block_delta", json!({
                            "type":"content_block_delta","index":0,
                            "delta":{"type":"text_delta","text":text}
                        }));
                    }
                }
            }
        });
        if task_cancellation.is_cancelled() { return; }
        match result {
            Ok(generation) => {
                completion_tokens = generation.token_ids.len().saturating_sub(prompt_tokens).max(completion_tokens);
                let text = if buffer_tools { raw } else { String::new() };
                let parsed = match parse_model_output(&text, &prepared.tools, &prepared.tool_choice) {
                    Ok(_parsed) if !buffer_tools => tool_call::ParsedAssistant { content: Some(String::new()), tool_calls: Vec::new() },
                    Ok(parsed) => parsed,
                    Err(error) => {
                        let _ = send("error", json!({"type":"error","error":{"type":"api_error","message":error}}));
                        let _ = send("message_stop", json!({"type":"message_stop"}));
                        return;
                    }
                };
                if buffer_tools {
                    if let Some(content) = parsed.content.as_ref().filter(|value| !value.is_empty()) {
                        let _ = send("content_block_start", json!({
                            "type":"content_block_start","index":0,
                            "content_block":{"type":"text","text":""}
                        }));
                        let _ = send("content_block_delta", json!({
                            "type":"content_block_delta","index":0,
                            "delta":{"type":"text_delta","text":content}
                        }));
                        let _ = send("content_block_stop", json!({"type":"content_block_stop","index":0}));
                    }
                }
                let mut index = if buffer_tools && parsed.content.as_ref().is_some_and(|v| !v.is_empty()) { 1 } else { 0 };
                for call in &parsed.tool_calls {
                    let _ = send("content_block_start", json!({
                        "type":"content_block_start","index":index,
                        "content_block":{"type":"tool_use","id":anthropic_tool_id(&call.id),"name":call.name,"input":{}}
                    }));
                    let _ = send("content_block_delta", json!({
                        "type":"content_block_delta","index":index,
                        "delta":{"type":"input_json_delta","partial_json":call.arguments.to_string()}
                    }));
                    let _ = send("content_block_stop", json!({"type":"content_block_stop","index":index}));
                    index += 1;
                }
                if !buffer_tools {
                    let _ = send("content_block_stop", json!({"type":"content_block_stop","index":0}));
                }
                let reason = stop_reason(&parsed, completion_tokens, max_tokens);
                let _ = send("message_delta", json!({
                    "type":"message_delta",
                    "delta":{"stop_reason":reason,"stop_sequence":null},
                    "usage":{"output_tokens":completion_tokens}
                }));
                let _ = send("message_stop", json!({"type":"message_stop"}));
            }
            Err(error) => {
                let _ = send("error", json!({"type":"error","error":{"type":"api_error","message":error.to_string()}}));
                let _ = send("message_stop", json!({"type":"message_stop"}));
            }
        }
    });

    let response = Response::builder()
        .header(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"))
        .header(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"))
        .header("request-id", next_request_id())
        .body(Body::from_stream(ReceiverStream::new(receiver).map(Ok::<_, Infallible>)))
        .map_err(|error| ApiError::internal(format!("could not construct SSE response: {error}")))?;
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_tool_choice_maps_to_common_choice() {
        assert_eq!(normalize_tool_choice(Some(json!({"type":"auto"}))).unwrap(), json!("auto"));
        assert_eq!(normalize_tool_choice(Some(json!({"type":"any"}))).unwrap(), json!("required"));
        assert_eq!(
            normalize_tool_choice(Some(json!({"type":"tool","name":"weather"}))).unwrap(),
            json!({"type":"function","function":{"name":"weather"}})
        );
    }

    #[test]
    fn anthropic_text_blocks_normalize_to_chat_messages() {
        let messages = normalize_messages(None, vec![AnthropicMessage {
            role: "user".into(),
            content: AnthropicContent::Blocks(vec![AnthropicBlock::Text { text: "Hello".into() }]),
        }]).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content.as_deref(), Some("Hello"));
    }

    #[test]
    fn anthropic_wire_request_deserializes_system_text_tools_and_blocks() {
        let request: MessagesRequest = serde_json::from_value(json!({
            "model":"qarnyx",
            "max_tokens":64,
            "system":"Be concise.",
            "messages":[{"role":"user","content":[{"type":"text","text":"Hello"}]}],
            "tools":[{
                "name":"weather",
                "description":"Get weather",
                "input_schema":{"type":"object","properties":{"city":{"type":"string"}}}
            }],
            "tool_choice":{"type":"tool","name":"weather"}
        })).unwrap();
        let messages = normalize_messages(request.system, request.messages).unwrap();
        let tools = normalize_tools(request.tools).unwrap();
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].content.as_deref(), Some("Hello"));
        assert_eq!(tools[0].function.name, "weather");
        assert_eq!(normalize_tool_choice(request.tool_choice).unwrap(),
            json!({"type":"function","function":{"name":"weather"}}));
    }

    #[test]
    fn anthropic_response_formats_tool_use_blocks() {
        let parsed = tool_call::ParsedAssistant {
            content: None,
            tool_calls: vec![tool_call::ToolCall {
                id: "call_qarnyx_9".into(),
                name: "weather".into(),
                arguments: json!({"city":"Jakarta"}),
            }],
        };
        let blocks = content_blocks(&parsed);
        assert_eq!(blocks[0]["type"], "tool_use");
        assert_eq!(blocks[0]["id"], "toolu_9");
        assert_eq!(blocks[0]["input"]["city"], "Jakarta");
    }

    #[test]
    fn anthropic_message_and_tool_choice_reject_unsupported_fields() {
        let message = serde_json::from_value::<AnthropicMessage>(json!({
            "role":"user","content":"Hello","cache_control":{"type":"ephemeral"}
        }));
        assert!(message.is_err());
        assert!(normalize_tool_choice(Some(json!({
            "type":"auto","unsupported_option":true
        }))).is_err());
        assert!(normalize_tool_choice(Some(json!({
            "type":"any","disable_parallel_tool_use":true
        }))).is_err());
    }

    #[test]
    fn anthropic_tool_use_and_result_keep_call_ids() {
        let messages = normalize_messages(None, vec![
            AnthropicMessage {
                role: "assistant".into(),
                content: AnthropicContent::Blocks(vec![AnthropicBlock::ToolUse {
                    id: "toolu_1".into(), name: "weather".into(), input: json!({"city":"Jakarta"})
                }]),
            },
            AnthropicMessage {
                role: "user".into(),
                content: AnthropicContent::Blocks(vec![AnthropicBlock::ToolResult {
                    tool_use_id: "toolu_1".into(), content: Some(json!("Sunny")), is_error: false
                }]),
            }
        ]).unwrap();
        assert_eq!(messages[0].tool_calls.as_ref().unwrap()[0].id, "toolu_1");
        assert_eq!(anthropic_tool_id("call_qarnyx_7"), "toolu_7");
        assert_eq!(messages[1].tool_call_id.as_deref(), Some("toolu_1"));
        assert_eq!(messages[1].content.as_deref(), Some("Sunny"));
    }
}
