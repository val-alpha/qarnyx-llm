//! Model-neutral tool-calling types and the Qarnyx JSON tool envelope.
//! OpenAI is the HTTP boundary; model adapters translate to/from this neutral IR.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TOOL_CALL_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct OpenAiTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionDefinition,
}
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<Value>,
}
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct OpenAiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: OpenAiFunctionCall,
}
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct OpenAiFunctionCall {
    pub name: String,
    pub arguments: String,
}
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}
#[derive(Debug, Clone, PartialEq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Function(String),
}
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedAssistant {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

impl ToolCall {
    pub fn to_openai(&self) -> OpenAiToolCall {
        OpenAiToolCall {
            id: self.id.clone(),
            kind: "function".into(),
            function: OpenAiFunctionCall {
                name: self.name.clone(),
                arguments: self.arguments.to_string(),
            },
        }
    }
}

pub fn parse_tool_choice(value: Option<&Value>) -> Result<ToolChoice, String> {
    let Some(value) = value else { return Ok(ToolChoice::Auto); };
    match value {
        Value::String(value) if value == "auto" => Ok(ToolChoice::Auto),
        Value::String(value) if value == "none" => Ok(ToolChoice::None),
        Value::String(value) if value == "required" => Ok(ToolChoice::Required),
        Value::Object(object) => {
            if object.get("type").and_then(Value::as_str) != Some("function") {
                return Err("tool_choice object must have type='function'".into());
            }
            let name = object.get("function").and_then(Value::as_object)
                .and_then(|f| f.get("name")).and_then(Value::as_str)
                .filter(|s| !s.trim().is_empty())
                .ok_or_else(|| "tool_choice.function.name must be a non-empty string".to_owned())?;
            Ok(ToolChoice::Function(name.to_owned()))
        }
        _ => Err("tool_choice must be 'auto', 'none', 'required', or a function selector".into()),
    }
}

pub fn validate_tools(tools: &[OpenAiTool]) -> Result<(), String> {
    if tools.is_empty() { return Err("tools must contain at least one function".into()); }
    let mut names = std::collections::HashSet::new();
    for tool in tools {
        if tool.kind != "function" { return Err("only tools with type='function' are supported".into()); }
        let name = tool.function.name.trim();
        if name.is_empty() || name.len() > 64 {
            return Err("function names must contain between 1 and 64 characters".into());
        }
        if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
            return Err(format!("function name '{name}' contains unsupported characters"));
        }
        if !names.insert(name.to_owned()) { return Err(format!("duplicate function name '{name}'")); }
        if tool.function.parameters.as_ref().is_some_and(|v| !v.is_object()) {
            return Err(format!("parameters for function '{name}' must be a JSON object"));
        }
    }
    Ok(())
}

pub fn validate_choice(choice: &ToolChoice, tools: &[OpenAiTool]) -> Result<(), String> {
    if let ToolChoice::Function(name) = choice {
        if !tools.iter().any(|tool| tool.function.name == *name) {
            return Err(format!("tool_choice references unknown function '{name}'"));
        }
    }
    Ok(())
}

pub fn tool_protocol_instruction(tools: &[OpenAiTool], choice: &ToolChoice) -> String {
    let schemas = serde_json::to_string(tools).unwrap_or_else(|_| "[]".into());
    let (policy, selected) = match choice {
        ToolChoice::Auto => (
            "Use a tool when it is needed to answer accurately. You may answer directly when no tool is needed.",
            String::new(),
        ),
        ToolChoice::None => (
            "Do not call any tool. Answer directly in ordinary assistant text.",
            String::new(),
        ),
        ToolChoice::Required => (
            "You MUST call at least one listed tool. Do not answer with ordinary text.",
            String::new(),
        ),
        ToolChoice::Function(name) => (
            "You MUST call exactly the selected function. Do not answer with ordinary text.",
            format!(" The selected function name is '{name}'."),
        ),
    };
    format!(
        "You are connected to Qarnyx function tools. {policy}{selected}\n\
         Available tools and JSON schemas: {schemas}\n\
         When calling a tool, return only a JSON object with this exact outer shape: {{\"tool_calls\":[{{\"name\":\"function_name\",\"arguments\":{{}}}}]}}.\n\
         Each call must use a listed function name and a JSON object for arguments. Arguments must satisfy that function's schema. Never invent a tool name.\n\
         Do not use markdown, code fences, Qwen tool tags, or explanatory text around the JSON object.\n\
         For tool_choice='tool', the call name must match the selected function exactly. For tool_choice='any' or 'required', call at least one listed function.\n\
         If the policy allows a direct answer and no tool is needed, return ordinary assistant text instead of a tool-call object."
    )
}

fn parse_tool_call_arguments(value: &Value, name: &str) -> Result<Value, String> {
    let arguments = match value {
        Value::String(encoded) => serde_json::from_str::<Value>(encoded)
            .map_err(|error| format!("arguments for function '{name}' are not valid JSON: {error}"))?,
        other => other.clone(),
    };
    if !arguments.is_object() {
        return Err(format!("arguments for function '{name}' must be a JSON object"));
    }
    Ok(arguments)
}

fn parse_one_tool_call(
    raw: &Value,
    tools: &[OpenAiTool],
    choice: &ToolChoice,
) -> Result<ToolCall, String> {
    let call = raw
        .as_object()
        .ok_or_else(|| "each tool call must be a JSON object".to_owned())?;

    let (name, raw_arguments) = if let Some(function_value) = call.get("function") {
        if call.keys().any(|key| !["id", "type", "function"].contains(&key.as_str())) {
            return Err("OpenAI-shaped tool calls may contain only id, type, and function".into());
        }
        if call.get("type").and_then(Value::as_str).is_some_and(|kind| kind != "function") {
            return Err("tool call type must be 'function'".into());
        }
        let function = function_value
            .as_object()
            .ok_or_else(|| "tool call function must be an object".to_owned())?;
        if function.keys().any(|key| !["name", "arguments"].contains(&key.as_str())) {
            return Err("tool call function may contain only name and arguments".into());
        }
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.trim().is_empty())
            .ok_or_else(|| "each tool call requires a non-empty function name".to_owned())?;
        let arguments = function
            .get("arguments")
            .ok_or_else(|| format!("tool call '{name}' is missing arguments"))?;
        (name, arguments)
    } else {
        if call
            .keys()
            .any(|key| !["id", "type", "name", "arguments"].contains(&key.as_str()))
        {
            return Err("tool calls may contain only id, type, name, and arguments".into());
        }
        if call.get("type").and_then(Value::as_str).is_some_and(|kind| kind != "function") {
            return Err("tool call type must be 'function'".into());
        }
        let name = call
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.trim().is_empty())
            .ok_or_else(|| "each tool call requires a non-empty name".to_owned())?;
        let arguments = call
            .get("arguments")
            .ok_or_else(|| format!("tool call '{name}' is missing arguments"))?;
        (name, arguments)
    };

    if !tools.iter().any(|tool| tool.function.name == name) {
        return Err(format!("model requested unknown function '{name}'"));
    }
    if let ToolChoice::Function(selected) = choice {
        if selected != name {
            return Err(format!("model called '{name}' but tool_choice selected '{selected}'"));
        }
    }

    Ok(ToolCall {
        id: format!("call_qarnyx_{}", NEXT_TOOL_CALL_ID.fetch_add(1, Ordering::Relaxed)),
        name: name.to_owned(),
        arguments: parse_tool_call_arguments(raw_arguments, name)?,
    })
}

fn payload_calls(value: Value) -> Result<Option<Vec<Value>>, String> {
    let Some(object) = value.as_object() else {
        return Ok(None);
    };

    if let Some(raw_calls) = object.get("tool_calls") {
        if object.len() != 1 {
            return Err("tool-call envelope must contain only tool_calls".into());
        }
        let calls = raw_calls
            .as_array()
            .filter(|calls| !calls.is_empty())
            .ok_or_else(|| "tool_calls must be a non-empty array".to_owned())?;
        return Ok(Some(calls.clone()));
    }

    if object.contains_key("name") || object.contains_key("function") {
        return Ok(Some(vec![value]));
    }
    Ok(None)
}

fn parse_tagged_tool_calls(output: &str) -> Result<Option<(Vec<Value>, String)>, String> {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    if !output.contains(OPEN) {
        return Ok(None);
    }

    let mut remainder = output;
    let mut calls = Vec::new();
    let mut surrounding_text = String::new();
    while let Some(start) = remainder.find(OPEN) {
        surrounding_text.push_str(&remainder[..start]);
        let after_open = &remainder[start + OPEN.len()..];
        let end = after_open
            .find(CLOSE)
            .ok_or_else(|| "model returned an unterminated <tool_call> block".to_owned())?;
        let encoded = after_open[..end].trim();
        let value = serde_json::from_str::<Value>(encoded)
            .map_err(|error| format!("model returned invalid JSON inside <tool_call>: {error}"))?;
        let block_calls = payload_calls(value)?
            .ok_or_else(|| "<tool_call> block did not contain a recognized function call".to_owned())?;
        calls.extend(block_calls);
        remainder = &after_open[end + CLOSE.len()..];
    }

    if remainder.contains(CLOSE) {
        return Err("model returned a closing </tool_call> without an opening tag".into());
    }
    surrounding_text.push_str(remainder);
    if calls.is_empty() {
        return Err("model returned an empty tool-call block".into());
    }
    Ok(Some((calls, surrounding_text.trim().to_owned())))
}

fn strip_json_code_fence(output: &str) -> &str {
    let trimmed = output.trim();
    let fence = "\x60\x60\x60";
    if !trimmed.starts_with(fence) || !trimmed.ends_with(fence) {
        return trimmed;
    }
    let after_open = trimmed.strip_prefix(fence).unwrap_or(trimmed);
    let without_language = after_open
        .strip_prefix("json")
        .or_else(|| after_open.strip_prefix("JSON"))
        .unwrap_or(after_open);
    without_language.strip_suffix(fence).unwrap_or(without_language).trim()
}

pub fn parse_model_output(
    output: &str,
    tools: &[OpenAiTool],
    choice: &ToolChoice,
) -> Result<ParsedAssistant, String> {
    let trimmed = output.trim();
    let (raw_calls, content) = if let Some((calls, surrounding_text)) =
        parse_tagged_tool_calls(trimmed)?
    {
        (Some(calls), surrounding_text)
    } else {
        let candidate = strip_json_code_fence(trimmed);
        match serde_json::from_str::<Value>(candidate) {
            Ok(value) => (payload_calls(value)?, String::new()),
            Err(error) => {
                let looks_like_tool_json = candidate.contains("\"tool_calls\"");
                if matches!(choice, ToolChoice::Required | ToolChoice::Function(_))
                    || looks_like_tool_json
                {
                    return Err(format!(
                        "model did not return a valid tool-call JSON object: {error}"
                    ));
                }
                return Ok(ParsedAssistant {
                    content: Some(output.to_owned()),
                    tool_calls: Vec::new(),
                });
            }
        }
    };

    let Some(raw_calls) = raw_calls else {
        if matches!(choice, ToolChoice::Required | ToolChoice::Function(_)) {
            return Err("model did not return the required tool-call envelope".into());
        }
        return Ok(ParsedAssistant {
            content: Some(output.to_owned()),
            tool_calls: Vec::new(),
        });
    };

    if matches!(choice, ToolChoice::None) {
        return Err("model emitted a tool call while tool_choice='none'".into());
    }

    let parsed = raw_calls
        .iter()
        .map(|call| parse_one_tool_call(call, tools, choice))
        .collect::<Result<Vec<_>, _>>()?;
    if parsed.is_empty() {
        if matches!(choice, ToolChoice::Required | ToolChoice::Function(_)) {
            return Err("model did not return the required tool call".into());
        }
        return Ok(ParsedAssistant {
            content: Some(output.to_owned()),
            tool_calls: Vec::new(),
        });
    }

    Ok(ParsedAssistant {
        content: if content.is_empty() { None } else { Some(content) },
        tool_calls: parsed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn weather() -> OpenAiTool {
        serde_json::from_value(json!({"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}})).unwrap()
    }
    #[test]
    fn parses_tool_choice_variants() {
        assert_eq!(parse_tool_choice(None).unwrap(), ToolChoice::Auto);
        assert_eq!(parse_tool_choice(Some(&json!("none"))).unwrap(), ToolChoice::None);
        assert_eq!(parse_tool_choice(Some(&json!("required"))).unwrap(), ToolChoice::Required);
        assert_eq!(parse_tool_choice(Some(&json!({"type":"function","function":{"name":"get_weather"}}))).unwrap(), ToolChoice::Function("get_weather".into()));
    }
    #[test]
    fn validates_tools_and_selected_function() {
        let tools = vec![weather()];
        validate_tools(&tools).unwrap();
        validate_choice(&ToolChoice::Function("get_weather".into()), &tools).unwrap();
        assert!(validate_choice(&ToolChoice::Function("missing".into()), &tools).is_err());
    }
    #[test]
    fn parses_neutral_envelope_and_serializes_openai_arguments() {
        let parsed = parse_model_output(r#"{"tool_calls":[{"name":"get_weather","arguments":{"city":"Jakarta"}}]}"#, &[weather()], &ToolChoice::Auto).unwrap();
        assert_eq!(parsed.content, None);
        assert_eq!(parsed.tool_calls[0].arguments["city"], "Jakarta");
        let call = parsed.tool_calls[0].to_openai();
        assert_eq!(call.kind, "function");
        assert_eq!(serde_json::from_str::<Value>(&call.function.arguments).unwrap()["city"], "Jakarta");
    }
    #[test]
    fn rejects_unknown_function_and_non_object_arguments() {
        let tools = vec![weather()];
        assert!(parse_model_output(r#"{"tool_calls":[{"name":"missing","arguments":{}}]}"#, &tools, &ToolChoice::Auto).is_err());
        assert!(parse_model_output(r#"{"tool_calls":[{"name":"get_weather","arguments":"x"}]}"#, &tools, &ToolChoice::Auto).is_err());
    }
    #[test]
    fn required_choice_rejects_plain_text() {
        assert!(parse_model_output("hello", &[weather()], &ToolChoice::Required).is_err());
    }

    #[test]
    fn prompt_makes_forced_tool_choice_explicit() {
        let instruction =
            tool_protocol_instruction(&[weather()], &ToolChoice::Function("get_weather".into()));
        assert!(instruction.contains("MUST call exactly the selected function"));
        assert!(instruction.contains(r#"{"tool_calls":[{"name":"function_name","arguments":{}}]}"#));
        assert!(instruction.contains("Do not use markdown"));
    }

    #[test]
    fn parses_qwen_tool_call_tags_with_surrounding_text() {
        let parsed = parse_model_output(
            r#"Checking now. <tool_call>{"name":"get_weather","arguments":{"city":"Jakarta"}}</tool_call>"#,
            &[weather()],
            &ToolChoice::Function("get_weather".into()),
        )
        .unwrap();
        assert_eq!(parsed.content.as_deref(), Some("Checking now."));
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].arguments["city"], "Jakarta");
    }

    #[test]
    fn parses_single_call_and_string_encoded_arguments() {
        let parsed = parse_model_output(
            r#"{"name":"get_weather","arguments":"{\"city\":\"Jakarta\"}"}"#,
            &[weather()],
            &ToolChoice::Function("get_weather".into()),
        )
        .unwrap();
        assert_eq!(parsed.tool_calls[0].arguments["city"], "Jakarta");
    }

    #[test]
    fn parses_openai_shaped_function_call_output() {
        let parsed = parse_model_output(
            r#"{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Jakarta\"}"}}]}"#,
            &[weather()],
            &ToolChoice::Required,
        )
        .unwrap();
        assert_eq!(parsed.tool_calls[0].arguments["city"], "Jakarta");
    }

    #[test]
    fn auto_choice_preserves_text_and_none_rejects_calls() {
        let tools = [weather()];
        let plain = parse_model_output("It is sunny.", &tools, &ToolChoice::Auto).unwrap();
        assert_eq!(plain.content.as_deref(), Some("It is sunny."));
        assert!(parse_model_output(
            r#"{"tool_calls":[{"name":"get_weather","arguments":{"city":"Jakarta"}}]}"#,
            &tools,
            &ToolChoice::None
        )
        .is_err());
    }

    #[test]
    fn malformed_tool_calls_and_forced_plain_text_are_rejected() {
        let tools = [weather()];
        assert!(parse_model_output("I will check the weather.", &tools, &ToolChoice::Required).is_err());
        assert!(parse_model_output(
            r#"{"tool_calls":[{"name":"get_weather","arguments":"not-json"}]}"#,
            &tools,
            &ToolChoice::Auto
        )
        .is_err());
        assert!(parse_model_output(
            "<tool_call>{bad json}</tool_call>",
            &tools,
            &ToolChoice::Auto
        )
        .is_err());
        assert!(parse_model_output(
            "<tool_call>{\"name\":\"get_weather\"}",
            &tools,
            &ToolChoice::Auto
        )
        .is_err());
    }

    #[test]
    fn selected_tool_and_tool_name_must_match_request() {
        let tools = [weather()];
        assert!(parse_model_output(
            r#"{"tool_calls":[{"name":"unknown","arguments":{}}]}"#,
            &tools,
            &ToolChoice::Auto
        )
        .is_err());
        assert!(parse_model_output(
            r#"{"tool_calls":[{"name":"get_weather","arguments":{"city":"Jakarta"}}]}"#,
            &tools,
            &ToolChoice::Function("another_tool".into())
        )
        .is_err());
    }


}
