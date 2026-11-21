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
        ToolChoice::Auto => ("You may call zero or more tools when useful.", String::new()),
        ToolChoice::None => ("Do not call tools. Answer directly.", String::new()),
        ToolChoice::Required => ("You must call at least one tool.", String::new()),
        ToolChoice::Function(name) => ("You must call the selected function.", format!(" Selected function: {name}.")),
    };
    format!(
        "Function tools are available. {policy}{selected}\nAvailable tools (OpenAI function schema): {schemas}\n\
         If calling tools, output ONLY this JSON envelope: {{\"tool_calls\":[{{\"name\":\"function_name\",\"arguments\":{{}}}}]}}.\n\
         Arguments must be a JSON object. Multiple calls are allowed. Do not invent tool names.\n\
         If not calling tools, answer with ordinary assistant text."
    )
}

pub fn parse_model_output(output: &str, tools: &[OpenAiTool], choice: &ToolChoice) -> Result<ParsedAssistant, String> {
    let trimmed = output.trim();
    let Ok(value) = serde_json::from_str::<Value>(trimmed) else {
        if matches!(choice, ToolChoice::Required | ToolChoice::Function(_)) {
            return Err("model did not return the required tool-call envelope".into());
        }
        return Ok(ParsedAssistant { content: Some(output.to_owned()), tool_calls: Vec::new() });
    };
    let Some(object) = value.as_object() else {
        return Ok(ParsedAssistant { content: Some(output.to_owned()), tool_calls: Vec::new() });
    };
    let Some(raw_calls) = object.get("tool_calls") else {
        if matches!(choice, ToolChoice::Required | ToolChoice::Function(_)) {
            return Err("model did not return the required tool-call envelope".into());
        }
        return Ok(ParsedAssistant { content: Some(output.to_owned()), tool_calls: Vec::new() });
    };
    if matches!(choice, ToolChoice::None) {
        return Err("model emitted a tool call while tool_choice='none'".into());
    }
    if object.len() != 1 { return Err("tool-call envelope must contain only tool_calls".into()); }
    let raw_calls = raw_calls.as_array().filter(|a| !a.is_empty())
        .ok_or_else(|| "tool_calls must be a non-empty array".to_owned())?;
    let mut parsed = Vec::with_capacity(raw_calls.len());
    for raw in raw_calls {
        let call = raw.as_object().ok_or_else(|| "each tool call must be a JSON object".to_owned())?;
        let name = call.get("name").and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| "each tool call requires a non-empty name".to_owned())?;
        if call.len() != 2 || !call.contains_key("arguments") {
            return Err("each tool call must contain only name and arguments".into());
        }
        let definition = tools.iter().find(|t| t.function.name == name)
            .ok_or_else(|| format!("model requested unknown function '{name}'"))?;
        let arguments = call.get("arguments").cloned().unwrap_or(Value::Null);
        if !arguments.is_object() { return Err(format!("arguments for function '{name}' must be a JSON object")); }
        if let ToolChoice::Function(selected) = choice {
            if selected != name { return Err(format!("model called '{name}' but tool_choice selected '{selected}'")); }
        }
        let _ = definition;
        parsed.push(ToolCall {
            id: format!("call_qarnyx_{}", NEXT_TOOL_CALL_ID.fetch_add(1, Ordering::Relaxed)),
            name: name.to_owned(),
            arguments,
        });
    }
    Ok(ParsedAssistant { content: None, tool_calls: parsed })
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
}
