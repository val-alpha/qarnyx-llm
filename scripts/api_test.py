#!/usr/bin/env python3
"""End-to-end API test runner for OpenAI, Anthropic, and native Qarnyx APIs.

Uses only the Python standard library. It never starts or stops qarnyx-server.
The server must already be running; otherwise the runner exits before tests.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: str


@dataclass
class TestResult:
    name: str
    suite: str
    passed: bool
    detail: str = ""


class TestFailure(Exception):
    """An assertion failure with a readable test-runner message."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TestFailure(message)


def http_request(
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 600.0,
) -> HttpResponse:
    data = None
    headers = {"Accept": "application/json, text/event-stream"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return HttpResponse(
                status=response.status,
                headers={key.lower(): value for key, value in response.headers.items()},
                body=body,
            )
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        return HttpResponse(
            status=error.code,
            headers={key.lower(): value for key, value in error.headers.items()} if error.headers else {},
            body=body,
        )


def json_body(response: HttpResponse) -> dict[str, Any]:
    try:
        value = json.loads(response.body)
    except json.JSONDecodeError as error:
        raise TestFailure(
            f"response is not valid JSON (HTTP {response.status}): {error}; "
            f"body={response.body[:500]!r}"
        ) from error
    require(isinstance(value, dict), "JSON response must be an object")
    return value


def expect_status(response: HttpResponse, expected: int, label: str) -> None:
    require(
        response.status == expected,
        f"{label}: expected HTTP {expected}, got HTTP {response.status}; "
        f"body={response.body[:700]!r}",
    )


def assistant_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "".join(str(part) for part in parts).strip()
    return ""


def parse_sse(raw: str) -> list[dict[str, Any]]:
    """Parse simple SSE frames emitted by the three Qarnyx adapters."""
    frames: list[dict[str, Any]] = []
    event_name = "message"
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal event_name, data_lines
        if data_lines:
            data = "\n".join(data_lines)
            if data == "[DONE]":
                frames.append({"event": event_name, "data": "[DONE]"})
            else:
                try:
                    decoded: Any = json.loads(data)
                except json.JSONDecodeError:
                    decoded = data
                frames.append({"event": event_name, "data": decoded})
        event_name = "message"
        data_lines = []

    for line in raw.splitlines():
        if not line:
            flush()
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event_name = line[6:].strip() or "message"
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    flush()
    return frames


def openai_tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city. Use this tool when requested.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }


def anthropic_tool_definition() -> dict[str, Any]:
    return {
        "name": "get_weather",
        "description": "Get the weather for a city. Use this tool when requested.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    }


def openai_forced_tool_payload(model: str, max_tokens: int, stream: bool = False) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use get_weather for Jakarta now. You must call the tool; "
                    "do not answer in natural language."
                ),
            }
        ],
        "tools": [openai_tool_definition()],
        "tool_choice": {
            "type": "function",
            "function": {"name": "get_weather"},
        },
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }


def anthropic_forced_tool_payload(
    model: str, max_tokens: int, stream: bool = False
) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use get_weather for Jakarta now. You must call the tool; "
                    "do not answer in natural language."
                ),
            }
        ],
        "tools": [anthropic_tool_definition()],
        "tool_choice": {"type": "tool", "name": "get_weather"},
    }


def extract_openai_tool_call(message: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    calls = message.get("tool_calls")
    require(isinstance(calls, list) and bool(calls), "assistant did not return tool_calls")
    call = calls[0]
    require(isinstance(call, dict), "tool call must be an object")
    call_id = call.get("id")
    function = call.get("function")
    require(isinstance(call_id, str) and bool(call_id), "tool call id is missing")
    require(isinstance(function, dict), "tool call function is missing")
    require(
        function.get("name") == "get_weather",
        f"expected get_weather, received {function.get('name')!r}",
    )
    raw_arguments = function.get("arguments")
    require(isinstance(raw_arguments, str), "tool arguments must be a JSON string")
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as error:
        raise TestFailure(f"tool arguments are invalid JSON: {error}") from error
    require(isinstance(arguments, dict), "tool arguments must decode to an object")
    require(arguments.get("city", "").lower() == "jakarta", "tool city must be Jakarta")
    return call_id, arguments


def extract_anthropic_tool_call(body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    blocks = body.get("content")
    require(isinstance(blocks, list), "Anthropic content must be an array")
    calls = [
        block for block in blocks
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]
    require(bool(calls), "assistant did not return a tool_use block")
    call = calls[0]
    call_id = call.get("id")
    require(isinstance(call_id, str) and bool(call_id), "tool_use id is missing")
    require(call.get("name") == "get_weather", "expected get_weather tool_use")
    arguments = call.get("input")
    require(isinstance(arguments, dict), "tool_use input must be an object")
    require(arguments.get("city", "").lower() == "jakarta", "tool city must be Jakarta")
    return call_id, arguments


class ApiTestRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.base_url = args.base_url.rstrip("/")
        self.model = args.model
        self.results: list[TestResult] = []

    def request(
        self, path: str, payload: dict[str, Any] | None = None
    ) -> HttpResponse:
        return http_request(self.base_url, path, payload, self.args.timeout)

    def run_case(self, suite: str, name: str, test: Callable[[], str | None]) -> None:
        started = time.monotonic()
        try:
            detail = test() or ""
            elapsed = time.monotonic() - started
            self.results.append(TestResult(name, suite, True, detail))
            suffix = f" — {detail}" if detail else ""
            print(f"[PASS] {suite}/{name} ({elapsed:.1f}s){suffix}")
        except Exception as error:  # Keep running so one failure does not hide later results.
            elapsed = time.monotonic() - started
            detail = str(error).strip() or error.__class__.__name__
            self.results.append(TestResult(name, suite, False, detail))
            print(f"[FAIL] {suite}/{name} ({elapsed:.1f}s) — {detail}")

    def check_ready(self) -> bool:
        try:
            response = http_request(
                self.base_url, "/health", timeout=min(self.args.timeout, 5.0)
            )
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            print(f"[SKIP] Qarnyx server is not reachable at {self.base_url}")
            print(f"       {error}")
            print("       Server is not started by this program; no tests were run.")
            return False

        if response.status != 200:
            print(
                f"[SKIP] {self.base_url}/health returned HTTP {response.status}; "
                "no tests were run."
            )
            return False
        try:
            health = json.loads(response.body)
        except json.JSONDecodeError:
            print("[SKIP] /health did not return valid JSON; no tests were run.")
            return False
        if not isinstance(health, dict) or health.get("status") != "ok":
            print("[SKIP] Qarnyx health check is not OK; no tests were run.")
            return False
        print(f"[INFO] Server ready at {self.base_url}")
        return True

    def common_tests(self) -> None:
        def models() -> str:
            response = self.request("/v1/models")
            expect_status(response, 200, "list models")
            body = json_body(response)
            entries = body.get("data")
            require(isinstance(entries, list) and bool(entries), "model list is empty")
            ids = [item.get("id") for item in entries if isinstance(item, dict)]
            require(self.model in ids or "qarnyx" == self.model, f"model {self.model!r} not listed: {ids!r}")
            return f"loaded models: {', '.join(str(item) for item in ids)}"

        self.run_case("common", "model-discovery", models)

    def openai_tests(self) -> None:
        def basic_chat() -> str:
            response = self.request(
                "/v1/chat/completions",
                {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a concise assistant."},
                        {"role": "user", "content": "Reply with a short greeting."},
                    ],
                    "max_tokens": self.args.max_tokens,
                    "temperature": 0,
                    "top_p": 1,
                    "seed": 7,
                },
            )
            expect_status(response, 200, "OpenAI basic chat")
            body = json_body(response)
            choices = body.get("choices")
            require(isinstance(choices, list) and bool(choices), "choices is missing")
            message = choices[0].get("message", {})
            require(message.get("role") == "assistant", "assistant role is missing")
            require(bool(assistant_text(message)), "assistant text content is empty")
            require(isinstance(body.get("usage"), dict), "usage is missing")
            return "chat.completion JSON and usage present"

        def multi_turn() -> str:
            response = self.request(
                "/v1/chat/completions",
                {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": "My name is Raka."},
                        {"role": "assistant", "content": "Hello, Raka."},
                        {"role": "user", "content": "What name did I tell you?"},
                    ],
                    "max_tokens": self.args.max_tokens,
                    "temperature": 0,
                },
            )
            expect_status(response, 200, "OpenAI multi-turn chat")
            body = json_body(response)
            message = body["choices"][0]["message"]
            require(bool(assistant_text(message)), "multi-turn assistant content is empty")
            return "multi-message history accepted"

        def stream_chat() -> str:
            response = self.request(
                "/v1/chat/completions",
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Say hello briefly."}],
                    "max_tokens": self.args.max_tokens,
                    "temperature": 0,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            )
            expect_status(response, 200, "OpenAI streaming chat")
            require(
                "text/event-stream" in response.headers.get("content-type", ""),
                "OpenAI stream content-type is not text/event-stream",
            )
            frames = parse_sse(response.body)
            require(any(frame["data"] == "[DONE]" for frame in frames), "SSE [DONE] marker missing")
            chunks = [frame["data"] for frame in frames if isinstance(frame["data"], dict)]
            text = "".join(
                choice.get("delta", {}).get("content", "")
                for chunk in chunks
                for choice in chunk.get("choices", [])
                if isinstance(choice, dict)
            )
            require(bool(text.strip()), "OpenAI stream produced no text delta")
            require(
                any(
                    choice.get("finish_reason") in ("stop", "length")
                    for chunk in chunks
                    for choice in chunk.get("choices", [])
                    if isinstance(choice, dict)
                ),
                "OpenAI stream finish_reason chunk missing",
            )
            require(any("usage" in chunk and chunk["usage"] for chunk in chunks), "stream usage chunk missing")
            return f"{len(frames)} SSE frames, {len(text)} text characters"

        def forced_tool() -> str:
            response = self.request(
                "/v1/chat/completions",
                openai_forced_tool_payload(self.model, self.args.tool_max_tokens),
            )
            expect_status(response, 200, "OpenAI forced tool call")
            body = json_body(response)
            choice = body["choices"][0]
            require(choice.get("finish_reason") == "tool_calls", "finish_reason is not tool_calls")
            call_id, arguments = extract_openai_tool_call(choice.get("message", {}))
            self.openai_tool_call_id = call_id
            self.openai_tool_arguments = arguments
            return f"get_weather({json.dumps(arguments, ensure_ascii=False)})"

        def streamed_tool() -> str:
            response = self.request(
                "/v1/chat/completions",
                openai_forced_tool_payload(self.model, self.args.tool_max_tokens, stream=True),
            )
            expect_status(response, 200, "OpenAI streamed tool call")
            frames = parse_sse(response.body)
            require(any(frame["data"] == "[DONE]" for frame in frames), "tool stream [DONE] missing")
            collected: dict[int, dict[str, str]] = {}
            finish_reasons = []
            for frame in frames:
                chunk = frame["data"]
                if not isinstance(chunk, dict):
                    continue
                for choice in chunk.get("choices", []):
                    if choice.get("finish_reason") is not None:
                        finish_reasons.append(choice["finish_reason"])
                    delta = choice.get("delta", {})
                    for item in delta.get("tool_calls", []):
                        index = int(item.get("index", 0))
                        target = collected.setdefault(index, {"id": "", "name": "", "arguments": ""})
                        target["id"] += item.get("id", "")
                        function = item.get("function", {})
                        target["name"] += function.get("name", "")
                        target["arguments"] += function.get("arguments", "")
            require(bool(collected), "stream did not emit tool_calls deltas")
            call = collected[min(collected)]
            require(call["name"] == "get_weather", f"streamed tool name was {call['name']!r}")
            args = json.loads(call["arguments"])
            require(args.get("city", "").lower() == "jakarta", "streamed tool city was not Jakarta")
            require("tool_calls" in finish_reasons, "streamed finish_reason is not tool_calls")
            return f"streamed {call['name']} arguments"

        def tool_round_trip() -> str:
            first = self.request(
                "/v1/chat/completions",
                openai_forced_tool_payload(self.model, self.args.tool_max_tokens),
            )
            expect_status(first, 200, "OpenAI tool round-trip setup")
            body = json_body(first)
            message = body["choices"][0]["message"]
            call_id, arguments = extract_openai_tool_call(message)
            second = self.request(
                "/v1/chat/completions",
                {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is the weather in Jakarta?",
                        },
                        {"role": "assistant", "content": None, "tool_calls": message["tool_calls"]},
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": "Sunny, 31 C",
                        },
                    ],
                    "max_tokens": self.args.max_tokens,
                    "temperature": 0,
                },
            )
            expect_status(second, 200, "OpenAI tool_result round trip")
            final = json_body(second)
            final_message = final["choices"][0]["message"]
            text = assistant_text(final_message)
            require(bool(text), "assistant returned no final text after tool result")
            require(not final_message.get("tool_calls"), "assistant unexpectedly requested another tool")
            require(
                "31" in text or "sunny" in text.lower(),
                f"assistant response did not use the tool result: {text!r}",
            )
            return "tool result accepted; assistant completed the turn"

        def invalid_request() -> str:
            response = self.request(
                "/v1/chat/completions",
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "hello"}],
                    "n": 2,
                },
            )
            expect_status(response, 400, "OpenAI unsupported n")
            body = json_body(response)
            require(isinstance(body.get("error"), dict), "OpenAI error envelope is missing")
            return "unsupported n=2 rejected with invalid_request_error"

        def missing_model() -> str:
            response = self.request(
                "/v1/chat/completions",
                {
                    "model": "qarnyx-test-model-that-is-not-loaded",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 8,
                },
            )
            expect_status(response, 404, "OpenAI missing model")
            return "unknown model rejected with HTTP 404"

        tests = [
            ("basic-chat", basic_chat),
            ("multi-turn", multi_turn),
            ("streaming", stream_chat),
            ("forced-tool-call", forced_tool),
            ("streaming-tool-call", streamed_tool),
            ("tool-result-round-trip", tool_round_trip),
            ("request-validation", invalid_request),
            ("unknown-model", missing_model),
        ]
        for name, test in tests:
            self.run_case("openai", name, test)

    def anthropic_tests(self) -> None:
        def basic_messages() -> str:
            response = self.request(
                "/v1/messages",
                {
                    "model": self.model,
                    "max_tokens": self.args.max_tokens,
                    "temperature": 0,
                    "system": "You are a concise assistant.",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "Reply with a short greeting."}],
                        }
                    ],
                },
            )
            expect_status(response, 200, "Anthropic basic messages")
            body = json_body(response)
            require(body.get("type") == "message", "Anthropic type is not message")
            require(body.get("role") == "assistant", "Anthropic role is not assistant")
            require(bool(assistant_text(body)), "Anthropic text content is empty")
            require(isinstance(body.get("usage"), dict), "Anthropic usage is missing")
            return "message JSON, text block, and usage present"

        def multi_turn() -> str:
            response = self.request(
                "/v1/messages",
                {
                    "model": self.model,
                    "max_tokens": self.args.max_tokens,
                    "temperature": 0,
                    "messages": [
                        {"role": "user", "content": "My favorite color is blue."},
                        {"role": "assistant", "content": "Got it."},
                        {"role": "user", "content": "What is my favorite color?"},
                    ],
                },
            )
            expect_status(response, 200, "Anthropic multi-turn messages")
            body = json_body(response)
            require(bool(assistant_text(body)), "Anthropic multi-turn content is empty")
            return "multi-message history accepted"

        def stream_messages() -> str:
            response = self.request(
                "/v1/messages",
                {
                    "model": self.model,
                    "max_tokens": self.args.max_tokens,
                    "temperature": 0,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Say hello briefly."}],
                },
            )
            expect_status(response, 200, "Anthropic streaming messages")
            require(
                "text/event-stream" in response.headers.get("Content-Type", ""),
                "Anthropic stream content-type is not text/event-stream",
            )
            frames = parse_sse(response.body)
            events = [frame["event"] for frame in frames]
            require("message_start" in events, "message_start event missing")
            require("content_block_start" in events, "content_block_start event missing")
            require("content_block_delta" in events, "content_block_delta event missing")
            require("message_delta" in events, "message_delta event missing")
            require("message_stop" in events, "message_stop event missing")
            text = "".join(
                frame["data"].get("delta", {}).get("text", "")
                for frame in frames
                if frame["event"] == "content_block_delta"
                and isinstance(frame["data"], dict)
            )
            require(bool(text.strip()), "Anthropic stream produced no text delta")
            return f"{len(frames)} events, {len(text)} text characters"

        def forced_tool() -> str:
            response = self.request(
                "/v1/messages",
                anthropic_forced_tool_payload(self.model, self.args.tool_max_tokens),
            )
            expect_status(response, 200, "Anthropic forced tool call")
            body = json_body(response)
            require(body.get("stop_reason") == "tool_use", "stop_reason is not tool_use")
            call_id, arguments = extract_anthropic_tool_call(body)
            self.anthropic_tool_call_id = call_id
            self.anthropic_tool_arguments = arguments
            return f"tool_use {call_id}: {json.dumps(arguments, ensure_ascii=False)}"

        def streamed_tool() -> str:
            response = self.request(
                "/v1/messages",
                anthropic_forced_tool_payload(
                    self.model, self.args.tool_max_tokens, stream=True
                ),
            )
            expect_status(response, 200, "Anthropic streamed tool call")
            frames = parse_sse(response.body)
            events = [frame["event"] for frame in frames]
            require("message_start" in events, "tool stream message_start missing")
            require("message_delta" in events, "tool stream message_delta missing")
            require("message_stop" in events, "tool stream message_stop missing")
            started: dict[int, dict[str, str]] = {}
            for frame in frames:
                value = frame["data"]
                if not isinstance(value, dict):
                    continue
                if frame["event"] == "content_block_start":
                    block = value.get("content_block", {})
                    if block.get("type") == "tool_use":
                        index = int(value.get("index", 0))
                        started[index] = {
                            "id": str(block.get("id", "")),
                            "name": str(block.get("name", "")),
                            "arguments": "",
                        }
                elif frame["event"] == "content_block_delta":
                    index = int(value.get("index", 0))
                    delta = value.get("delta", {})
                    if index in started and delta.get("type") == "input_json_delta":
                        started[index]["arguments"] += delta.get("partial_json", "")
            require(bool(started), "Anthropic stream did not start a tool_use block")
            call = started[min(started)]
            require(call["name"] == "get_weather", f"streamed tool name was {call['name']!r}")
            args = json.loads(call["arguments"])
            require(args.get("city", "").lower() == "jakarta", "streamed tool city was not Jakarta")
            return f"streamed tool_use {call['name']} with input JSON"

        def tool_result_round_trip() -> str:
            first = self.request(
                "/v1/messages",
                anthropic_forced_tool_payload(self.model, self.args.tool_max_tokens),
            )
            expect_status(first, 200, "Anthropic round-trip setup")
            body = json_body(first)
            call_id, _ = extract_anthropic_tool_call(body)
            second = self.request(
                "/v1/messages",
                {
                    "model": self.model,
                    "max_tokens": self.args.max_tokens,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is the weather in Jakarta?",
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": call_id,
                                    "name": "get_weather",
                                    "input": {"city": "Jakarta"},
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": call_id,
                                    "content": "Sunny, 31 C",
                                }
                            ],
                        },
                    ],
                    "tools": [anthropic_tool_definition()],
                    "tool_choice": {"type": "auto"},
                },
            )
            expect_status(second, 200, "Anthropic tool_result round trip")
            final = json_body(second)
            require(final.get("stop_reason") == "end_turn", "Anthropic did not end the tool-result turn")
            text = assistant_text(final)
            require(bool(text), "Anthropic final text after tool_result is empty")
            require(
                "31" in text or "sunny" in text.lower(),
                f"Anthropic response did not use tool result: {text!r}",
            )
            return "tool_result accepted; assistant completed the turn"

        def invalid_request() -> str:
            response = self.request(
                "/v1/messages",
                {
                    "model": self.model,
                    "max_tokens": self.args.max_tokens,
                    "messages": [{"role": "user", "content": "hello"}],
                    "stop_sequences": ["STOP"],
                },
            )
            expect_status(response, 400, "Anthropic unsupported stop_sequences")
            body = json_body(response)
            require(body.get("type") == "error", "Anthropic error type envelope is missing")
            require(isinstance(body.get("error"), dict), "Anthropic error payload is missing")
            return "unsupported stop_sequences rejected with invalid_request_error"

        def missing_model() -> str:
            response = self.request(
                "/v1/messages",
                {
                    "model": "qarnyx-test-model-that-is-not-loaded",
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            expect_status(response, 404, "Anthropic missing model")
            return "unknown model rejected with HTTP 404"

        tests = [
            ("basic-messages", basic_messages),
            ("multi-turn", multi_turn),
            ("streaming", stream_messages),
            ("forced-tool-call", forced_tool),
            ("streaming-tool-call", streamed_tool),
            ("tool-result-round-trip", tool_result_round_trip),
            ("request-validation", invalid_request),
            ("unknown-model", missing_model),
        ]
        for name, test in tests:
            self.run_case("anthropic", name, test)

    def native_tests(self) -> None:
        def basic_generate() -> str:
            response = self.request(
                "/v1/generate",
                {
                    "model": self.model,
                    "prompt": "Write one short sentence about the sky.",
                    "config": {
                        "max_new_tokens": self.args.max_tokens,
                        "temperature": 0,
                        "top_k": 20,
                        "top_p": 1,
                        "seed": 7,
                    },
                },
            )
            expect_status(response, 200, "Native generation")
            body = json_body(response)
            require(body.get("object") == "qarnyx.generation", "native object type is wrong")
            require(isinstance(body.get("text"), str) and body["text"].strip(), "native text is empty")
            require(isinstance(body.get("token_ids"), list), "native token_ids is missing")
            usage = body.get("usage")
            require(isinstance(usage, dict), "native usage is missing")
            require(usage.get("total_tokens") == usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0), "native token usage totals do not add up")
            require(body.get("finish_reason") in ("stop", "length"), "native finish_reason is invalid")
            return f"{usage['completion_tokens']} generated tokens"

        def stream_generate() -> str:
            response = self.request(
                "/v1/generate",
                {
                    "model": self.model,
                    "prompt": "Write one short sentence about rain.",
                    "stream": True,
                    "config": {
                        "max_new_tokens": self.args.max_tokens,
                        "temperature": 0,
                        "top_k": 20,
                        "top_p": 1,
                        "seed": 8,
                    },
                },
            )
            expect_status(response, 200, "Native streaming generation")
            require(
                "text/event-stream" in response.headers.get("Content-Type", ""),
                "native stream content-type is not text/event-stream",
            )
            frames = parse_sse(response.body)
            events = [frame["event"] for frame in frames]
            require("token" in events or "text_delta" in events, "native stream emitted no token/text events")
            require("completed" in events, "native completed event missing")
            require("done" in events, "native done event missing")
            completed = next(
                frame["data"] for frame in frames
                if frame["event"] == "completed" and isinstance(frame["data"], dict)
            )
            require(bool(completed.get("text", "").strip()), "native completed text is empty")
            require(isinstance(completed.get("token_ids"), list), "native completed token_ids missing")
            return f"{len(frames)} native events"

        def empty_prompt() -> str:
            response = self.request(
                "/v1/generate",
                {"model": self.model, "prompt": "", "config": {"max_new_tokens": 8}},
            )
            expect_status(response, 400, "Native empty prompt")
            body = json_body(response)
            require(isinstance(body.get("error"), dict), "native error payload is missing")
            return "empty prompt rejected with invalid_request"

        def invalid_config() -> str:
            response = self.request(
                "/v1/generate",
                {
                    "model": self.model,
                    "prompt": "hello",
                    "config": {"max_new_tokens": 0},
                },
            )
            expect_status(response, 400, "Native zero max_new_tokens")
            return "zero max_new_tokens rejected"

        def missing_model() -> str:
            response = self.request(
                "/v1/generate",
                {
                    "model": "qarnyx-test-model-that-is-not-loaded",
                    "prompt": "hello",
                    "config": {"max_new_tokens": 8},
                },
            )
            expect_status(response, 404, "Native missing model")
            return "unknown model rejected with HTTP 404"

        tests = [
            ("basic-generation", basic_generate),
            ("streaming", stream_generate),
            ("empty-prompt-validation", empty_prompt),
            ("generation-config-validation", invalid_config),
            ("unknown-model", missing_model),
        ]
        for name, test in tests:
            self.run_case("native", name, test)

    def print_summary(self, elapsed: float) -> int:
        passed = sum(1 for result in self.results if result.passed)
        failed = len(self.results) - passed
        print()
        print("=" * 72)
        print(f"API TEST SUMMARY: {passed} passed, {failed} failed, {elapsed:.1f}s")
        print(f"Target: {self.base_url} | model request name: {self.model}")
        print("=" * 72)
        if failed:
            print("Failed tests:")
            for result in self.results:
                if not result.passed:
                    print(f"  - {result.suite}/{result.name}: {result.detail}")
            return 1
        print("All selected API tests passed.")
        return 0

    def run(self) -> int:
        if not self.check_ready():
            return 0

        started = time.monotonic()
        self.common_tests()
        selected = self.args.suite
        if selected in ("all", "openai"):
            self.openai_tests()
        if selected in ("all", "anthropic"):
            self.anthropic_tests()
        if selected in ("all", "native"):
            self.native_tests()
        return self.print_summary(time.monotonic() - started)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end OpenAI, Anthropic, and native Qarnyx API tests."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Qarnyx server base URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--model",
        default="qarnyx",
        help="model request name; qarnyx is the server alias (default: qarnyx)",
    )
    parser.add_argument(
        "--suite",
        choices=("all", "openai", "anthropic", "native"),
        default="all",
        help="test one API suite or all suites (default: all)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=48,
        help="completion token limit for regular requests (default: 48)",
    )
    parser.add_argument(
        "--tool-max-tokens",
        type=int,
        default=128,
        help="completion token limit for forced tool-call requests (default: 128)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="per-request timeout in seconds (default: 600)",
    )
    args = parser.parse_args()
    if args.max_tokens < 1 or args.tool_max_tokens < 1 or args.timeout <= 0:
        parser.error("token limits and timeout must be greater than zero")
    return args


def main() -> int:
    args = parse_args()
    return ApiTestRunner(args).run()


if __name__ == "__main__":
    sys.exit(main())
