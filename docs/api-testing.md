# Qarnyx API end-to-end tests

This repository includes a dependency-free integration test runner for the three
HTTP surfaces served by `qarnyx-server`:

- OpenAI-compatible Chat Completions
- Anthropic-compatible Messages
- Native Qarnyx generation

The runner talks to a running Qarnyx server. It does **not** start, stop, or
restart the server, and it does not call the real OpenAI or Anthropic services.
Inference admission is bounded: once all configured request slots are in use,
new requests fail immediately with HTTP 429 rather than waiting in an
unbounded server-side queue.

## 1. Start Qarnyx yourself

In terminal 1, start the server with your GGUF model:

```bash
cargo run --release -p qarnyx-api --bin qarnyx-server -- \
  --model /path/to/your/model.gguf \
  --workers 10
```

Keep that process running. The default address is
`http://127.0.0.1:8000`.

## 2. Build and run the complete test suite

In terminal 2, from the repository root:

```bash
bash scripts/api-test.sh
```

The wrapper first runs:

```bash
cargo build --release -p qarnyx-api
```

If the Cargo build fails, the command stops and no API tests run. After a
successful build, the test runner calls `GET /health`. If the server is down,
unreachable, or does not report `{"status":"ok"}`, the runner prints a
`[SKIP]` message and exits without executing any test. It never tries to
launch, stop, or restart the server.

If the current API binary is already built and you only want to run the
integration suite:

```bash
bash scripts/api-test.sh --skip-build
```

All arguments after `--skip-build` are passed through to the Python runner.

The runner uses only Python's standard library; no `pip install` is needed.
Requests run sequentially because the server defaults to one concurrent
inference slot.

## 3. Select a target or tune limits

```bash
# Another local port or remote test instance
bash scripts/api-test.sh --base-url http://127.0.0.1:9000

# Request the loaded GGUF model ID instead of the qarnyx alias
bash scripts/api-test.sh --model "Huihui Qwen3.5 2B Abliterated"

# Run only one protocol surface
bash scripts/api-test.sh --suite openai
bash scripts/api-test.sh --suite anthropic
bash scripts/api-test.sh --suite native

# Adjust generation limits and per-request timeout
bash scripts/api-test.sh --max-tokens 64 --tool-max-tokens 160 --timeout 900
```

Options:

| Option | Default | Purpose |
|---|---:|---|
| `--base-url` | `http://127.0.0.1:8000` | Qarnyx server URL |
| `--model` | `qarnyx` | Request model name/alias |
| `--suite` | `all` | `all`, `openai`, `anthropic`, or `native` |
| `--max-tokens` | `48` | Output limit for ordinary generation tests |
| `--tool-max-tokens` | `128` | Output limit for tool-call tests |
| `--timeout` | `600` | Timeout per HTTP request, in seconds |

## 4. Test coverage

The `all` suite runs these checks serially. Check `GET /health` and
`GET /ready` separately for current/max request-slot telemetry:

| Surface | Coverage |
|---|---|
| Shared | Health probe, model discovery |
| OpenAI | Basic chat, system role, multi-turn history, JSON completion and usage, text SSE, forced tool call, streamed tool call, tool-result round trip, stop strings, JSON-object response mode, invalid request rejection, unknown model rejection |
| Anthropic | Basic Messages, system prompt, text content blocks, multi-turn history, message SSE lifecycle, forced `tool_use`, streamed `tool_use`, `tool_result` round trip, `stop_sequences`, `metadata.user_id`, unsupported parameter rejection, unknown model rejection |
| Native Qarnyx | Raw-prompt generation, generation controls, token IDs and usage, native SSE token/text/completed/done events, empty prompt rejection, invalid generation config rejection, unknown model rejection |

A successful tool round trip performs a simulated client-side tool execution:
the runner supplies the deterministic fixture result `Sunny, 31 C` to the
server in a follow-up request. It does not query a weather service. This checks
the protocol contract and conversation history; Qarnyx itself does not execute
external tools.

The native `/v1/generate` API is a raw-prompt generation interface and does
not expose function-tool calling. Tool-call coverage therefore belongs to the
OpenAI and Anthropic adapters.


## Request limits, cancellation, and protocol compatibility

The server acquires an inference permit before prompt tokenization. If all
`--max-concurrent-requests` slots are occupied, the OpenAI and native adapters
return HTTP 429 with a rate-limit error; Anthropic returns its rate-limit error
shape with the same HTTP status. The server does not accumulate waiting
inference tasks. `/health` and `/ready` report `active_requests` and
`max_concurrent_requests`.

SSE workers observe a disconnected receiver while generating and cancel at the
next token boundary. Buffered tool/stop/JSON responses also check whether the
client receiver has closed. Non-streaming handlers use a drop guard to request
cancellation if their handler future is dropped. Ctrl+C and SIGTERM cancel all
active request tokens before the server drains HTTP connections. Cancellation
is cooperative: the current model/token evaluation is allowed to finish.

OpenAI `stop` accepts one string or up to four strings. Anthropic
`stop_sequences` accepts up to sixteen strings. Stop text is excluded from
returned content, and Anthropic reports the matching `stop_sequence` with
`stop_reason="stop_sequence"`. Stop matching is text-based and cancels at the
next token boundary; it does not interrupt the middle of a token evaluation.

OpenAI `response_format.type="json_object"` injects a JSON-only instruction,
buffers streamed output, and validates that the completed response is a valid
top-level JSON object before returning it. This is prompt-guided validation,
not grammar-constrained decoding; a model that fails to follow the instruction
returns a server-side model-output error. Anthropic `metadata.user_id` is
accepted and validated but is not persisted or used for authentication.

The native `/v1/generate` endpoint remains a raw-prompt interface and does not
inherit OpenAI/Anthropic stop fields or response-format controls.

## 5. Reading results

Each case prints `[PASS]` or `[FAIL]` and the runner continues after an
individual failure so that one incompatible behavior does not hide the rest.
The final summary reports totals. The process exits with:

- `0` when all selected tests pass, or when the server is unavailable and the
  suite is skipped before testing.
- `1` when one or more tests fail.

Tool calling is model-dependent. The forced-tool cases verify that the model
actually returns a valid call for the selected `get_weather` tool and Jakarta
arguments. A model that ignores the forced choice will be reported as a failed
integration test even if ordinary text generation works. Increase
`--tool-max-tokens` if the model needs more room to emit the call.

## 6. Safety and scope

Run the runner against a development or test instance. It intentionally sends
invalid model names and invalid request payloads to verify error handling. The
requests are sequential and use short generation limits by default, but they
still consume inference time and CPU.

This verifies Qarnyx's OpenAI-compatible, Anthropic-compatible, and native
HTTP implementations. It is not a certification of full vendor API parity,
and it does not test authentication, multimodal inputs, unsupported vendor
features, or real external tool execution.
