# Qarnyx API end-to-end tests

This repository includes a dependency-free integration test runner for the three
HTTP surfaces served by `qarnyx-server`:

- OpenAI-compatible Chat Completions
- Anthropic-compatible Messages
- Native Qarnyx generation

The runner talks to a running Qarnyx server. It does **not** start, stop, or
restart the server, and it does not call the real OpenAI or Anthropic services.

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

The `all` suite runs these checks serially:

| Surface | Coverage |
|---|---|
| Shared | Health probe, model discovery |
| OpenAI | Basic chat, system role, multi-turn history, JSON completion and usage, text SSE, forced tool call, streamed tool call, tool-result round trip, invalid request rejection, unknown model rejection |
| Anthropic | Basic Messages, system prompt, text content blocks, multi-turn history, message SSE lifecycle, forced `tool_use`, streamed `tool_use`, `tool_result` round trip, unsupported parameter rejection, unknown model rejection |
| Native Qarnyx | Raw-prompt generation, generation controls, token IDs and usage, native SSE token/text/completed/done events, empty prompt rejection, invalid generation config rejection, unknown model rejection |

A successful tool round trip performs a simulated client-side tool execution:
the runner supplies the deterministic fixture result `Sunny, 31 C` to the
server in a follow-up request. It does not query a weather service. This checks
the protocol contract and conversation history; Qarnyx itself does not execute
external tools.

The native `/v1/generate` API is a raw-prompt generation interface and does
not expose function-tool calling. Tool-call coverage therefore belongs to the
OpenAI and Anthropic adapters.

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
