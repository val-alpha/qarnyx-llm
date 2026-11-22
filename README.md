# Qarnyx

**Quantized intelligence. Engineered in Rust.**

Qarnyx is an experimental, Rust-native, CPU-first inference runtime for
[GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) language
models. It is being designed from scratch around quantized CPU kernels,
explicit memory ownership, and predictable resource usage.

> **Project status: early development.** The GGUF v3 reader validates tensor
> descriptors and payload ranges without loading weights during inspection.
> Scalar decoders support F32, F16, Q4_0, Q8_0, Q4_K, Q5_K, and Q6_K. The CPU
> backend has runtime-dispatched AVX2/FMA matvec and matmul kernels, a persistent
> Rayon pool, batched independent Qwen3.5 projections, reusable per-session
> inference scratch, parallel recurrent heads, and an opt-in direct-block Q4_K
> AVX2/FMA path with a scalar fallback. Experimental Qwen3.5 2B generation is
> implemented; further quantized kernels and end-to-end performance validation
> remain in progress.

## Design goals

- **Rust-native runtime:** no C/C++ inference engine as a runtime dependency.
- **CPU first:** scalar reference kernels first, followed by runtime-dispatched
  SIMD implementations where the host CPU supports them.
- **GGUF support:** parse v3 metadata and tensor descriptors directly from a
  seekable reader, with bounded allocations and checked offsets.
- **Quantized execution:** operate on packed quantized weights without
  materializing the entire model as F32.
- **Memory efficiency:** stream model metadata, later add read-only model
  mapping, bounded KV cache, and reusable activation buffers.
- **Correctness before speed:** compare optimized kernels against scalar
  references and validate model outputs against independent implementations.

Rust does not automatically make a program leak-free or faster than C/C++.
Qarnyx will use safe Rust by default, isolate required unsafe code, and
measure performance on the same hardware as its benchmarks.

## Workspace

| Crate | Responsibility |
|---|---|
| qarnyx-core | Tensor views, data types, backend contracts, shared errors |
| qarnyx-gguf | GGUF v3 header, metadata, tensor directory, structural validation |
| qarnyx-quant | Quantization block layouts and reference decoders |
| qarnyx-cpu | Scalar/SIMD CPU kernels, feature detection, CPU execution |
| qarnyx-tokenizer | Tokenizer metadata and tokenization |
| qarnyx-models | Model definitions, including the experimental Qwen3.5 path |
| qarnyx-runtime | Prefill, decode, KV cache, sampling, generation |
| qarnyx-cli | Command-line interface |
| qarnyx-api | OpenAI-compatible HTTP serving adapter |

See [docs/architecture.md](docs/architecture.md) for the architecture,
backend contract, memory strategy, and implementation roadmap.
See [Kimi K3 in C research and CPU optimization notes](docs/research/kimi-k3-in-c-lessons-and-m6-plan.md) for the external implementation study, current benchmark record, and proposed follow-up experiments.

## Current capabilities

Inspect a GGUF v3 file without loading its tensor payloads:

```bash
cargo run --release -p qarnyx-cli -- inspect /path/to/model.gguf
```

Print every tensor descriptor:

```bash
cargo run --release -p qarnyx-cli -- inspect /path/to/model.gguf --list-tensors
```

The parser handles GGUF metadata scalar types and nested arrays, validates
UTF-8 strings, metadata keys, unique tensor names, tensor dimensions, alignment,
directory padding, and complete tensor payload byte ranges. Payload lengths
are calculated by the qarnyx-quant type registry, including per-row block
alignment, and overlapping tensor ranges are rejected. Supported layouts and
scalar F32 output decoders currently include F32, F16, Q4_0, Q8_0, Q4_K,
Q5_K, and Q6_K. Unknown GGML types are rejected rather than assigned guessed
sizes. The reader remains little-endian only and does not read payload bytes
during inspection.

The CPU backend executes matrix-vector and matrix-matrix operations with all
currently registered storage types. On x86_64 hosts with AVX2+FMA, the default
`auto` backend uses direct-block F32-input kernels for Q4_K, Q5_K, and Q6_K,
avoiding F32 weight-row materialization. Other hosts retain the runtime-SIMD
dequantize-then-dot path (or its portable fallback). The previous dequantized
path remains available through the `CpuBackend::new_dequant()` API for A/B
benchmarks.

The explicit `fused-q4k` mode uses a Q8_K-quantized activation vector and an
integer AVX2 Q4_K dot for wide inputs. The `llama-q4k` mode implements the x86
AVX2/FMA Q4_K x Q8_K per-row dot strategy used by `ggml_vec_dot_q4_K_q8_K`
in llama.cpp, including its SIMD minimum correction. It does not call llama.cpp
or use its repacked/interleaved matrix format. The explicit `fused-kq-fma`
mode selects the direct-block F32-input AVX2/FMA K-quant path for comparison;
this is also the path selected by `auto` when AVX2+FMA are available. The
explicit `fused-qk` mode retains the integer/AVX2 direct-block alternatives.
Unsupported CPU features use the corresponding portable fallback.

Large matvec/matmul output rows and independent Qwen3.5 projection batches use
the same persistent Rayon pool. Qwen3.5's Q/K/V, recurrent projections, and
FFN gate/up projections can be submitted as batches over a shared input.
Its recurrent and attention workspaces are retained in the per-session cache
and reused across tokens/layers rather than repeatedly allocating the same
intermediate buffers.

Run the CPU microbenchmarks with `cargo bench -p qarnyx-cpu --bench kernels`.
The harness compares scalar/reference and runtime-dispatched matvec paths,
including opt-in fused Q4_K/Q5_K/Q6_K comparisons at selected worker counts.
Cargo's bench profile is optimized by default; do not pass `--release` to
`cargo bench`.

To compare end-to-end generation using the same model and prompt, select a
kernel mode explicitly:

```bash
./target/release/qarnyx generate model.gguf "hi there!" --max-new-tokens 32 --cpu-kernel scalar
./target/release/qarnyx generate model.gguf "hi there!" --max-new-tokens 32 --cpu-kernel auto
./target/release/qarnyx generate model.gguf "hi there!" --max-new-tokens 32 --cpu-kernel fused-q4k
./target/release/qarnyx generate model.gguf "hi there!" --max-new-tokens 32 --cpu-kernel llama-q4k
./target/release/qarnyx generate model.gguf "hi there!" --max-new-tokens 32 --cpu-kernel fused-kq-fma
./target/release/qarnyx generate model.gguf "hi there!" --max-new-tokens 32 --cpu-kernel fused-qk
```

For Qwen3.5, add `--thinking` to enable model reasoning output; without the
flag the CLI uses the non-thinking prefix.

The `auto` mode now prefers direct-block F32-input Q4_K/Q5_K/Q6_K AVX2/FMA
kernels when the runtime detects both AVX2 and FMA. Use `CpuBackend::new_dequant()`
in Rust benchmarks to compare against the previous dequantize-plus-dot baseline.
`fused-q4k`, `llama-q4k`, and `fused-qk` remain explicit alternatives for
same-machine comparisons. `scalar` remains the non-SIMD correctness reference.

Generation is capped at 4096 combined prompt and requested output tokens by
default. Use `--max-context-tokens N` to set a different per-request limit.
The retained per-session KV/recurrent cache and Qwen3.5 workspace also have a
512 MiB default budget; use `--max-cache-mib N` to adjust it. Both limits are
checked before cache allocation, and cache/workspace capacity is reserved up
front. The cache budget excludes model weights and transient logits/outputs;
it is not a process-wide RSS limit. Neither default changes the model's native
context window.

The CLI also has an 8192 MiB `--max-memory-mib` admission budget. Before
reading a model file, it reserves the configured `--max-cache-mib` allowance
and verifies that the file fits in the remaining bytes. The read itself is
bounded to that allowance. This is an accounted model-file-plus-session-cache
budget, not a process-wide RSS guarantee: parser metadata, tokenizer data,
transient logits/outputs, allocator overhead, and other process memory are
excluded.

For Qwen3.5 text generation, the CLI encodes a single-turn chat prompt using
the model's ChatML control tokens. Non-thinking is the default; pass
`--thinking` to leave the reasoning block open and include generated reasoning
in the output. The CLI displays the opening `<think>` marker for that mode
and omits the end-of-turn control token.

Generation can be cancelled with Ctrl+C. Cancellation is cooperative: the
currently executing token evaluation finishes, then inference stops and its
request-local cache is released. The CLI does not print partial generated text.


## HTTP APIs (experimental)

Build and start the server with one local GGUF model:

```bash
cargo run --release -p qarnyx-api --bin qarnyx-server -- \
  --model /path/to/model.gguf --workers 10
```

The server binds to `127.0.0.1:8000` by default. It exposes three protocol
adapters over the same Qarnyx runtime generation path:

| Protocol | Endpoint | Streaming |
|---|---|---|
| OpenAI Chat Completions | `POST /v1/chat/completions` | OpenAI-style SSE |
| Anthropic Messages | `POST /v1/messages` | Anthropic Messages SSE events |
| Native Qarnyx | `POST /v1/generate` | Native token/text/completion events |

The shared discovery endpoints are `GET /health` and `GET /v1/models`.

OpenAI-compatible request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qarnyx",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_completion_tokens": 64,
    "temperature": 0
  }'
```

Anthropic Messages request:

```bash
curl http://127.0.0.1:8000/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qarnyx",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0
  }'
```

Native Qarnyx request. The prompt is passed as-is; no vendor chat template is
inserted by this endpoint:

```bash
curl http://127.0.0.1:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qarnyx",
    "prompt": "Once upon a time",
    "config": {
      "max_new_tokens": 64,
      "temperature": 0,
      "top_k": 40,
      "top_p": 0.95,
      "seed": 7
    }
  }'
```

Set `"stream": true` and use `curl -N` to receive streaming events from each
adapter. The native endpoint accepts `max_new_tokens`, `max_context_tokens`,
`max_cache_mib`, `temperature`, `top_k`, `top_p`, and `seed` under its
`config` object. Anthropic text blocks and client-executed tool-use/tool-result
round trips are supported. The server does not execute client tools.
Tool-call output is normalized from the Qarnyx JSON envelope, Qwen-style
`<tool_call>` blocks, single-call JSON, and OpenAI-shaped function-call
objects. Unknown tools, malformed arguments, and failed forced-tool responses
are rejected; a model's compliance with `tool_choice` remains model-dependent.
The client executes returned tools and sends their results in a later request.

Qwen3.5 messages use its ChatML format. LLaMA currently uses a generic
role-labelled prompt because model-specific LLaMA chat templates are not
implemented. Multimodal inputs, stop sequences, OpenAI logprobs, broad vendor
parameter parity, and built-in authentication remain outstanding.

The server defaults to loopback and has no built-in authentication. Do not
expose it to an untrusted network without an authenticated gateway. Concurrent
inference is limited by a bounded semaphore (one request by default); increase
`--max-concurrent-requests` only after measuring available memory.

## Development

A recent stable Rust toolchain with Cargo is required.

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
cargo build --release
```

For host-specific release builds, benchmark a build using
`RUSTFLAGS="-C target-cpu=native"`. Such binaries are specific to the CPU
features available on the build machine and should not be distributed as
portable binaries without an appropriate dispatch/build strategy.

## Implementation roadmap

1. **M0 - Hardware and benchmark baseline:** capture host capabilities and
   establish repeatable kernel and end-to-end benchmarks.
2. **M1 - GGUF reader:** implemented for GGUF v3 metadata and tensor directory.
   Streaming file parsing and malformed-input tests are included.
3. **M2 - Quantization reference:** initial supported set implemented for
   F32/F16, Q4_0/Q8_0, Q4_K/Q5_K/Q6_K, with exact payload-range and overlap
   validation.
4. **M3 - Scalar tensor engine:** initial scalar kernels implemented for
   matvec, matmul, RMSNorm, split-half RoPE, softmax, SiLU-times-up, and
   residual addition. Shape/numerical tests and a dependency-free microbenchmark
   harness are included.
5. **M4 - Model forward pass:** implement and validate the experimental
   Qwen3.5 hybrid path against an independent reference; retain the legacy
   LLaMA path separately.
6. **M5 - Tokenization and generation:** add prefill, incremental decode,
   bounded KV cache, stopping rules, and sampling.
7. **M6 - CPU optimization (in progress):** AVX2/FMA matvec/matmul, a persistent
   Rayon pool, parallel Qwen3.5 recurrent heads, shared-pool multi-projection
   scheduling, per-session activation scratch reuse, and opt-in direct-block
   Q4_K AVX2/FMA are implemented with scalar fallbacks and correctness tests.
   Opt-in direct-block Q5_K/Q6_K AVX2 kernels with runtime FMA accumulation
   are also implemented and covered by differential tests. Three-round VPS
   microbenchmarks show median gains for Q5_K/Q6_K, varying by worker count;
   representative-model validation, AVX-512, and NEON remain outstanding.
8. **M7 - Operational hardening:** combined prompt/output context cap
   (default 4096), per-session cache/workspace memory budget (default 512 MiB),
   cooperative token-boundary cancellation (including CLI Ctrl+C), and the
   M7-D model-load admission budget (default 8192 MiB, including the configured
   session-cache reservation) are implemented. Session isolation, stress tests,
   and reproducible benchmarks remain planned.

9. **M8 - API and serving layer:** runtime token streaming, OpenAI Chat
   Completions, Anthropic Messages, and native Qarnyx generation endpoints are
   implemented over a shared protocol-neutral GenerationRequest/GenerationOutput
   contract. Broader vendor parameter parity, multimodal inputs, and built-in
   authentication remain outstanding.

## Non-goals for the first milestone

- GPU execution
- Multi-model serving
- Distributed inference
- Broad support for every GGUF architecture and quantization type

## License

MIT. See LICENSE.
