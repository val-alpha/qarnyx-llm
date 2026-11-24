# Qarnyx Architecture

## Project constraints

Qarnyx is a CPU-first inference runtime written in Rust. The initial
implementation is intentionally narrow: one model family, a small set of
well-tested GGUF tensor types, one CPU backend, and one inference session.

The runtime must not depend on llama.cpp, Candle, tch, ONNX Runtime, or another
inference engine. Small system and utility crates are acceptable when they do
not provide the inference implementation itself.

The first deployment profile is a Linux VPS with a limited CPU and memory
budget. The engine must measure actual hardware capabilities rather than
assuming that a VPS allocation represents dedicated physical cores.

## Workspace responsibilities

| Crate | Responsibility | Must not own |
|---|---|---|
| qarnyx-core | Tensor views, data types, backend contracts, common errors | File parsing or model architecture |
| qarnyx-gguf | GGUF v3 header, metadata, tensor directory, validated tensor views | Tensor math or generation |
| qarnyx-quant | Quantization layouts and scalar reference decoders | Scheduling or model loading |
| qarnyx-cpu | Scalar/SIMD kernels, CPU feature detection, persistent workers | Tokenization or GGUF parsing |
| qarnyx-tokenizer | Tokenizer metadata and token-to-text conversion | Tensor operations |
| qarnyx-models | Model configuration and transformer forward graph | Hardware-specific intrinsics |
| qarnyx-runtime | Model lifecycle, prefill/decode, KV cache, sampling | Low-level SIMD kernels |
| qarnyx-cli | User-facing command-line interface | Inference implementation |

Dependencies should point inward toward core contracts. Avoid cyclic crate
dependencies and avoid putting model-specific logic in the GGUF parser.

## GGUF reader boundary

The current qarnyx-gguf reader:

- Supports the GGUF v3 little-endian representation.
- Reads from a seekable source and determines file size without reading the
  model's tensor payload into memory.
- Parses all defined metadata scalar types and recursively nested arrays.
- Applies configurable limits to metadata counts, tensor counts, metadata
  bytes, string lengths, array lengths/depth, and alignment.
- Rejects invalid magic/version, invalid UTF-8, malformed booleans, duplicate
  metadata keys/tensor names/tensor offsets, invalid dimensions, overflowing
  shapes, invalid alignment, non-zero alignment padding, and tensor data start
  offsets outside the file.
- Uses the qarnyx-quant registry to calculate exact encoded payload lengths
  for supported GGML types and validates complete tensor byte ranges.
- Rejects unknown tensor types, invalid quantized row dimensions, payload
  ranges extending beyond the file, and overlapping tensor ranges.

The supported registry currently includes F32, F16, Q4_0, Q8_0, Q4_K, Q5_K,
and Q6_K. Tensor payload bytes remain unread during inspection; only their
shape-derived extents are validated.

The reader currently rejects big-endian files. This limitation is explicit;
the initial deployment target is little-endian x86_64/aarch64.

The CLI can inspect a file using:

```bash
cargo run --release -p qarnyx-cli -- inspect model.gguf
cargo run --release -p qarnyx-cli -- inspect model.gguf --list-tensors
```

## Data path

1. Open a GGUF file and validate its magic, version, metadata counts, tensor
   dimensions, alignment, offsets, and file bounds.
2. The current CLI reads the complete GGUF payload into a bounded owned byte
   buffer. Matrix weights remain immutable borrowed tensor views into those
   bytes; memory-mapped loading is a future optimization.
3. Resolve tensor metadata by name and expose checked, non-owning tensor views.
4. Load model configuration and map tensor names to a model implementation.
5. Allocate a bounded request-local KV/recurrent cache and reusable activation
   scratch workspace.
6. Execute prefill, then incremental decode through the selected CPU backend.
7. Sample the next token and decode it with the tokenizer.

The current CLI uses an owned model-file buffer rather than a memory map.
Track process RSS during benchmarks. If memory-mapped loading is added later,
report process RSS and system page cache separately because mapped file size
and resident memory are different quantities.

## Core tensor contract

A tensor view is non-owning and immutable. It describes a byte range, a shape,
and a data type. It must not silently allocate or dequantize the entire tensor.

All file-derived views must be created only after validating:
- the tensor offset and byte length are within the mapped file;
- all arithmetic uses checked operations;
- the tensor data offset obeys GGUF alignment;
- the shape is valid for the selected quantization block format;
- the byte length is sufficient for the number of encoded blocks.

Quantized tensors are block encoded. Their required byte length must be
computed from the format's block size and encoded block size, not from a
naive bits-per-element multiplication.

## Backend contract

Models call backend operations and do not select AVX intrinsics themselves.
The CPU backend selects kernels using runtime CPU feature detection. On
x86_64 hosts with AVX2, matrix-vector dot products and matrix-multiplication
output tiles use AVX2; FMA is used when available. Scalar implementations
remain available as the correctness reference and portable fallback.

The CPU backend implements:
- matrix-vector multiplication for decode;
- matrix multiplication for prompt prefill;
- RMSNorm;
- split-half LLaMA-style RoPE;
- numerically stable softmax;
- SiLU multiplied by the up projection;
- in-place residual addition.

Matvec weights use [out_features, in_features] row-major logical shape and
produce out_features values. Matmul weights are [m, k], the row-major input is
[k, n], and the row-major output is [m, n]. The physical quantized block layout
remains specific to the tensor type. Matrix kernels validate exact encoded
storage lengths and do not materialize a full F32 matrix. On x86_64 with AVX2
and FMA, Q4_K/Q5_K/Q6_K matvecs quantize their F32 activation vector into
reusable Q8_K blocks and execute integer SIMD dot products against packed
weights. Q4_K/Q5_K minimum terms and the Q6_K -32 offset use Q8_K partial sums.
Batched projections with the same input share one Q8_K buffer. CPUs without
both features keep the runtime-SIMD dequantize-plus-dot fallback. The selectable
CPU surface stays intentionally small: `auto` is the default dispatch mode
and `fused-kq-fma` selects the same optimized path for controlled comparisons.

General matvec and matmul fallbacks reuse a per-thread F32 scratch allocation.
Large operations are partitioned by encoded weight bytes into several tasks
per worker, while small operations avoid Rayon scheduling overhead. The Q8_K
activation scratch is also retained per thread and reused across matvec calls;
matvec_batch prepares it once before scheduling projections that share input.
F32 matmul decodes up to four weight rows together and uses an AVX2/FMA
register tile to reuse each input vector across those rows when the tile and
shape fit. This is a F32 GEMM micro-kernel after row decoding, not a packed
Q4_Kx8 GEMM.
RMSNorm requires a positive finite epsilon. RoPE pairs the first and second
halves of one head vector, matching the split-half convention used by
LLaMA-style rotary embeddings. Softmax rejects empty or non-finite inputs.

The CPU runtime intentionally keeps only two selectable kernel modes: `auto`
and `fused-kq-fma`. On x86_64 with AVX2+FMA, both use the Q8_K activation
quantization and integer-dot Q4_K/Q5_K/Q6_K path. `auto` is the
production/default mode and retains the portable fallback when the required
ISA is unavailable; `fused-kq-fma` exposes the optimized path explicitly for
controlled benchmarking. Earlier F32-input direct-block kernels are no longer
the active K-quant matvec path.
RMSNorm, RoPE, softmax, and residual addition currently use scalar
implementations.

The `workers` setting controls a persistent Rayon thread pool owned by the CPU
backend. The pool is created once during backend construction and reused for
large matvec/matmul operations. `matvec_batch` schedules independent
projections that share an input on that same pool; it does not create a second
pool or per-projection OS threads. Each output remains disjoint, and every
projection completes before the model continues. Small matrices remain on the
single-threaded path to avoid scheduling overhead. Benchmark worker counts
1, 2, 4, 6, 8, and 10 on the target VPS because scheduler contention and memory
bandwidth may make fewer workers faster. Run
`cargo bench -p qarnyx-cpu --bench kernels` for focused comparisons of the
retained `auto` and `fused-kq-fma` paths.

The scalar implementation is the correctness oracle. SIMD implementations
must be compared against it with documented tolerances and must pass
correctness tests before dispatch is enabled.

## CPU execution strategy

Keep portable scalar/dequantization code internal as a correctness reference
and fallback. The public CLI exposes only two selectable modes:
- `auto`: production/default runtime dispatch; uses Q8_K activations and
  Q4_K/Q5_K/Q6_K AVX2 integer-dot kernels on supported x86_64 CPUs, with
  portable fallback otherwise;
- `fused-kq-fma`: explicit selection of that same Q8_K/integer-dot path for
  controlled comparisons; it is not a separate algorithm from the current
  `auto` fast path;
- aarch64: portable fallback until a NEON implementation is separately
  validated;
- other configurations: portable fallback.

Do not add selectable kernel variants without a reproducible target-machine
throughput gain and correctness evidence. Internal scalar reference code is
not a third CLI kernel mode.

Do not compile the default release build with target-cpu=native. That setting
can produce a binary that fails on a different CPU. Use it only for
host-specific builds and benchmark it separately.

The CPU backend uses a persistent Rayon worker pool rather than creating OS
threads for each tensor operation. Large matvec/matmul work is partitioned into
independent output-row chunks. Independent Qwen3.5 projections can also be
submitted as one batch to the same pool; nested work is kept within that shared
scheduler rather than creating additional pools. Benchmark 1, 2, 4, 6, 8, and
10 workers. VPS CPU scheduling and memory bandwidth may make fewer workers
faster than all available vCPUs.

## Memory policy

- Model weights: immutable GGUF bytes in an owned `Vec<u8>` in the current
  CLI; no full-model F32 copy. Memory-mapped loading remains future work.
- KV cache: allocated per generation request with an explicit context limit.
- Generation requests: the runtime enforces a configurable maximum combined
  prompt and generated length (max_context_tokens, default 4096) before it
  allocates the per-request KV/recurrent cache. The CLI exposes this as
  --max-context-tokens; raise it deliberately when a longer context is needed.
- Per-generation cache budget: the runtime estimates retained KV/recurrent
  state and reusable Qwen3.5 workspace capacity before allocation. The default
  is 512 MiB, configurable with --max-cache-mib. Qwen3.5 and LLaMA reserve
  their maximum sequence capacity up front to avoid cache growth reallocations.
  Each concurrent generation call has its own budget; this is not a global
  process-wide RSS or aggregate concurrency limit. The estimate excludes
  immutable model weights, allocator metadata, and transient output/logit
  buffers.
- Model-load admission budget: the CLI defaults --max-memory-mib to 8192 MiB.
  Before reading a GGUF file, it reserves the configured --max-cache-mib
  allowance and checks that the model file fits in the remainder. The read
  itself is capped at that remainder to guard against a file changing after
  the metadata check. This accounts for model-file bytes plus one maximum
  generation-cache reservation. Additional concurrent cache reservations,
  parser/tokenizer allocations, transient logits and outputs, allocator
  overhead, and unrelated process memory are excluded. It is an admission-
  control envelope, not an RSS guarantee.
- Cooperative cancellation: callers can share a CancellationToken with a
  generation task. Cancellation is checked between complete token evaluations;
  the in-flight token finishes, then request-local state is dropped. The CLI
  maps Ctrl+C to this signal and emits no partial generated text.
- The Qwen3.5 CLI path applies a single-turn text ChatML prompt with the
  assistant generation prefix from the model's chat template. Non-thinking is
  the default; --thinking leaves the reasoning block open and displays its
  opening <think> marker with generated reasoning. Control-token spellings
  found in the vocabulary are encoded as their atomic token IDs instead of
  ordinary byte-level text.
- Known end-of-turn tokens (<|im_end|>, <|fim_suffix|>, and <|endoftext|>) plus
  the configured EOS ID stop generation and are excluded from returned output.
- Activations: reusable scratch buffers sized from model configuration.
- Ownership: share immutable model data and the CPU backend; keep one
  generator/RNG state and one mutable cache/workspace per concurrent stream.
- Isolation: never share a mutable LLaMA KV cache or Qwen3.5 recurrent/attention
  cache between independent streams.
- Admission: per-generation limits are not a substitute for a process-wide
  in-flight request limit when a serving layer is introduced.
- Synchronization: keep locks out of hot tensor loops; avoid a global
  Mutex around model weights.
- Limits: validate context length, batch size, and allocation sizes before
  reserving memory.

FP16 KV cache is a target option, not an assumed universal implementation.
It must be tested for numerical stability and supported cache layouts.

## Initial model scope

The current experimental target is Qwen3.5 2B, a hybrid architecture with
Gated DeltaNet recurrent layers and full-attention layers. Keep architecture-
specific recurrence, RoPE, and tensor mapping inside qarnyx-models. The legacy
LLaMA path remains separate and is not the target for Qwen3.5 validation.
Validate tensor layouts, numerical outputs, tokenizer behavior, and recurrent
state against an independent reference before declaring model compatibility.

## Implementation roadmap

### M0 - Hardware and benchmark baseline
- Capture CPU model, architecture, feature flags, available memory, and limits.
- Record compiler version and build flags.
- Create repeatable kernel and end-to-end benchmark fixtures.

### M1 - GGUF reader (implemented)
- Parse GGUF v3 little-endian header and metadata value types.
- Parse tensor descriptors and resolve the aligned tensor data section.
- Validate counts, UTF-8 strings, dimensions, alignment, directory padding,
  duplicate entries, and tensor data start offsets.
- Parse from a seekable reader without loading tensor payloads into memory.
- Add malformed-file and resource-limit tests.

### M2 - Quantization reference (initial supported set implemented)
- Register F32/F16, Q4_0/Q8_0, and K-quant Q4_K/Q5_K/Q6_K block layouts.
- Add safe scalar reference decoders that write into caller-owned F32 buffers.
- Calculate exact per-tensor encoded lengths using per-row block alignment.
- Validate complete tensor byte ranges and reject overlaps without reading
  tensor payload bytes.
- Add type, size, row-alignment, decoder, truncation, and overlap tests.
- Expand the registry only when a new GGML layout has a verified reference.

### M3 - Scalar tensor engine (initial implementation)
- Implement scalar matvec and matmul with row-at-a-time dequantization for
  registered F32/F16 and Q4_0/Q8_0/Q4_K/Q5_K/Q6_K weight types.
- Implement RMSNorm, split-half LLaMA-style RoPE, stable softmax, SiLU-times-up,
  and residual addition.
- Test shape errors, invalid scalar parameters, and numerical behavior.
- Add dependency-free scalar matvec/matmul microbenchmarks before SIMD work.
- Keep kernels single-threaded and use them as the correctness reference for
  later SIMD implementations.

### M4 - Model forward pass
- Implement the experimental Qwen3.5 hybrid forward path.
- Keep the legacy LLaMA implementation as a separate compatibility path.
- Compare logits and token IDs against an independent reference runtime.

### M5 - Tokenization and generation
- Read tokenizer metadata and special tokens.
- Implement prompt prefill and incremental decode.
- Add KV cache, stopping rules, and sampling.
- Verify generated token IDs, not only rendered text.

### M6 - CPU optimization (implemented on main)
Implemented:
- Runtime AVX2/FMA dispatch for matvec dot products and matmul output tiles,
  with scalar fallbacks and differential tests.
- Persistent Rayon scheduling across independent matvec/matmul output rows.
- Parallel Qwen3.5 recurrent-head state updates and head outputs, with the
  recurrence and per-head reduction order preserved.
- A `matvec_batch` backend contract with a sequential default and a CPU
  implementation that schedules independent Qwen3.5 Q/K/V, recurrent, and
  FFN gate/up projections on the same persistent Rayon pool.
- Per-session Qwen3.5 scratch/workspace reuse across layers and decode steps,
  including ping-pong hidden buffers, normalization/projection buffers,
  attention scores, recurrent output, and FFN intermediates.
- Opt-in direct-block Q4_K matvec with AVX2/FMA on supported x86_64 CPUs and a
  scalar direct-block fallback, covered by scalar-reference differential tests.
- Opt-in direct-block Q5_K/Q6_K matvec with AVX2 and runtime FMA accumulation
  when supported on x86_64 CPUs, plus AVX2-only and scalar direct-block
  fallbacks, covered by multi-block differential tests.

Target-machine evidence:
- On the Qwen3.5 2B Q4_K_M 32-token greedy workload, M6-A reduced the measured
  mean runtime from 25.77 s to 21.48 s (16.6%).
- A later three-run benchmark recorded 17.64 s for M6-A, 16.22 s for the
  auto path, and 12.57 s for fused Q4_K. Fused Q4_K was 28.74% below the
  M6-A baseline and 22.50% below auto.
- Outputs were identical across the compared Q4_K runs.
- Synthetic Q5_K/Q6_K microbenchmarks show fused gains depend on worker count
  and workload; they are not substitutes for representative end-to-end model
  measurements.
- AVX-512 and NEON remain future hardware-specific work.

### M7 - Operational hardening (implemented on main; target validation ongoing)

Implemented safeguards:
- Combined prompt + generated-token context cap via `max_context_tokens`,
  default 4096, checked with overflow-safe arithmetic before cache allocation.
- Per-generation cache/workspace budget via `max_cache_mib`, default 512 MiB,
  checked before allocation. Qwen3.5 recurrent/attention state and reusable
  scratch capacity are included; immutable model weights and transient output
  buffers are excluded.
- Cooperative cancellation through a shared `CancellationToken`. Runtime
  checks occur between complete token evaluations; the in-flight token is
  allowed to finish, then request-local state is dropped. CLI Ctrl+C uses the
  same mechanism and does not print partial generated text.
- Model-load memory admission via `max_memory_mib`, default 8192 MiB. The
  current CLI requires model-file bytes plus one configured maximum cache
  reservation to fit before reading the GGUF payload; the read is bounded to
  the remaining allowance. This is an admission envelope, not a process-wide
  RSS guarantee or a reservation for an arbitrary number of concurrent runs.
- Qwen3.5 opt-in thinking mode via `--thinking`. Non-thinking remains the
  default. Thinking mode uses the model's `<think>` generation prefix; known
  end-of-turn control tokens are recognized and excluded from returned text.
- Qwen3.5 chat-template special control tokens are encoded atomically when
  present in the tokenizer vocabulary.

Session ownership and isolation:
- Model weights and the CPU backend are immutable/shareable across generation
  requests. Each concurrent stream must own its own generator instance because
  the generator contains mutable sampling RNG state.
- Every `generate`/`generate_ids` call allocates a fresh request-local
  LLaMA KV cache or Qwen3.5 recurrent/attention cache plus scratch workspace.
  Mutable cache state must not be shared between independent streams.
- Create one `CancellationToken` per request; clone it only to share that
  request's cancellation signal between its caller and generation task.
- Runtime stress coverage: eight concurrent LLaMA generation sessions each
  issue four sequential requests while sharing one model and CPU backend but
  keeping separate generators. The test checks prompt-specific token IDs and
  decoded output on every request.
- Qwen3.5 cache coverage: two independently allocated caches are checked for
  non-shared recurrent state, convolution state, attention K/V data, and
  scratch buffers; `clear` is checked for recurrent/conv/KV reset.

Known operational limits:
- The CLI still serves one blocking generation per process; no HTTP server,
  request queue, global concurrency limit, or process-wide cache admission
  controller exists. Several simultaneous library requests can each consume
  their own configured cache budget, so aggregate memory can scale with the
  number of in-flight requests.
- The current CLI reads the complete GGUF file into an owned byte `Vec`.
  Memory-mapped loading is not implemented yet; there is no full-model F32
  expansion, but model-file bytes are resident in the process allocation.
- `max_memory_mib` covers the model-file buffer plus one maximum cache
  allowance only. Tokenizer/parser metadata, allocator overhead, transient
  logits/output buffers, other process memory, and additional concurrent cache
  reservations are outside that envelope.
- Cancellation is cooperative at token boundaries, not an asynchronous
  interruption inside a SIMD kernel or model-layer evaluation.
- `--thinking` has no separate reasoning-token budget. It is bounded only by
  the request's `max_new_tokens` / context limits; a long reasoning trace can
  consume most of that output allowance.
- Generation caches are request-local and are not retained for multi-turn
  conversation continuation. A caller must provide the conversation context
  again as part of the next prompt.

Validation observed on CI:
- At commit `5b5c8a4`, GitHub Actions completed `cargo check --workspace`
  and `cargo test --workspace` successfully, including the new concurrency,
  cache-isolation, and cancellation-token tests.
- `cargo fmt --all -- --check` still fails on repository-wide formatting
  differences in existing runtime/tokenizer code. No broad auto-format was
  applied because it would introduce unrelated source churn.

Validation confirmed on the target Ubuntu host (user-reported):
- Fast-forward pull from `4a3f772` to `3063b52` completed cleanly.
- `cargo check --workspace`: passed.
- `cargo test --workspace`: 89 passed, 0 failed (CLI 4, core 3, CPU 26,
  GGUF 14, models 12, quant 14, runtime 9, tokenizer 7; doc-tests also passed).
- `cargo build --release -p qarnyx-cli`: passed.

Remaining target validation:
- Capture a repeatable end-to-end benchmark result using the protocol in
  [docs/benchmarking.md](benchmarking.md). The procedure is documented, but
  benchmark numbers remain host-, model-, and kernel-specific.
- Add process-wide admission/concurrency controls only when the serving layer
  is designed; do not start HTTP serving inside M7.

## M8 - API and serving layer (implemented; protocol parity remains scoped)

M8 is an API adapter/serving layer, not another model or inference layer. OpenAI,
Anthropic, and native Qarnyx requests are translated to one protocol-neutral
request contract before reaching the runtime.

Architecture:

```text
OpenAI request ─┐
Anthropic       ├──> protocol adapters ──> GenerationRequest
Native Qarnyx ─┘                              │
                                              ▼
                                       Qarnyx Runtime
                                              │
                                              ▼
                                       GenerationOutput
                                         ├── JSON response
                                         └── streaming events
```

Milestones:
- M8-A Streaming (complete): the runtime exposes protocol-neutral
  `generate_with_events` and `generate_ids_with_events` APIs for both LLaMA
  and Qwen3.5. Each generated token emits its ID and an incremental text
  delta; a tokenizer-owned stream decoder buffers split UTF-8 sequences.
  Prompt tokens and end-of-turn markers are not emitted as completion events.
  Blocking generation remains backward compatible and bypasses stream decoder
  and callback work. Event callbacks run synchronously on the generation
  thread and can request cooperative cancellation through `CancellationToken`.
- M8-B OpenAI-compatible API (implemented): `qarnyx-api` exposes
  `GET /v1/models` and `POST /v1/chat/completions`, including JSON completion
  responses, OpenAI-style SSE chunks, `data: [DONE]`, token usage, supported
  chat roles, sampling controls, and client-executed function tool calls.
  Tool-call requests and tool-result history are normalized before prompt
  formatting; the server never executes client functions.
- M8-C Anthropic Messages-compatible API (implemented): `POST /v1/messages`
  accepts system instructions, user/assistant text, text content blocks,
  function tools, `auto`/`any`/named-tool choices, assistant `tool_use`
  history, and user `tool_result` blocks. JSON responses use Anthropic message
  and content-block shapes. Streaming uses message/content-block lifecycle
  events and text/input-JSON deltas. Tool execution remains the client's
  responsibility.
- M8-D Native Qarnyx API (implemented): `POST /v1/generate` accepts a raw
  prompt plus a native `config` containing `max_new_tokens`,
  `max_context_tokens`, `max_cache_mib`, `temperature`, `top_k`,
  `top_p`, and `seed`. Raw prompts are not wrapped in vendor chat syntax.
  JSON responses expose generated token IDs and usage. Streaming emits native
  token, text-delta, completion, and done events.
- M8-F Tool-call reliability (implemented; model-backed validation remains):
  tool instructions explicitly describe the forced-choice contract and the
  Qarnyx JSON envelope. Model output normalization accepts that envelope,
  Qwen-style `<tool_call>` JSON blocks, single-call JSON objects, and
  OpenAI-shaped function-call objects. Encoded JSON-string arguments are
  decoded into objects; unknown function names, mismatched forced choices,
  malformed JSON, and non-object arguments are rejected. A model that ignores
  a valid forced-tool request is reported as a server-side model-output error,
  not as an invalid client request. Tool execution remains client-owned.
- M8-E Common API abstraction (implemented): all adapters normalize prompt
  text and `GenerationConfig` into the public, protocol-neutral
  `GenerationRequest` contract in `qarnyx-api::common`. The shared
  `ApiState::prepare_generation` validates limits, tokenizes the prompt, and
  produces the same prepared request consumed by the single runtime dispatch
  path. `GenerationOutput` presents generated-only token IDs and shared usage
  counts to protocol response serializers. HTTP routing, JSON decoding,
  protocol validation, SSE transport, and authentication boundaries remain in
  `qarnyx-api`; vendor-specific protocol code is not added to
  `qarnyx-runtime`.
- M8-G Serving hardening (implemented; live overload/disconnect tests remain):
  the server acquires a process-wide inference permit before prompt tokenization
  and rejects saturation immediately with HTTP 429 and `Retry-After: 1`; it
  does not accumulate an unbounded inference queue. The configured memory
  admission reserves the per-request cache allowance multiplied by
  `max_concurrent_requests` before accepting the model file. `/health` and
  `/ready` expose active and maximum request slots. Each request receives an
  independently cancellable token linked to the server shutdown token; SSE
  receiver closure and dropped non-stream handlers request cancellation, and
  Ctrl+C/SIGTERM cancel active inference at token boundaries. This is an
  admission envelope, not an operating-system RSS limit.
- M8-H Protocol compatibility (implemented; parity remains scoped):
  OpenAI Chat Completions accepts `stop` as a non-empty string or an array of
  1–4 strings, excludes the matched stop text, and reports `finish_reason:
  "stop"`. It also accepts `response_format.type="json_object"`; the adapter
  adds a JSON-only prompt instruction, buffers streamed JSON, and validates
  that the final result is a JSON object. This is prompt-guided output
  validation, not grammar-constrained decoding. Anthropic Messages accepts
  `stop_sequences` (up to 16 sequences under the Qarnyx implementation guard)
  and returns the matched `stop_sequence` with
  `stop_reason="stop_sequence"`; `metadata.user_id` is accepted and validated
  up to 512 characters but is not persisted. Stop detection occurs on decoded
  text at token boundaries.

Implementation limits (not a claim of full vendor API parity):
- The API currently serves one loaded GGUF model per server process and
  supports the LLaMA and Qwen3.5 runtime architectures.
- Multimodal content, OpenAI logprobs, broad vendor-specific parameter parity,
  model-specific LLaMA chat templates, and built-in authentication remain
  outstanding. Native `/v1/generate` intentionally remains a raw-prompt
  endpoint and does not accept vendor stop/response-format fields.
- Anthropic system and message content is text-only. Unsupported content block
  types and unsupported request fields are rejected rather than silently
  ignored.
- The native endpoint is a deliberately small, explicit runtime surface. It
  exposes the currently supported generation configuration, not internal
  model/cache implementation details.
- Tool execution is always performed by the client. Qarnyx formats and returns
  tool calls and accepts the corresponding tool-result history; it does not
  invoke external functions.
- Tool calling remains model-dependent: a forced `tool_choice` cannot guarantee
  that a small or instruction-insensitive model will emit a valid function call.
  Qarnyx validates and rejects invalid output; it does not invent or infer tool
  arguments from ordinary assistant text.

All three adapters share the runtime event source and the same non-blocking
request-admission gate. Saturated inference requests receive HTTP 429 rather
than waiting in a server-side queue. They must not duplicate model forward
passes, decode loops, or model-specific prompt-formatting logic. Model-specific chat
formatting remains below M8: Qwen3.5 ChatML templates, thinking-mode control
tokens, and end-of-turn handling belong to tokenizer/model integration rather
than HTTP protocol adapters.

## Benchmark methodology

The repeatable CLI comparison, two-mode Python runner, phase-profiling output,
CSV fields, warm-up rules, and interpretation limits are maintained in
[docs/benchmarking.md](benchmarking.md). Whole-process timings include model
loading and request setup; Qwen3.5 profiling separately reports prompt prefill,
decode-loop, decode-forward, and per-operator aggregates. Do not infer
end-to-end performance from synthetic kernel tests alone.


## M9 - CPU performance gap closure (in progress)

M9 focuses on measurable Qwen3.5 2B CPU performance, not expanding the kernel
menu. Keep only `auto` and `fused-kq-fma` as selectable modes. The existing
scalar/dequantization code remains internal for fallback and differential
correctness checks.

- **M9-A — Baseline and phase profiling:** separate Qwen3.5 prompt prefill from
  autoregressive decode; report prefill tokens/sec, decode-loop tokens/sec,
  decode-forward tokens/sec, and per-operator totals by phase. Add one
  standard-library benchmark runner that compares only the two retained modes,
  uses warm-up and round-robin trials, checks output hashes, and records median
  and range.
- **M9-B — Quantized GEMV:** Q8_K activation quantization and AVX2 integer-dot
  kernels for Q4_K/Q5_K/Q6_K are implemented. Activation blocks are reused
  across batched projections; partial sums handle Q4_K/Q5_K minimum
  corrections and the Q6_K -32 bias. Scalar-oracle regression tests are added.
  Target-host performance validation remains required before claiming a speedup.
- **M9-C — Fused elementwise operations:** optimize only operators proven
  material by M9-A profiling, such as RMSNorm, RoPE, softmax, activation, or
  recurrent-state updates. Avoid speculative SIMD work on negligible phases.
- **M9-D — Prefill matrix path:** measure prompt prefill separately and assess
  tiled/repacked quantized GEMM only if projection matmul is a significant
  bottleneck. Keep the decode path and model outputs unchanged.
- **M9-E — Final validation:** compare the same GGUF, prompt, output-token
  count, worker count, compiler/build, and host conditions against llama.cpp.
  Report prefill and decode separately, output parity, median/spread, and peak
  RSS. Historical or synthetic kernel timings are not substitutes for this
  end-to-end evidence.

M9 changes should be narrow and reversible: one measured bottleneck per
change, a focused numerical regression test, and the same two-mode
whole-model benchmark. Do not multiply kernel variants or create a large
cross-product test matrix.
