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
and FMA, the automatic Q4_K/Q5_K/Q6_K matvec path computes directly from packed
K-quant blocks using F32-input AVX2/FMA kernels; CPUs without both features keep
the runtime-SIMD dequantize-plus-dot fallback. The previous dequantized mode is
retained as an explicit Rust API baseline through `CpuBackend::new_dequant()`.

General matvec and matmul fallbacks reuse a per-thread F32 scratch allocation.
Large operations are partitioned by encoded weight bytes into several tasks
per worker, while small operations avoid Rayon scheduling overhead. F32-input
matmul decodes up to four weight rows together and uses an AVX2/FMA register
tile to reuse each input vector across those rows when the tile and shape fit.
This is a F32 GEMM micro-kernel after row decoding, not a packed Q4_Kx8 GEMM.
RMSNorm requires a positive finite epsilon. RoPE pairs the first and second
halves of one head vector, matching the split-half convention used by
LLaMA-style rotary embeddings. Softmax rejects empty or non-finite inputs.

The explicit `fused-q4k` mode computes directly from packed Q4_K blocks without
materializing an F32 row. It uses runtime AVX2/FMA on supported x86_64 CPUs and
falls back to the scalar direct-block dot elsewhere. The explicit `fused-qk`
mode extends direct-block matvec to Q5_K and Q6_K, using AVX2 on supported
x86_64 CPUs and scalar direct-block references elsewhere; it also enables the
fused Q4_K path. The automatic mode selects the F32-input direct-block K-quant
kernels when AVX2+FMA are available. The explicit `llama-q4k` and
`fused-q4k` modes remain available for controlled A/B comparisons. RMSNorm,
RoPE, softmax, and residual addition currently use scalar implementations.

The `workers` setting controls a persistent Rayon thread pool owned by the CPU
backend. The pool is created once during backend construction and reused for
large matvec/matmul operations. `matvec_batch` schedules independent
projections that share an input on that same pool; it does not create a second
pool or per-projection OS threads. Each output remains disjoint, and every
projection completes before the model continues. Small matrices remain on the
single-threaded path to avoid scheduling overhead. Benchmark worker counts
1, 2, 4, 6, 8, and 10 on the target VPS because scheduler contention and memory
bandwidth may make fewer workers faster. Run
`cargo bench -p qarnyx-cpu --bench kernels` to compare scaling for F32 and Q4_K
matvec in addition to the storage-format comparisons.

The scalar implementation is the correctness oracle. SIMD implementations
must be compared against it with documented tolerances and must pass
correctness tests before dispatch is enabled.

## CPU execution strategy

Keep the scalar backend as the correctness reference and dispatch by runtime
feature detection:
- x86_64: AVX2/FMA matvec and four-row F32 GEMM micro-kernels are implemented;
- x86_64: automatic direct-block Q4_K/Q5_K/Q6_K matvec uses AVX2+FMA when
  supported, with dequantized SIMD and scalar fallback paths;
- x86_64: explicit Q8_K-based Q4_K and integer K-quant alternatives remain
  selectable for A/B comparisons;
- aarch64: NEON kernels remain planned;
- all other configurations: scalar fallback.

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

Implementation limits (not a claim of full vendor API parity):
- The API currently serves one loaded GGUF model per server process and
  supports the LLaMA and Qwen3.5 runtime architectures.
- Multimodal content, stop sequences, OpenAI logprobs, broad vendor-specific
  parameter parity, model-specific LLaMA chat templates, and built-in
  authentication remain outstanding.
- Anthropic system and message content is text-only. Unsupported content block
  types and unsupported request fields are rejected rather than silently
  ignored.
- The native endpoint is a deliberately small, explicit runtime surface. It
  exposes the currently supported generation configuration, not internal
  model/cache implementation details.
- Tool execution is always performed by the client. Qarnyx formats and returns
  tool calls and accepts the corresponding tool-result history; it does not
  invoke external functions.

All three adapters share the runtime event source and the same request
admission semaphore. They must not duplicate model forward passes, decode
loops, or model-specific prompt-formatting logic. Model-specific chat
formatting remains below M8: Qwen3.5 ChatML templates, thinking-mode control
tokens, and end-of-turn handling belong to tokenizer/model integration rather
than HTTP protocol adapters.

## Benchmark methodology

The repeatable CLI benchmark procedure, command template, CSV fields, warm-up
rules, and interpretation limits are maintained in
[docs/benchmarking.md](benchmarking.md). CLI wall-clock results include model
loading and request setup as well as generation; they are not decode-only
throughput measurements. Do not infer end-to-end performance from synthetic
kernel tests alone.
