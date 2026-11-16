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
2. Keep immutable model weights in a read-only mapped region where supported.
3. Resolve tensor metadata by name and expose checked, non-owning tensor views.
4. Load model configuration and map tensor names to a model implementation.
5. Allocate bounded per-session KV cache and reusable activation scratch.
6. Execute prefill, then incremental decode through the selected CPU backend.
7. Sample the next token and decode it with the tokenizer.

A mapped file is not equivalent to resident RAM usage. Pages are faulted in as
they are accessed; the operating system may reclaim file-backed pages. Track
process RSS and system page cache separately during benchmarks.

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
The current CPU backend uses scalar kernels. Once optimized implementations
are added, it will select a kernel per operation or kernel family based on
runtime CPU feature detection.

The scalar CPU backend implements:
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
storage lengths and decode only the active weight row into a reusable F32
scratch buffer; they do not materialize a full F32 matrix. RMSNorm requires a
positive finite epsilon. RoPE pairs the first and second halves of one head
vector, matching the split-half convention used by LLaMA-style rotary
embeddings. Softmax rejects empty or non-finite inputs.

The first implementation is scalar and single-threaded even when a worker
count greater than one is configured. The CPU microbenchmark target is run
with `cargo bench -p qarnyx-cpu --bench kernels`.

The scalar implementation is the correctness oracle. SIMD implementations
must be compared against it with documented tolerances. Do not enable a SIMD
path merely because it is faster; it must also pass correctness tests.

## CPU execution strategy

Start with a scalar backend and establish benchmarks before introducing SIMD.
Then add runtime-dispatched kernels for the actual VPS CPU features:
- x86_64: AVX2/FMA first; AVX-512 only when supported and measured to help;
- aarch64: NEON where applicable;
- all other configurations: scalar fallback.

Do not compile the default release build with target-cpu=native. That setting
can produce a binary that fails on a different CPU. Use it only for
host-specific builds and benchmark it separately.

Use a persistent worker pool rather than creating threads for each tensor
operation. Partition large matvec/matmul work into chunks, avoid nested
parallelism, and benchmark 1, 2, 4, 6, 8, and 10 workers. VPS CPU scheduling
and memory bandwidth may make fewer workers faster than all available vCPUs.

## Memory policy

- Model weights: immutable mapped GGUF bytes; no full-model F32 copy.
- KV cache: allocated per session with an explicit context limit.
- Activations: reusable scratch buffers sized from model configuration.
- Sessions: bounded; do not create unbounded per-request caches.
- Ownership: shared immutable model data, session-owned mutable state.
- Synchronization: keep locks out of hot tensor loops; avoid a global
  Mutex around model weights.
- Limits: validate context length, batch size, and allocation sizes before
  reserving memory.

FP16 KV cache is a target option, not an assumed universal implementation.
It must be tested for numerical stability and supported cache layouts.

## Initial model scope

Start with a LLaMA-style decoder-only Transformer with grouped-query attention
support. Do not claim broad compatibility with all models labelled LLaMA-like.
Validate the metadata keys, tensor names, RoPE parameters, normalization
epsilon, vocabulary, and tokenizer behavior for each supported model variant.

Additional architectures should be added only after the first model family has
end-to-end correctness tests.

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

### M4 - First model forward pass
- Load one supported LLaMA-style GGUF model.
- Implement embedding, attention, GQA, feed-forward, residuals, and logits.
- Compare logits against an independent reference runtime.

### M5 - Tokenization and generation
- Read tokenizer metadata and special tokens.
- Implement prompt prefill and incremental decode.
- Add KV cache, stopping rules, and sampling.
- Verify generated token IDs, not only rendered text.

### M6 - CPU optimization
- Add runtime feature detection and AVX2/FMA kernels.
- Add other SIMD paths only when the host warrants them.
- Implement a persistent worker pool and avoid nested scheduling.
- Benchmark prefill tokens/s, decode tokens/s, latency, and peak RSS.

### M7 - Operational hardening
- Add context and memory limits, cancellation, session isolation, and stress tests.
- Add CLI and optional HTTP serving only after the library is stable.
- Publish reproducible benchmark methodology and known limitations.

## Benchmark methodology

Always record:
- exact model file and quantization;
- CPU model, vCPU allocation, feature flags, and compiler version;
- thread count and affinity policy;
- prompt length, generation length, and context size;
- prefill tokens/s and decode tokens/s separately;
- first-token latency and steady-state latency;
- peak RSS, swap activity, and memory pressure.

Compare against other runtimes using the same model file, prompt, token count,
CPU allocation, and thread settings. Do not infer performance from synthetic
kernel tests alone.
