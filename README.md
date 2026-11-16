# Qarnyx

**Quantized intelligence. Engineered in Rust.**

Qarnyx is an experimental, Rust-native, CPU-first inference runtime for
[GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) language
models. It is being designed from scratch around quantized CPU kernels,
explicit memory ownership, and predictable resource usage.

> **Project status: early development.** The GGUF v3 reader validates tensor
> descriptors and payload ranges without loading weights during inspection.
> Scalar decoders support F32, F16, Q4_0, Q8_0, Q4_K, Q5_K, and Q6_K. The CPU
> backend retains runtime-dispatched AVX2/FMA matvec dot products and matmul
> output tiles, and includes an opt-in fused direct-block Q4_K matvec path.
> Experimental Qwen3.5 2B GGUF loading and text generation are implemented;
> numerical validation, quantized-kernel fusion, and worker-pool scheduling
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

See [docs/architecture.md](docs/architecture.md) for the architecture,
backend contract, memory strategy, and implementation roadmap.

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
currently registered storage types. The default automatic backend keeps the AVX2/FMA path and dequantizes Q4_K
rows into reusable F32 scratch before the dot product. An experimental
`fused-q4k` mode computes Q4_K matvec directly from packed blocks without
creating an F32 scratch row; that fused dot currently uses scalar arithmetic.
Other quantized matrix operations still decode one row at a time, and the full
weight matrix is never expanded.
Worker scheduling is not active yet, so `--workers` does not currently create
parallel threads.

Run the CPU microbenchmarks with `cargo bench -p qarnyx-cpu --bench kernels`.
The harness compares the scalar reference, default runtime dispatch, and
opt-in fused Q4_K matvec paths.

To compare end-to-end generation using the same model and prompt, select a
kernel mode explicitly:

```bash
./target/release/qarnyx generate model.gguf "hi there!" --max-new-tokens 32 --cpu-kernel scalar
./target/release/qarnyx generate model.gguf "hi there!" --max-new-tokens 32 --cpu-kernel auto
./target/release/qarnyx generate model.gguf "hi there!" --max-new-tokens 32 --cpu-kernel fused-q4k
```

The `auto` mode is the existing AVX2/FMA baseline with row dequantization.
`fused-q4k` enables the experimental direct-block Q4_K matvec while retaining
AVX2/FMA for other eligible kernels. `scalar` remains the non-SIMD
correctness reference.

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
7. **M6 - CPU optimization:** initial x86_64 AVX2/FMA matvec and matmul kernels
   are implemented with scalar fallbacks and scalar-vs-runtime benchmarks.
   An opt-in mode provides direct fused Q4_K matvec; fused Q5_K/Q6_K kernels,
   fused SIMD implementations, NEON, and a persistent worker pool remain
   planned.
8. **M7 - Operational hardening:** add resource limits, cancellation, session
   isolation, stress tests, and reproducible benchmarks.

## Non-goals for the first milestone

- GPU execution
- Multi-model serving
- Distributed inference
- Broad support for every GGUF architecture and quantization type

## License

MIT. See LICENSE.
