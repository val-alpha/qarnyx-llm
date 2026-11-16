# Qarnyx

**Quantized intelligence. Engineered in Rust.**

Qarnyx is an experimental, Rust-native, CPU-first inference runtime for
[GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) language
models. It is being designed from scratch around quantized CPU kernels,
explicit memory ownership, and predictable resource usage.

> **Project status: early development.** The GGUF v3 reader streams metadata
> and tensor descriptors and validates complete tensor byte ranges without
> loading weight payloads into memory. Scalar reference decoding is implemented
> for F32, F16, Q4_0, Q8_0, Q4_K, Q5_K, and Q6_K. Transformer execution,
> tokenization, and text generation are not implemented yet.

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
| qarnyx-cpu | Scalar CPU kernels, SIMD dispatch, CPU execution |
| qarnyx-tokenizer | Tokenizer metadata and tokenization |
| qarnyx-models | Transformer model definitions, starting with LLaMA-style models |
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
4. **M3 - Scalar tensor engine:** implement matvec, matmul, RMSNorm, RoPE,
   softmax, SiLU, and residual operations.
5. **M4 - First model forward pass:** implement and validate one LLaMA-style
   model against an independent reference.
6. **M5 - Tokenization and generation:** add prefill, incremental decode,
   bounded KV cache, stopping rules, and sampling.
7. **M6 - CPU optimization:** add runtime-dispatched SIMD kernels and a
   persistent worker pool.
8. **M7 - Operational hardening:** add resource limits, cancellation, session
   isolation, stress tests, and reproducible benchmarks.

## Non-goals for the first milestone

- GPU execution
- Multi-model serving
- Distributed inference
- Broad support for every GGUF architecture and quantization type

## License

MIT. See LICENSE.
