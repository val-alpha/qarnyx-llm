# Kernforge

**Forge performance. Own every kernel.**

Kernforge is an experimental, Rust-native, CPU-first inference runtime for
[GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) language
models. It is being designed from scratch around quantized CPU kernels,
explicit memory ownership, and predictable resource usage.

> **Project status: early scaffold.** GGUF parsing, quantized kernels,
> transformer execution, tokenization, and text generation are not implemented
> yet. The current repository establishes the workspace and architectural
> boundaries for that work.

## Design goals

- **Rust-native runtime:** no C/C++ inference engine as a runtime dependency.
- **CPU first:** scalar reference kernels first, followed by runtime-dispatched
  SIMD implementations where the host CPU supports them.
- **GGUF support:** read model metadata and tensor layouts directly from GGUF.
- **Quantized execution:** operate on packed quantized weights without
  materializing the entire model as F32.
- **Memory efficiency:** read-only model mapping, bounded KV cache, and reusable
  activation buffers.
- **Correctness before speed:** compare optimized kernels against scalar
  references and validate model outputs against independent implementations.

Rust does not automatically make a program leak-free or faster than C/C++.
Kernforge will use safe Rust by default, isolate required unsafe code, and
measure performance on the same hardware as its benchmarks.

## Planned architecture

| Crate | Responsibility |
|---|---|
| kernforge-core | Tensor views, data types, backend contracts, shared errors |
| kernforge-gguf | GGUF header, metadata, tensor directory, file mapping |
| kernforge-quant | Quantization block layouts and reference decoders |
| kernforge-cpu | Scalar CPU kernels, SIMD dispatch, CPU execution |
| kernforge-tokenizer | Tokenizer metadata and tokenization |
| kernforge-models | Transformer model definitions, starting with LLaMA-style models |
| kernforge-runtime | Prefill, decode, KV cache, sampling, generation |
| kernforge-cli | Command-line interface |

See [docs/architecture.md](docs/architecture.md) for the architecture,
backend contract, memory strategy, and implementation roadmap.

## Initial implementation order

1. Inspect host CPU capabilities and establish reproducible benchmarks.
2. Implement a defensive GGUF v3 reader and tensor metadata validation.
3. Add scalar reference decoding for F32, F16, Q8_0, Q4_0, and K-quants.
4. Build and test scalar tensor operations.
5. Implement one LLaMA-style forward pass and validate logits.
6. Add tokenizer, prefill, incremental decode, and KV cache.
7. Optimize hot kernels with runtime-dispatched SIMD.
8. Add resource limits and production-facing interfaces.

## Development

A recent stable Rust toolchain with Cargo is required.

```bash
cargo check --workspace
cargo test --workspace
cargo build --release
```

The current code is an architectural scaffold. There is no working model loader
or inference command yet.

For host-specific release builds, benchmark a build using
RUSTFLAGS="-C target-cpu=native". Such binaries are specific to the CPU
features available on the build machine and should not be distributed as
portable binaries without an appropriate dispatch/build strategy.

## Non-goals for the first milestone

- GPU execution
- Multi-model serving
- Distributed inference
- Broad support for every GGUF architecture and quantization type

## License

MIT. See LICENSE.
