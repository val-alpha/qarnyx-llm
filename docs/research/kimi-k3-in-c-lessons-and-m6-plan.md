# Kimi K3 in C: research notes and Qarnyx CPU optimization follow-up

**Recorded:** 2026-11-17  
**Qarnyx repository:** `val-alpha/qarnyx-llm`  
**Qarnyx baseline reviewed:** `main` at `8b91794b31add1472cbea6099888ae97a63cd8ba` (`Add persistent CPU worker pool`)  
**Status:** research note and experiment plan; proposed follow-up work is not yet implemented.

This document preserves the findings that led to the next Qarnyx CPU optimization experiments. It is intended to be usable without the original chat history. Treat measurements as machine-specific observations, not universal performance claims.

## 1. The important architectural principle

Do **not** try to run Transformer layer (L+1) before layer (L) has produced the hidden state it consumes. Ordinary decoder layers have a real data dependency:

```text
hidden state
    |
  layer 0
    |
  layer 1
    |
  layer 2
    |
   ...
```

The useful opportunity is to find independent work *inside* a layer, or to overlap non-compute work (such as loading the next layer's weights) with the current layer's compute.

Kimi K3 in C is a good example of both approaches:
- parallel execution over independent output rows and attention/recurrent heads;
- a separate reader thread that prefetches upcoming layer weights while the main thread computes the current layer.

That is more accurate than saying that Kimi executes dependent Transformer layers concurrently.

## 2. Kimi K3 in C: what the project actually does

Reference repository: <https://github.com/FareedKhan-dev/kimi-k3-in-c>  
Author's overview: <https://medium.com/@fareedkhandev/building-kimi-k3-in-c-to-run-a-2-8t-model-on-consumer-hardware-a5792cbf3b59>  
The repository describes a CPU-only portable C99 engine, with no BLAS, inference framework, or GPU requirement. Its implementation uses OpenMP for selected parallel loops and pthreads for asynchronous trunk I/O.

### 2.1 Model and memory layout

The repository's architecture document describes the released Kimi K3 checkpoint as:
- 2.78T total parameters and approximately 104B active parameters per token;
- 93 layers: 69 KDA and 24 gated-MLA layers;
- 896 routed experts per MoE layer, top-16 routing, plus two shared experts;
- MXFP4 routed expert weights;
- approximately 108.81 GB of dense trunk, 1.45 TB of routed experts, and a very large checkpoint on disk.

The engine's central memory idea is to keep the always-needed pieces resident within a user-selected budget, while streaming other weights from disk. These dimensions and figures belong to Kimi K3, not Qarnyx's Qwen3.5-2B model.

Source: <https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/docs/ARCHITECTURE.md>

### 2.2 Parallel work inside a layer

In `src/core/k3_ops.c`, Kimi uses OpenMP for operations whose output regions are independent:
- matrix multiplication output rows;
- KDA recurrent heads;
- MLA attention heads;
- other independent vector/output work.

The important correctness property is that each worker writes to a disjoint output/state region. The arithmetic sequence *inside* a row or recurrent head can remain unchanged, avoiding a cross-worker floating-point reduction.

Examples:
- <https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/src/core/k3_ops.c>
- <https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/benchmarks/bench_kernels.c>

### 2.3 Layer I/O pipeline: overlap, not layer-level compute parallelism

In `src/io/k3_trunk.c`, the engine has:
- a dedicated pthread reader;
- a ring buffer with multiple slots;
- a queue of upcoming layers;
- prefetch/lookahead in the known layer traversal order;
- slot-state protection so a slot being read, currently consumed, or needed imminently is not overwritten;
- chunked reads, with parallel I/O work where available.

The main thread can compute on the current layer while the reader fetches a future layer. The next layer's computation still waits for the current layer's result.

The implementation also uses aligned direct-I/O paths and huge-page advice for large buffers. These are storage/memory-management techniques, not an automatic speed-up for every model or machine.

Source: <https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/src/io/k3_trunk.c>

### 2.4 MoE expert loading and batched prefill

Kimi's `src/cache/k3_cache.c` and `src/core/k3_ops.c` implement a model-specific MoE path:
1. determine selected experts and reserve distinct cache slots serially;
2. read the expert payloads concurrently into those slots;
3. publish the slots only after each read has succeeded.

The code sorts requests by their actual disk offsets and may split reads into smaller aligned chunks to improve I/O concurrency and reduce tail latency.

For prompt prefill, it also collects the unique experts selected across a chunk of tokens, reads each unique expert once, then reuses it for all tokens in that chunk. Its correctness approach preserves the original per-token expert accumulation order.

These are valuable designs for a future Qarnyx MoE backend, but they are **not directly reusable** for today's Qwen3.5-2B path because that model path does not implement Kimi-style routed experts.

Sources:
- <https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/src/cache/k3_cache.c>
- <https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/src/core/k3_ops.c>

### 2.5 Quantized kernels

Kimi operates directly on its native BF16 trunk and packed MXFP4 expert weights. Its MXFP4 kernel consumes packed nibbles and scale metadata instead of expanding the full expert matrix to FP32 first. It also uses hand-written AVX2/FMA paths, with explicit accumulation layouts and correctness checks against scalar/reference results.

Do not confuse MXFP4 with GGML Q4_K. They have different block formats, scales, numerical rules, and kernel contracts. Kimi's kernel is an optimization reference, not code that can simply be dropped into Qarnyx's Q4_K implementation.

Source: <https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/src/core/k3_ops.c>

### 2.6 Interpreting Kimi's performance claims

The repository reports different time-per-token numbers at different memory budgets. Its performance campaign was run on a large AMD EPYC workstation with 124 CPU cores, substantial RAM, and NVMe storage. Those numbers cannot be compared directly with Qarnyx's Qwen3.5-2B measurements: model, hardware, memory residency, storage traffic, token counts, and benchmark conditions differ.

Importantly, Kimi's own performance documentation reports a 33.1% spread across three identical short runs in one measurement set, and warns that single-sample differences smaller than its noise floor are not established effects. This is a useful reminder for our own benchmark discipline.

Sources:
- <https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/docs/PERFORMANCE.md>
- <https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/docs/BENCHMARKING.md>
- <https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/docs/data/speed-2026-08.md>

## 3. Qarnyx state at the M5 / persistent-worker-pool checkpoint

### 3.1 Repository state reviewed

At the time of this review, Qarnyx `main` pointed to:

```text
8b91794b31add1472cbea6099888ae97a63cd8ba  Add persistent CPU worker pool
```

The persistent Rayon worker pool is already merged into `main`. The CPU backend owns a reusable `Arc<ThreadPool>`; large matvec/matmul work can partition independent output rows across that pool, while small operations can remain sequential.

The CLI exposes:
- `--cpu-kernel auto`: runtime SIMD dispatch, with the normal Q4_K row-dequantization path;
- `--cpu-kernel fused-q4k`: experimental direct packed-block Q4_K dot path (the current fused Q4_K dot uses scalar arithmetic);
- `--cpu-kernel scalar`: non-SIMD reference path.

The current Qwen3.5-2B model implementation has 24 layers, hidden width 2048, FFN width 6144, 16 recurrent heads of width 128, and 8 full-attention query heads. The implementation traverses decoder layers sequentially and carries recurrent/convolution state or attention KV state between generation steps.

Relevant source:
- <https://github.com/val-alpha/qarnyx-llm/blob/main/crates/qarnyx-cpu/src/lib.rs>
- <https://github.com/val-alpha/qarnyx-llm/blob/main/crates/qarnyx-models/src/qwen35.rs>
- <https://github.com/val-alpha/qarnyx-llm/blob/main/crates/qarnyx-runtime/src/lib.rs>

### 3.2 Already-identified parallelism opportunities

The current implementation contains several candidate regions:
- **Recurrent heads:** each head owns its own recurrent-state block and output slice. The time recurrence within one head is ordered, but different heads do not need to update the same state block.
- **Independent projections:** several projections consume the same normalized input but write to separate outputs. Examples include recurrent-layer gate/beta/alpha projections and attention Q/K/V projections.
- **FFN gate and up projections:** both consume the same normalized FFN input; their outputs are combined only after both are available.
- **Attention heads:** per-query-head attention work has separate output slices, subject to the shared KV data being read-only during that phase.

Caution: each backend matvec already schedules work on the persistent Rayon pool. Adding an independent thread pool or blindly nesting parallel loops can add overhead or oversubscribe the CPU. Prefer a design that lets the existing backend own scheduling, such as a batch/multi-operation backend API or carefully scoped work on the same pool.

### 3.3 Current weight-loading model

The CLI currently reads the GGUF file into a byte vector with `fs::read()`, and model tensors reference that in-memory payload. Therefore, Kimi's asynchronous disk-to-ring trunk pipeline is not the first optimization to port to the current 2B workload.

It becomes relevant if Qarnyx later adds file-backed tensor views, memory mapping, or a bounded streaming loader for models that do not fit comfortably in RAM.

## 4. Qarnyx benchmark record

**Workload used in the captured experiment**

- Model: `models/Huihui-Qwen3.5-2B-abliterated.Q4_K_M.gguf`
- Prompt: `hi there!`
- `--max-new-tokens 32`
- `--temperature 0 --top-k 0 --top-p 1`
- Profiling: `QARNYX_PROFILE=1`
- External elapsed time: `/usr/bin/time`
- CPU kernel: as shown in the tables

### 4.1 Kernel-mode comparison at 10 workers

| Kernel mode | Elapsed | Relative to auto |
|---|---:|---:|
| `auto` | 21.37 s | baseline |
| `fused-q4k` | 22.41 s | about 4.9% slower |
| `scalar` | 27.50 s | about 28.7% slower |

Each mode was measured once. This supports using `auto` as the provisional choice for this workload, but it is not a replicated statistical result. The current fused Q4_K path should not be made the default based on the name alone.

### 4.2 Worker sweep using `auto`

| Workers | Elapsed |
|---:|---:|
| 1 | 79.48 s |
| 2 | 45.43 s |
| 3 | 38.50 s |
| 4 | 36.71 s |
| 5 | 27.81 s |
| 6 | 27.74 s |
| 7 | 23.96 s |
| 8 | 23.37 s |
| 9 | **21.33 s** |
| 10 | 21.83 s |

These are one run per worker count, in ascending order, on the user's Ubuntu environment. Nine workers was the fastest *observed* point in this sweep. The 9-vs-10 difference is small and should not be called a proven optimum until repeated runs confirm it.

### 4.3 Selected `auto` profile at 10 workers

| Profile label | Calls | Aggregate time |
|---|---:|---:|
| `layer_linear_total` | 630 | 12.311 s |
| `output_logits_matvec` | 35 | 3.842 s |
| `layer_attention_total` | 210 | 2.905 s |
| `ffn_down_matvec` | 840 | 2.668 s |
| `ffn_gate_matvec` | 840 | 2.538 s |
| `ffn_up_matvec` | 840 | 2.531 s |
| `linear_qkv_matvec` | 630 | 2.202 s |

Profile labels are nested: for example, matvecs contribute to the layer totals. Do **not** sum all profile rows and treat the result as wall time. The profile is useful for locating hot sections, not as a disjoint time partition.

### 4.4 Benchmark caveats

- There was one measurement per arm; no repeated-run mean or variance was captured.
- The worker sweep was run in ascending order, so thermal state, background load, and cache state may bias later measurements.
- Record CPU model, physical/logical core counts, RAM, OS, kernel, compiler/build flags, storage, and background load with the next campaign.
- Keep prompt, token count, sampling settings, binary, and model fixed when comparing a change.
- Report every run, not only the fastest. A small speed difference is not meaningful until it exceeds measured run-to-run variation.
- Always verify generated token IDs or text against the pre-optimization baseline when the change is intended to preserve numerical behavior.

## 5. Proposed Qarnyx experiments inspired by Kimi

These are **proposals**, not completed features and not a claim that all are part of the existing official M6 roadmap. The existing README milestone description remains authoritative until deliberately updated.

### Experiment 0 — establish a repeatable baseline

Before changing code:
1. Build release once and use the same binary for all arms.
2. Run the selected `auto` workload at workers 8, 9, and 10 at least three times each (ideally interleave the order rather than always running ascending).
3. Retain full wall-clock outputs and `QARNYX_PROFILE=1` reports.
4. Capture host details (`lscpu`, `free -h`, `uname -a`, filesystem/device information).
5. Record generated token IDs or exact output.

Treat 9 workers as a provisional configuration, not a permanent default, until this experiment confirms it.

### M6 follow-up A — parallel recurrent heads

**Why:** the 16 recurrent heads have separate state blocks. Kimi's KDA implementation uses the same general principle: parallelize across independent heads while preserving the sequential recurrence within each head.

**Implementation direction:**
- partition recurrent state into disjoint per-head mutable slices;
- give each task the corresponding head's q/k/v, decay/beta values, and output slice;
- preserve the existing inner-loop order within each head;
- use the existing Rayon pool or a backend-owned scheduling primitive, not a second pool.

**Correctness gates:**
- compare all recurrent state values after a step between serial and parallel implementations;
- compare resulting hidden vectors/logits;
- test multiple worker counts, including 1 and the configured pool size;
- require bitwise equality when the arithmetic order is unchanged; otherwise define and justify a tight numeric tolerance and also verify identical greedy token IDs.

**Performance gate:** the head-parallel path must improve end-to-end elapsed time on repeated runs, not merely a synthetic head microbenchmark. It must not regress small workloads.

### M6 follow-up B — multi-projection scheduling

**Why:** Q/K/V or gate/up operations can share an input while producing disjoint output buffers.

**Implementation direction:**
- investigate a backend API such as a batch of independent matvec operations;
- execute these operations on the same persistent Rayon pool;
- avoid creating OS threads or separate pools per projection;
- benchmark both operation-level parallel scheduling and the current sequential-call approach.

**Correctness gates:** each output must match the current implementation; ensure no mutable output aliasing and no nested-pool oversubscription.

### M6 follow-up C — FFN fusion and reusable scratch

Investigate:
- shared scheduling for gate/up projections;
- fusion of activation and elementwise gate/up combination;
- reusing per-generation/per-layer buffers instead of repeatedly allocating `Vec<f32>`;
- scratch sizing and lifetime ownership that remain explicit and safe.

Fusion should be driven by memory traffic and measured allocation/scheduling cost, not by reducing source-code line count. Do not fuse operations if it increases temporary memory or slows the critical path.

### M6 follow-up D — improve the fused Q4_K path

The current `fused-q4k` path is slower than `auto` in the captured end-to-end test. Profile and microbenchmark it before changing it:
- separate packed-block decode cost from dot-product cost;
- test AVX2/FMA implementations against the scalar reference;
- retain the dequantized `auto` route as a baseline;
- verify quantized block edge cases and numerical agreement;
- only select the fused path automatically if repeated end-to-end tests demonstrate a win.

Kimi's MXFP4 kernel is an algorithmic reference only; Q4_K needs its own correct block decoder and vectorization.

### Later research — prefill batching and file-backed weights

These are larger follow-on projects:
- **Batched prefill:** process multiple prompt tokens in a model path that preserves causal attention and recurrent state semantics. Kimi's unique-expert batching is specific to MoE, but the broader principle of reusing weights across prompt tokens is relevant.
- **File-backed / streamed weights:** if model size exceeds available RAM, investigate memory mapping or a read-ahead ring that overlaps layer loading with current-layer compute. This is not expected to be the first win for the current fully loaded Qwen3.5-2B model.

## 6. Experiment rules and project guardrails

1. Work on **`main` only** unless the project owner explicitly changes this rule. Do not create an experiment branch by default.
2. Never overwrite or discard local modifications without inspecting and preserving them first. A successful `cargo test` on a local tree does not prove that a failed `git pull` updated the source.
3. Benchmark release builds; keep the binary, model, prompt, generation parameters, and machine constant across A/B runs.
4. Change one variable at a time. Keep scalar/reference implementations available for correctness comparisons.
5. No optimization is accepted on timing alone: require correctness tests, repeated end-to-end measurements, and no material regression.
6. Prefer one persistent scheduler. Avoid nested independent pools and unbounded thread creation.
7. Distinguish **compute parallelism**, **I/O overlap**, **batching**, and **memory-layout optimization**. They solve different bottlenecks.
8. Keep the official milestone/roadmap description synchronized only when a proposed item is accepted and actually implemented.

### Reconnecting to the project after losing chat context

Start in the existing repository:

```bash
git branch --show-current
git status --short
git log -1 --oneline
git pull --ff-only origin main
cargo check --workspace
cargo test --workspace
cargo build --release -p qarnyx-cli
```

Expected branch: `main`. Inspect and preserve any local changes before pulling. Do not blindly reset, clean, drop stashes, or pop old stashes while trying to resume work.

The original worker sweep can be reproduced with this command shape:

```bash
for workers in 1 2 3 4 5 6 7 8 9 10; do
  echo "========== WORKERS: $workers | KERNEL: auto =========="
  QARNYX_PROFILE=1 \
  /usr/bin/time -f 'elapsed: %e s' \
  ./target/release/qarnyx generate \
    models/Huihui-Qwen3.5-2B-abliterated.Q4_K_M.gguf \
    "hi there!" \
    --max-new-tokens 32 \
    --temperature 0 \
    --top-k 0 \
    --top-p 1 \
    --workers "$workers" \
    --cpu-kernel auto
done
```

## 7. Reference links

### Qarnyx
- <https://github.com/val-alpha/qarnyx-llm>
- <https://github.com/val-alpha/qarnyx-llm/blob/main/README.md>
- <https://github.com/val-alpha/qarnyx-llm/blob/main/docs/architecture.md>
- <https://github.com/val-alpha/qarnyx-llm/blob/main/crates/qarnyx-models/src/qwen35.rs>
- <https://github.com/val-alpha/qarnyx-llm/blob/main/crates/qarnyx-cpu/src/lib.rs>

### Kimi K3 in C
- <https://github.com/FareedKhan-dev/kimi-k3-in-c>
- <https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/docs/ARCHITECTURE.md>
- <https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/docs/PERFORMANCE.md>
- <https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/docs/TUNING.md>
- <https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/docs/ROADMAP.md>
- <https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/src/core/k3_ops.c>
- <https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/src/io/k3_trunk.c>
- <https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/src/cache/k3_cache.c>
