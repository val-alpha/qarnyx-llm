# Kimi K3 in C: research notes and Qarnyx CPU optimization follow-up

**Recorded:** 2026-11-17  
**Qarnyx repository:** `val-alpha/qarnyx-llm`  
**Qarnyx baseline reviewed:** `main` at `8b91794b31add1472cbea6099888ae97a63cd8ba` (`Add persistent CPU worker pool`)  
**Status:** M6 follow-ups A–D have initial implementations on `main`: parallel recurrent heads, batched independent projections, reusable per-session Qwen3.5 scratch, and opt-in AVX2/FMA direct-block Q4_K matvec. Workspace check/tests pass in GitHub Actions. Two A-only target-Ubuntu A/B pairs showed 20.8% and 11.6% lower elapsed time (16.6% lower two-sample mean), with identical displayed output. B/C/D end-to-end timings have not yet been measured on the target workstation. M6 remains in progress; fused Q5_K/Q6_K, AVX-512/NEON, and further performance validation remain outstanding.

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

### 4.5 Repeated worker baseline update (2026-11-17)

The user ran the fixed Qwen3.5-2B generation workload six rounds at worker counts 8, 9, and 10, with the round order varied after the initial three rounds. One Round 6 / 9-worker log was empty and is excluded. No further worker-count testing is planned for this milestone.

| Workers | Valid samples | Mean | Median | Min | Max |
|---:|---:|---:|---:|---:|---:|
| 8 | 6 | 21.12 s | 21.27 s | 19.75 s | 22.22 s |
| 9 | 5 | 20.54 s | 20.82 s | 19.17 s | 21.31 s |
| 10 | 6 | **19.08 s** | **19.10 s** | **18.39 s** | **19.79 s** |

Ten workers was the fastest of these three tested configurations by both mean and median: approximately 9.6% faster than eight workers and 7.1% faster than nine workers by mean elapsed time. Use ten workers as the provisional end-to-end comparison setting for the next M6 A/B test. This remains one machine and one short workload; it does not establish a universal worker-count optimum.

### 4.6 M6-A end-to-end before/after result (2026-11-17)

The user ran the same Qwen3.5-2B generation workload once with the pre-M6 binary and once with the M6 binary on the target Ubuntu workstation.

| Setting | Value |
|---|---|
| Model | `models/Huihui-Qwen3.5-2B-abliterated.Q4_K_M.gguf` |
| Prompt | `hi there!` |
| Max new tokens | 32 |
| Sampling | temperature 0, top-k 0, top-p 1 |
| Workers / CPU kernel | 10 / `auto` |
| Profiling | `QARNYX_PROFILE=1` |

| Binary | Wall time |
|---|---:|
| Before M6 | 28.35 s |
| After M6 | 22.45 s |
| Observed difference | **5.90 s faster (20.8%)** |

The displayed generated text was identical in both runs:
`hi there! I am a 25 year old male, 5'5" tall, 180 lbs, and I have a 10 year old`

The post-M6 profile recorded `linear_recurrent_heads` 630 times, with 475.127 ms aggregate measured time and 754.170 µs average per call. This is the new section that covers recurrent-head state updates and head output work.

**Interpretation:** this is a promising initial end-to-end result and is consistent with the intended optimization. It is one before/after pair only, however. It does not establish statistical significance or prove that all 5.90 seconds came from the recurrent-head change; run-to-run system variation may contribute. Preserve the result as an observed measurement, not a universal speedup claim. The user considers the current testing sufficient and has requested no further benchmark sweep for this stage.

### 4.7 Second M6-A end-to-end before/after pair (2026-11-17)

The user repeated the same release-binary A/B comparison on the target Ubuntu workstation with the same model, prompt, token limit, sampling settings, ten workers, and `auto` CPU kernel.

| Binary | Wall time |
|---|---:|
| Before M6 | 23.19 s |
| After M6 | 20.51 s |
| Observed difference | **2.68 s faster (11.6%)** |

The displayed generated text was identical in both runs:
`hi there! I am a 25 year old male, 5'5" tall, 180 lbs, and I have a 10 year old`

The post-M6 profile recorded `linear_recurrent_heads` 630 times, with 550.594 ms aggregate measured time and 873.959 µs average per call. The before-M6 profile's `layer_linear_total` row showed 13,221.142 ms. The after-M6 excerpt starts at `output_logits_matvec`, so its `layer_linear_total` value is not available from the supplied `tail -n 20` output and should not be inferred.

**Combined observation across the two A/B pairs:** the first pair was 28.35 s before versus 22.45 s after (20.8% lower); this second pair was 23.19 s before versus 20.51 s after (11.6% lower). Across the two samples per arm, the arithmetic mean was 25.77 s before and 21.48 s after, a 4.29 s / 16.6% lower mean elapsed time. Both pairs produced identical displayed text before and after.

This second pair strengthens the evidence that the direction of the improvement is repeatable on this machine, but two pairs are still too few to establish statistical significance or isolate all of the observed difference to the recurrent-head scheduling change. The user considers testing sufficient and has requested no further benchmark sweep at this stage.

### 4.8 M6-B/C/D implementation and CI validation (2026-11-17)

The following changes are implemented on `main` after the M6-A measurements:

**M6-B — batched independent projections**
- Added `Backend::matvec_batch`, with a sequential default for backends that do
  not provide parallel scheduling.
- `CpuBackend` runs independent projections over the existing persistent
  Rayon pool. It creates no additional worker pool or per-projection OS
  threads.
- Qwen3.5 now batches the recurrent-layer QKV/gate/beta/alpha projections,
  full-attention Q/K/V projections, and FFN gate/up projections.
- Profile labels now report aggregate batch timing as
  `linear_projection_batch`, `attention_projection_batch`, and
  `ffn_projection_batch` rather than separate timing for every projection
  inside those groups.

**M6-C — reusable per-session scratch**
- `Qwen35Cache` owns reusable hidden-state ping-pong buffers and a workspace
  for normalization, projections, convolution output, recurrent output,
  attention, scores, and FFN intermediates.
- Qwen3.5 layer execution writes into caller-provided output slices and reuses
  vector capacity across layers and decode steps. Cache clear retains scratch
  capacity for reuse.
- The gate/up combination keeps the original SiLU-then-multiply arithmetic
  order to avoid an unnecessary numerical behavior change.

**M6-D — direct-block Q4_K SIMD**
- The opt-in `fused-q4k` path now dispatches to an AVX2 or AVX2/FMA packed-block
  dot kernel on supported x86_64 CPUs. It expands quant values in vector
  registers and does not materialize an F32 weight row.
- Scalar direct-block Q4_K remains the fallback where AVX2 is unavailable.
- The `auto` mode remains the dequantize-plus-AVX2/FMA baseline. The fused
  mode is not made the default pending target-machine comparisons.

**Validation:** GitHub Actions run `44057801727` for commit
`ce72d083ed106dd889225982e8e32a12802cec12` passed `cargo check --workspace`
and `cargo test --workspace`. The CPU crate had 25 passing tests and the
Qwen3.5 model crate had 7 passing tests, including batch projection checks,
parallel fused-Q4_K versus scalar-reference coverage, scratch-capacity reuse,
and the exact recurrent-head reference test. The overall workflow remains
red only at `cargo fmt --all -- --check`, which reports existing formatting
differences in workspace files such as `qarnyx-tokenizer`; formatting also
failed before these changes. No local machine build was run by the assistant.

**Performance status:** no end-to-end M6-B/C/D benchmark result has been
captured yet. The implementation is ready for the user's Ubuntu A/B comparison
described in section 4.9; correctness CI success does not prove a wall-time
improvement.

### 4.9 Target Ubuntu A/B comparison for M6-B/C/D

This comparison separates the changes as far as the current CLI permits:
- M6-A binary with `auto` versus new B/C/D binary with `auto` measures the
  impact of projection batching and scratch reuse in the default path.
- New B/C/D binary with `auto` versus `fused-q4k` measures the opt-in Q4_K
  direct SIMD path separately.
- Keep model, prompt, token limit, sampling, worker count, and machine fixed.

First inspect local changes and save the existing M6-A release binary **before**
pulling or rebuilding:

```bash
git status --short
cp target/release/qarnyx /tmp/qarnyx-m6a

QARNYX_PROFILE=1 /usr/bin/time -f 'elapsed_seconds=%e' \
  /tmp/qarnyx-m6a generate \
  models/Huihui-Qwen3.5-2B-abliterated.Q4_K_M.gguf \
  "hi there!" --max-new-tokens 32 \
  --temperature 0 --top-k 0 --top-p 1 \
  --workers 10 --cpu-kernel auto \
  > /tmp/qarnyx-m6a.log 2>&1
```

Then update the existing `main` checkout and build:

```bash
git pull --ff-only origin main
cargo build --release -p qarnyx-cli
```

Run the new build first with `auto`, then with `fused-q4k`:

```bash
QARNYX_PROFILE=1 /usr/bin/time -f 'elapsed_seconds=%e' \
  ./target/release/qarnyx generate \
  models/Huihui-Qwen3.5-2B-abliterated.Q4_K_M.gguf \
  "hi there!" --max-new-tokens 32 \
  --temperature 0 --top-k 0 --top-p 1 \
  --workers 10 --cpu-kernel auto \
  > /tmp/qarnyx-m6bcd-auto.log 2>&1

QARNYX_PROFILE=1 /usr/bin/time -f 'elapsed_seconds=%e' \
  ./target/release/qarnyx generate \
  models/Huihui-Qwen3.5-2B-abliterated.Q4_K_M.gguf \
  "hi there!" --max-new-tokens 32 \
  --temperature 0 --top-k 0 --top-p 1 \
  --workers 10 --cpu-kernel fused-q4k \
  > /tmp/qarnyx-m6bcd-fused.log 2>&1

tail -n 30 /tmp/qarnyx-m6a.log
tail -n 30 /tmp/qarnyx-m6bcd-auto.log
tail -n 30 /tmp/qarnyx-m6bcd-fused.log
```

The new profile groups will include `linear_projection_batch`,
`attention_projection_batch`, and `ffn_projection_batch`; these replace
some of the old per-projection rows, so compare aggregate labels carefully.
Check generated text or token IDs as well as elapsed time. The assistant has
not run this benchmark on the user's workstation.

## 5. Qarnyx M6 work inspired by Kimi

Follow-ups A–D now have initial implementations on `main`. Their end-to-end performance effects have not yet been established together; see sections 4.8–4.9. The broader M6 roadmap remains in progress because additional quantized kernels and target-machine performance evidence are outstanding.

### Experiment 0 — repeatable worker baseline (completed 2026-11-17)

The user completed six rounds of the selected `auto` workload at workers 8, 9, and 10, in varied order after the first three rounds. Full timing and profile logs were retained under `/tmp/qarnyx-m6-baseline/`. One Round 6 / 9-worker log was empty and excluded. No additional worker-count sweep is planned for this milestone.

The recorded results are summarized in section 4.5. Ten workers is the provisional setting for the M6 before/after comparison on this workload; this is not a universal optimum. Use the same model, prompt, token settings, release build, and ten-worker setting for the next A/B run. Compare exact generated output/token IDs as well as elapsed time.

### M6 follow-up A — parallel recurrent heads (implemented; initial end-to-end improvement observed)

**Why:** the 16 recurrent heads have separate state blocks. Kimi's KDA implementation uses the same general principle: parallelize across independent heads while preserving the sequential recurrence within each head.

**Implementation on main:**
- added a fixed-size mutable-chunk scheduling method to the shared Backend contract, with a sequential default for other backends;
- CpuBackend overrides it using the existing persistent Rayon pool, without creating another pool;
- each recurrent head updates its own disjoint state block, keeping the decay, memory read, and rank-one update loop order unchanged;
- each head then computes its output into a disjoint output slice using the same pool;
- the decoder layer loop remains sequential; only independent work inside a recurrent layer is scheduled concurrently;
- the model profile now reports the new work under linear_recurrent_heads.

**Correctness evidence:**
- a unit test compares q/k normalization, the complete recurrent state, and per-head output against a serial reference;
- the test checks exact equality at 1, 4, and 10 workers, because the arithmetic order within each head is unchanged;
- GitHub Actions run #163 at commit `41708c0` passed both `cargo check --workspace` and `cargo test --workspace`, including the new exact-reference test;
- the target-machine release generation run must still confirm end-to-end output/token stability.

**CI note:** the run's final conclusion remains failed because the repository-wide `cargo fmt --all -- --check` reports formatting differences across multiple existing Rust files. The same formatting step also failed before this M6 change (run #159 on `8b91794`), so it is existing repository-wide formatting debt, not a new compile/test failure. The CI workflow now runs build and tests before that formatting gate so future runs still expose compiler and test results.

**Performance evidence:** two target-machine before/after pairs showed 20.8% and 11.6% lower elapsed time, respectively, with identical displayed output in both pairs (see sections 4.6–4.7). The two-sample means were 25.77 s before and 21.48 s after, or 16.6% lower mean elapsed time. This remains a small sample, not a statistically established gain. The user has elected to stop benchmarking this stage here.

### M6 follow-up B — multi-projection scheduling (implemented; performance unmeasured)

Independent Qwen3.5 recurrent, attention, and FFN projections now use the
`matvec_batch` API. The CPU implementation schedules them using the existing
persistent Rayon pool; other backends retain the sequential default. Correctness
tests compare batch results to expected individual projection results and
validate weight/output count mismatches.

### M6 follow-up C — reusable per-session scratch (implemented; performance unmeasured)

The Qwen3.5 cache now owns the scratch buffers reused across layer/token
execution. Hidden states ping-pong between two retained buffers. Projection,
normalization, convolution, recurrent output, attention scores/output, and FFN
intermediates reuse capacity. The test suite checks capacity reuse and CI
checks the full workspace. The change is intended to reduce allocation churn;
its wall-time and peak-memory effects are not yet measured.

### M6 follow-up D — SIMD fused Q4_K path (implemented; performance unmeasured)

The opt-in `fused-q4k` matvec now uses AVX2/FMA direct-block arithmetic when
available, with a scalar fallback. Differential tests compare both serial and
parallel fused rows against the scalar reference within a documented tolerance.
The default `auto` path remains unchanged. Do not make fused mode the default
until target-machine measurements show it wins without an output regression.

Kimi's MXFP4 kernel remains an algorithmic reference only; Q4_K has its own
block decoder and vectorized implementation.

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
