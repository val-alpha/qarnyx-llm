> **Current kernel policy:** only `auto` and `fused-kq-fma` are selectable.
> Historical benchmark sections below may contain retired kernel names and
> measurements from earlier revisions; those names are not valid CLI options
> on the current branch.

# Qarnyx benchmarking protocol

This document defines a repeatable end-to-end CLI comparison. Keep the same
model file, prompt, token count, context limits, build, and host conditions
when comparing CPU kernels or worker counts.

## Record the environment

From the repository root, record:

```bash
git rev-parse HEAD
rustc --version
cargo --version
uname -a
lscpu
free -h
```

Also record the exact GGUF path, file size, quantization, CPU feature flags,
available vCPUs, worker count, kernel mode, context limit, prompt, and sampling
settings. Avoid running unrelated CPU-heavy workloads during measurements.
Do not use `-C target-cpu=native` for a portable release binary; if a
host-specific build is benchmarked, label it separately.

## Build and run a controlled matrix

Build once before collecting measurements:

```bash
cargo build --release -p qarnyx-cli
python3 scripts/cpu_kernel_cli_test.py --bin target/release/qarnyx
```

The Python smoke test verifies that help advertises exactly `auto` and
`fused-kq-fma`, that both are accepted by the argument parser, and that the
retired `fused-q4k`, `llama-q4k`, `fused-qk`, and `scalar` CLI values are
rejected. It uses a deliberately missing model path, so no model download or
inference run is required.

The following Bash example runs one warm-up and five measured trials for each
kernel/worker combination. It writes elapsed wall time, peak process RSS, and
an output hash to CSV. Set `MODEL` to the exact GGUF file.

```bash
set -euo pipefail

MODEL="${MODEL:?set MODEL to a GGUF path}"
BIN="${BIN:-target/release/qarnyx}"
PROMPT="${PROMPT:-hi there!}"
RESULTS="${RESULTS:-benchmark-results.csv}"
RUNS="${RUNS:-5}"
WARMUPS="${WARMUPS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-4096}"
MAX_CACHE_MIB="${MAX_CACHE_MIB:-512}"
MAX_MEMORY_MIB="${MAX_MEMORY_MIB:-8192}"

KERNELS=(auto fused-kq-fma)
WORKERS=(1 2 4 6 8 10)

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

printf 'kernel,workers,run,elapsed_seconds,max_rss_kib,output_sha256\n' > "$RESULTS"

for kernel in "${KERNELS[@]}"; do
  for workers in "${WORKERS[@]}"; do
    args=(
      generate "$MODEL" "$PROMPT"
      --max-new-tokens "$MAX_NEW_TOKENS"
      --temperature 0
      --top-k 0
      --top-p 1
      --workers "$workers"
      --cpu-kernel "$kernel"
      --max-context-tokens "$MAX_CONTEXT_TOKENS"
      --max-cache-mib "$MAX_CACHE_MIB"
      --max-memory-mib "$MAX_MEMORY_MIB"
    )

    for ((warmup = 1; warmup <= WARMUPS; warmup++)); do
      "$BIN" "${args[@]}" > /dev/null
    done

    for ((run = 1; run <= RUNS; run++)); do
      /usr/bin/time -f '%e,%M' -o "$tmpdir/time" \
        "$BIN" "${args[@]}" > "$tmpdir/output"
      IFS=, read -r elapsed_seconds max_rss_kib < "$tmpdir/time"
      output_sha256="$(sha256sum "$tmpdir/output" | cut -d ' ' -f 1)"
      printf '%s,%s,%s,%s,%s,%s\n' \
        "$kernel" "$workers" "$run" "$elapsed_seconds" \
        "$max_rss_kib" "$output_sha256" >> "$RESULTS"
    done
  done
done

echo "Wrote $RESULTS"
```

Keep the kernel matrix intentionally small: `auto` is the production/default
mode and `fused-kq-fma` is the explicit optimized path. Use the worker sweep
only when investigating CPU scaling; do not reintroduce retired kernel modes
just to enlarge the test matrix. Portable scalar/dequantization code remains
internal and is covered by unit tests rather than exposed as a CLI mode.

## Interpret the results

- Run the same matrix more than once on a quiet host. Report median and spread
  (minimum/maximum or interquartile range), not only the best run.
- Compare output hashes within each workload. With greedy sampling, a kernel
  change should not silently change generated output; investigate any mismatch.
- The time reported above is whole-process wall time. It includes model-file
  reading, GGUF/tokenizer setup, backend initialization, and generation. It
  does not isolate first-token latency or decode tokens/s.
- Peak RSS is the maximum resident size of the CLI process, reported in KiB by
  GNU `time`. The CLI's memory-admission limits are estimates/envelopes, not
  RSS caps; report observed RSS separately.
- For Qwen3.5 operator profiling, repeat a representative run with
  `QARNYX_PROFILE=1` and retain stderr output. Profiling adds instrumentation
  overhead; do not mix profiled timings with unprofiled wall-time trials.
- Report prefill/decode tokens per second only when measured by a harness that
  times those phases separately. Do not derive phase throughput from whole
  process wall time.
- Synthetic kernel benchmarks are useful for kernel comparisons but do not
  replace a representative GGUF end-to-end run.


## User-reported VPS Q4_K end-to-end comparison (2026-11-19)

A three-run comparison was reported from an Ubuntu VPS using the release CLI
and the following workload:

- Model: `Huihui-Qwen3.5-2B-abliterated.Q4_K_M.gguf`
- Prompt: `model apa kamu`
- Generated tokens: 32, greedy sampling (`temperature=0`, `top-k=0`,
  `top-p=1`)
- Workers: 10
- Kernel modes: `scalar`, `auto`, and `fused-q4k`
- Limits: `max-context-tokens=4096`, `max-cache-mib=512`,
  `max-memory-mib=8192`
- Trials: three per kernel, with no separate warm-up trial reported
- Measurement: GNU `/usr/bin/time` elapsed wall time (`%e`) and peak RSS
  (`%M`)

| Kernel | Run 1 (s) | Run 2 (s) | Run 3 (s) | Mean (s) | Median (s) |
|---|---:|---:|---:|---:|---:|
| scalar | 28.74 | 29.19 | 29.20 | 29.04 | 29.19 |
| auto | 18.73 | 19.30 | 19.36 | 19.13 | 19.30 |
| fused-q4k | 15.23 | 15.52 | 16.15 | 15.63 | 15.52 |

Using the three-run means, `auto` reduced end-to-end wall time by about
34.1% versus `scalar`. `fused-q4k` reduced it by about 46.2% versus
`scalar`, and by about 18.3% versus `auto`.

All nine runs produced the same output SHA-256:

```text
c5413d2a495ace3cbb7d89f3729c1a408cd77316add0fb50fa986736a559cb11
```

The highest observed peak RSS across these runs was 1,362,372 KiB (about
1.30 GiB). The observed peak RSS values were close across kernel modes.

**Interpretation:** these are whole-process timings, not isolated kernel
timings. They include model reading, GGUF/tokenizer setup, backend
initialization, and generation. The elapsed values were printed to 0.01-second
precision by GNU `time`. The benchmark log did not record the VPS CPU model,
available vCPUs, Rust/Cargo versions, exact source revision, or a separate
warm-up, so this is a useful host-specific preliminary result rather than a
fully specified cross-machine benchmark.

The `scalar` mode disables SIMD arithmetic, but it does **not** disable the
Rayon worker pool. The run configured 10 workers for all three modes; eligible
large matvec/matmul row work and projection batches can still use the shared
persistent pool. Small operations may remain sequential. Therefore this
comparison measures kernel modes under the same configured worker-pool policy;
it is not a comparison of single-thread scalar against multithreaded SIMD.



## User-reported VPS Rayon worker scaling (2026-11-19)

A clean 18-run matrix was reported from the Ubuntu VPS. The script validated
exactly 18 data rows, six kernel/worker groups, runs 1-3 in every group, and
one output hash shared by all runs.

Workload:

- Model: `Huihui-Qwen3.5-2B-abliterated.Q4_K_M.gguf`
- Prompt: `model apa kamu`
- Generated tokens: 32, greedy sampling (`temperature=0`, `top-k=0`,
  `top-p=1`)
- Kernel modes: `scalar`, `auto`, and `fused-q4k`
- Worker counts: 1 and 10
- Limits: `max-context-tokens=4096`, `max-cache-mib=512`,
  `max-memory-mib=8192`
- Trials: three per kernel/worker combination; no separate warm-up trial
  reported
- Measurement: GNU `/usr/bin/time` elapsed wall time (`%e`) and peak RSS
  (`%M`)

| Kernel | Workers | Run 1 (s) | Run 2 (s) | Run 3 (s) | Mean (s) | Median (s) |
|---|---:|---:|---:|---:|---:|---:|
| scalar | 1 | 200.47 | 201.76 | 209.48 | 203.90 | 201.76 |
| scalar | 10 | 35.64 | 35.47 | 34.70 | 35.27 | 35.47 |
| auto | 1 | 106.05 | 106.40 | 105.55 | 106.00 | 106.05 |
| auto | 10 | 25.48 | 26.83 | 28.68 | 27.00 | 26.83 |
| fused-q4k | 1 | 83.16 | 85.11 | 89.36 | 85.88 | 85.11 |
| fused-q4k | 10 | 21.24 | 20.71 | 19.71 | 20.55 | 20.71 |

Based on the three-run means, moving from 1 to 10 configured workers reduced
whole-process wall time by approximately:

- `scalar`: 5.78x speedup (82.7% less elapsed time)
- `auto`: 3.93x speedup (74.5% less elapsed time)
- `fused-q4k`: 4.18x speedup (76.1% less elapsed time)

All 18 runs produced the same output SHA-256:

```text
c5413d2a495ace3cbb7d89f3729c1a408cd77316add0fb50fa986736a559cb11
```

The highest observed peak RSS in this matrix was 1,363,780 KiB (about
1.30 GiB). RSS remained in a similar range across worker counts and kernels.

**Interpretation:** the result shows a substantial benefit from the 10-worker
configuration on this host for this workload. It is not a pure measurement of
Rayon overhead: each CLI process also reads the model, initializes the GGUF
and tokenizer, constructs the backend, and performs generation. The
`scalar` mode disables SIMD arithmetic, not the worker pool; with 10 workers,
eligible large row operations can still use Rayon. With one configured worker,
the backend owns a one-thread pool and its parallel helpers take sequential
paths where appropriate. Small operations can remain sequential at either
worker count.

The measured times differ from the earlier three-kernel, 10-worker comparison
in this document (notably for `auto` and `fused-q4k`). Treat these results as
a separate run rather than combining the two samples. Whole-process timings
can vary with VPS contention and scheduling. The VPS CPU model, available
vCPUs, Rust/Cargo versions, and exact source revision were not recorded, and
there was no separate warm-up; repeat on a quiet host before selecting a
permanent worker default. This matrix compares only 1 and 10 workers and does
not establish whether 2, 4, 6, or 8 workers would be optimal.


## Known benchmark limitations

The current CLI runs one blocking generation per process and prints the final
text after generation. It does not expose token streaming, time-to-first-token,
or independent prefill/decode timers. End-to-end CLI timing therefore remains
the reliable metric available from this command-line path until a dedicated
benchmark harness is added.


## User-reported VPS Q4_K fused-q4k worker scaling (2026-11-19)

A clean 18-run worker sweep was reported from the Ubuntu VPS. The script
validated exactly 18 data rows, six worker groups, runs 1-3 in each group,
and one output hash shared by all runs. Trials were executed round-robin by
worker count to reduce systematic time-of-run bias.

Workload:

- Model: `Huihui-Qwen3.5-2B-abliterated.Q4_K_M.gguf`
- Prompt: `model apa kamu`
- Generated tokens: 32, greedy sampling (`temperature=0`, `top-k=0`,
  `top-p=1`)
- Kernel: `fused-q4k`
- Worker counts: 1, 2, 4, 6, 8, and 10
- Limits: `max-context-tokens=4096`, `max-cache-mib=512`,
  `max-memory-mib=8192`
- Trials: three per worker count; no separate warm-up trial reported
- Measurement: GNU `/usr/bin/time` elapsed wall time (`%e`) and peak RSS
  (`%M`)

| Workers | Run 1 (s) | Run 2 (s) | Run 3 (s) | Mean (s) | Median (s) | Range (s) | Speedup vs 1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 90.25 | 84.46 | 80.36 | 85.02 | 84.46 | 80.36-90.25 | 1.00x |
| 2 | 58.59 | 49.82 | 46.72 | 51.71 | 49.82 | 46.72-58.59 | 1.64x |
| 4 | 36.67 | 35.25 | 31.53 | 34.48 | 35.25 | 31.53-36.67 | 2.47x |
| 6 | 29.18 | 28.10 | 27.70 | 28.33 | 28.10 | 27.70-29.18 | 3.00x |
| 8 | 27.07 | 24.54 | 25.58 | 25.73 | 25.58 | 24.54-27.07 | 3.30x |
| 10 | 23.73 | 23.02 | 23.45 | 23.40 | 23.45 | 23.02-23.73 | 3.63x |

All 18 runs produced the same output SHA-256:

```text
c5413d2a495ace3cbb7d89f3729c1a408cd77316add0fb50fa986736a559cb11
```

The highest observed peak RSS was 1,363,784 KiB (about 1.30 GiB); RSS
remained in a narrow range across worker counts.

**Interpretation:** the mean whole-process time decreased at every tested
worker-count increase. Moving from 1 to 10 workers improved the measured
time by 3.63x (72.5% less elapsed time). Moving from 8 to 10 workers still
reduced the mean by about 9.1%, so this sweep does not show a clear plateau.
Ten workers is the fastest configuration tested, not a proven global optimum;
counts above 10 were not tested. The result is specific to this host, model,
prompt, and kernel mode.

These are whole-process timings, not decode-only throughput. They include
model reading, GGUF/tokenizer setup, backend initialization, and generation.
No separate warm-up was reported. The script saved host/compiler details and
the source revision to
`benchmark-results/m7-q4k-workers-sweep.env` on the VPS; that environment
file should be retained alongside the CSV for reproducibility. The CPU model,
available vCPUs, Rust/Cargo versions, and exact revision are not reproduced
in the terminal output captured here.

This single-request worker sweep does not establish the best configuration
under concurrent serving load. M8 serving benchmarks must measure both
per-request latency and aggregate throughput under controlled concurrency.
