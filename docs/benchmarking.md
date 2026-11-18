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
```

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

KERNELS=(auto fused-q4k fused-qk scalar)
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

Adjust `KERNELS` and `WORKERS` to the hardware under test. If a format is
not supported by the selected model, omit kernel modes that do not apply.
The `scalar` mode is a correctness/performance reference, not necessarily a
production recommendation.

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


## Known benchmark limitations

The current CLI runs one blocking generation per process and prints the final
text after generation. It does not expose token streaming, time-to-first-token,
or independent prefill/decode timers. End-to-end CLI timing therefore remains
the reliable metric available from this command-line path until a dedicated
benchmark harness is added.
