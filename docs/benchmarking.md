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

## Known benchmark limitations

The current CLI runs one blocking generation per process and prints the final
text after generation. It does not expose token streaming, time-to-first-token,
or independent prefill/decode timers. End-to-end CLI timing therefore remains
the reliable metric available from this command-line path until a dedicated
benchmark harness is added.
