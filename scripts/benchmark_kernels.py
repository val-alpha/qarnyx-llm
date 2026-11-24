#!/usr/bin/env python3
"""Repeatable Qarnyx end-to-end comparison for the two retained CPU modes.

This runner deliberately compares only auto and fused-kq-fma. It uses
greedy decoding, one warm-up per mode, and round-robin measured trials. Timings
include process startup, model loading, tokenizer setup, and generation; use
--profile for separate Qwen3.5 prefill/decode timings (instrumented).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

KERNELS = ("auto", "fused-kq-fma")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="GGUF model file")
    parser.add_argument("prompt", help="prompt text passed to Qarnyx")
    parser.add_argument(
        "--bin",
        type=Path,
        default=Path("target/release/qarnyx"),
        help="Qarnyx CLI binary (default: target/release/qarnyx)",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--max-cache-mib", type=int, default=512)
    parser.add_argument("--max-memory-mib", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("qarnyx-kernel-benchmark.csv"),
        help="output CSV path",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="enable QARNYX_PROFILE=1; timings are instrumented and not comparable "
        "to unprofiled end-to-end runs",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.max_context_tokens < 1:
        parser.error("--max-context-tokens must be at least 1")
    if args.max_cache_mib < 1 or args.max_memory_mib < 1:
        parser.error("memory limits must be at least 1 MiB")
    if args.warmups < 0:
        parser.error("--warmups cannot be negative")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def run_once(args: argparse.Namespace, kernel: str) -> tuple[float, str, str]:
    binary = args.bin.expanduser().resolve()
    model = args.model.expanduser().resolve()
    if not binary.is_file():
        raise RuntimeError(f"Qarnyx CLI binary not found: {binary}")
    if not model.is_file():
        raise RuntimeError(f"GGUF model file not found: {model}")

    command = [
        str(binary),
        "generate",
        str(model),
        args.prompt,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-context-tokens",
        str(args.max_context_tokens),
        "--max-cache-mib",
        str(args.max_cache_mib),
        "--max-memory-mib",
        str(args.max_memory_mib),
        "--temperature",
        "0",
        "--top-k",
        "0",
        "--top-p",
        "1",
        "--workers",
        str(args.workers),
        "--cpu-kernel",
        kernel,
        "--seed",
        str(args.seed),
    ]

    env = os.environ.copy()
    if args.profile:
        env["QARNYX_PROFILE"] = "1"
    else:
        env.pop("QARNYX_PROFILE", None)

    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=args.timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"{kernel} exceeded timeout of {args.timeout_seconds:.1f}s"
        ) from exc
    elapsed = time.perf_counter() - started

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{kernel} exited with status {result.returncode}:\n{stderr}"
        )

    output_hash = hashlib.sha256(result.stdout).hexdigest()
    stderr_text = result.stderr.decode("utf-8", errors="replace")
    if args.profile and stderr_text:
        print(f"\n--- {kernel} profile ---", file=sys.stderr)
        print(stderr_text, file=sys.stderr, end="" if stderr_text.endswith("\n") else "\n")

    return elapsed, output_hash, stderr_text


def main() -> int:
    args = parse_args()
    rows: list[dict[str, object]] = []
    output_hashes: dict[str, set[str]] = {kernel: set() for kernel in KERNELS}

    print(
        f"Model: {args.model}\nWorkers: {args.workers}\n"
        f"Tokens: {args.max_new_tokens}\nWarmups: {args.warmups}\n"
        f"Measured runs: {args.runs}\nProfiled: {args.profile}"
    )

    for kernel in KERNELS:
        for warmup in range(1, args.warmups + 1):
            elapsed, output_hash, _ = run_once(args, kernel)
            output_hashes[kernel].add(output_hash)
            print(f"[WARMUP] {kernel} {warmup}/{args.warmups}: {elapsed:.3f}s")

    # Alternate modes per trial to reduce systematic time-of-run bias.
    for run_index in range(1, args.runs + 1):
        for kernel in KERNELS:
            elapsed, output_hash, _ = run_once(args, kernel)
            output_hashes[kernel].add(output_hash)
            rows.append(
                {
                    "kernel": kernel,
                    "workers": args.workers,
                    "run": run_index,
                    "elapsed_seconds": f"{elapsed:.6f}",
                    "profiled": str(args.profile).lower(),
                    "output_sha256": output_hash,
                }
            )
            print(f"[RUN] {kernel} {run_index}/{args.runs}: {elapsed:.3f}s")

    all_hashes = set().union(*output_hashes.values())
    if len(all_hashes) != 1:
        details = ", ".join(
            f"{kernel}={sorted(hashes)}" for kernel, hashes in output_hashes.items()
        )
        raise RuntimeError(
            "greedy output hashes differ between trials or kernels; "
            f"investigate numerical/output parity before comparing speed: {details}"
        )

    csv_path = args.csv.expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "kernel",
                "workers",
                "run",
                "elapsed_seconds",
                "profiled",
                "output_sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    medians: dict[str, float] = {}
    print("\nSummary (whole-process wall time; includes model load and setup):")
    for kernel in KERNELS:
        timings = [
            float(row["elapsed_seconds"])
            for row in rows
            if row["kernel"] == kernel
        ]
        medians[kernel] = statistics.median(timings)
        print(
            f"  {kernel:<14} median={medians[kernel]:.3f}s "
            f"min={min(timings):.3f}s max={max(timings):.3f}s"
        )

    if medians["fused-kq-fma"] > 0:
        speedup = medians["auto"] / medians["fused-kq-fma"]
        print(f"  auto / fused-kq-fma median ratio: {speedup:.3f}x")
    print(f"  output SHA-256: {next(iter(all_hashes))}")
    print(f"  CSV: {csv_path}")
    if args.profile:
        print(
            "  NOTE: profiling was enabled; do not compare these wall times "
            "with unprofiled runs."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
