#!/usr/bin/env python3
"""Smoke-test the intentionally small Qarnyx CPU kernel CLI surface.

The public CLI must expose exactly two selectable paths:
  - auto
  - fused-kq-fma

Portable scalar/dequantization implementations are internal fallbacks and
correctness references; they are not selectable CLI kernel modes.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

EXPECTED_KERNELS = {"auto", "fused-kq-fma"}
RETIRED_KERNELS = ("fused-q4k", "llama-q4k", "fused-qk", "scalar")


def run(binary: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [str(binary), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"CLI command timed out: {args!r}") from exc
    except OSError as exc:
        raise RuntimeError(f"cannot execute Qarnyx CLI at {binary}: {exc}") from exc


def advertised_kernel_values(help_text: str) -> set[str]:
    match = re.search(
        r"\[possible values:\s*([^\]]+)\]",
        help_text,
        flags=re.IGNORECASE,
    )
    if not match:
        raise AssertionError(
            "could not find Clap's '[possible values: ...]' list in generate help"
        )
    return {
        value.strip().strip("`'") for value in match.group(1).split(",") if value.strip()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bin",
        type=Path,
        default=Path("target/release/qarnyx"),
        help="path to the built Qarnyx CLI binary (default: target/release/qarnyx)",
    )
    args = parser.parse_args()
    binary = args.bin.expanduser()
    if not binary.is_file():
        parser.error(f"Qarnyx CLI binary not found: {binary}")

    help_result = run(binary, "generate", "--help")
    help_text = help_result.stdout + help_result.stderr
    if help_result.returncode != 0:
        raise AssertionError(
            f"'generate --help' failed ({help_result.returncode}):\n{help_text}"
        )

    advertised = advertised_kernel_values(help_text)
    if advertised != EXPECTED_KERNELS:
        raise AssertionError(
            "CLI kernel choices changed: "
            f"expected {sorted(EXPECTED_KERNELS)}, got {sorted(advertised)}"
        )
    print("[PASS] generate help advertises only auto and fused-kq-fma")

    # A missing model makes valid modes stop at file loading. That proves Clap
    # accepted the mode without requiring a real GGUF or running inference.
    with tempfile.TemporaryDirectory(prefix="qarnyx-kernel-cli-") as tmp:
        missing_model = str(Path(tmp) / "intentionally-missing.gguf")
        for mode in sorted(EXPECTED_KERNELS):
            result = run(
                binary,
                "generate",
                missing_model,
                "kernel mode smoke test",
                "--cpu-kernel",
                mode,
            )
            output = result.stdout + result.stderr
            if result.returncode == 0:
                raise AssertionError(
                    f"mode {mode!r} unexpectedly succeeded with a missing model"
                )
            if re.search(r"invalid value|possible values|unexpected value", output, re.I):
                raise AssertionError(
                    f"valid mode {mode!r} was rejected by argument parsing:\n{output}"
                )
            print(f"[PASS] --cpu-kernel {mode} is accepted by the CLI parser")

        for mode in RETIRED_KERNELS:
            result = run(
                binary,
                "generate",
                missing_model,
                "kernel mode smoke test",
                "--cpu-kernel",
                mode,
            )
            output = result.stdout + result.stderr
            if result.returncode == 0 or not re.search(
                r"invalid value|possible values|unexpected value", output, re.I
            ):
                raise AssertionError(
                    f"retired mode {mode!r} was not rejected during argument parsing:\n"
                    f"exit={result.returncode}\n{output}"
                )
            print(f"[PASS] retired mode {mode} is rejected")

    print("CPU kernel CLI smoke tests: 7 passed, 0 failed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
