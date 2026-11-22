#!/usr/bin/env python3
"""Opt-in concurrent load test for a running Qarnyx API server.

This program never starts, stops, or restarts the server. It fans out real
inference requests, records HTTP status/latency, and checks that the server
remains healthy after the load burst.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class Probe:
    status: int | None
    elapsed_ms: float
    retry_after: str | None = None
    error: str = ""


def request_json(
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 600.0,
) -> tuple[int, dict[str, str], str]:
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return (
                response.status,
                {key.lower(): value for key, value in response.headers.items()},
                body,
            )
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        return (
            error.code,
            {key.lower(): value for key, value in error.headers.items()}
            if error.headers
            else {},
            body,
        )


def decode_object(body: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(body)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} did not return JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} response must be a JSON object")
    return value


def make_payload(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    prompt = (
        "This is a Qarnyx concurrent API load test. "
        "Reply with one brief sentence confirming you are ready."
    )
    if args.endpoint == "openai":
        return "/v1/chat/completions", {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": args.max_tokens,
            "temperature": 0,
        }
    if args.endpoint == "anthropic":
        return "/v1/messages", {
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
    return "/v1/generate", {
        "model": args.model,
        "prompt": prompt,
        "config": {
            "max_new_tokens": args.max_tokens,
            "temperature": 0,
            "top_k": 20,
            "top_p": 1,
            "seed": 7,
        },
    }


def run_probe(args: argparse.Namespace, path: str, payload: dict[str, Any]) -> Probe:
    started = time.monotonic()
    try:
        status, headers, body = request_json(
            args.base_url, path, payload, args.timeout
        )
        elapsed = (time.monotonic() - started) * 1000
        if status not in (200, 429):
            return Probe(
                status=status,
                elapsed_ms=elapsed,
                retry_after=headers.get("retry-after"),
                error=body[:500],
            )
        if status == 200:
            try:
                decode_object(body, path)
            except ValueError as error:
                return Probe(status, elapsed, error=str(error))
        return Probe(status, elapsed, retry_after=headers.get("retry-after"))
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return Probe(
            status=None,
            elapsed_ms=(time.monotonic() - started) * 1000,
            error=str(error),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send concurrent inference requests to an already-running Qarnyx server."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Qarnyx server URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--model",
        default="qarnyx",
        help="model request name or alias (default: qarnyx)",
    )
    parser.add_argument(
        "--endpoint",
        choices=("openai", "anthropic", "native"),
        default="openai",
        help="API surface to load-test (default: openai)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=8,
        help="total inference requests in one burst (default: 8)",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=8,
        help="maximum client-side concurrent requests (default: 8)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="generation limit per request (default: 32)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="per-request timeout in seconds (default: 600)",
    )
    args = parser.parse_args()
    if args.requests < 1 or args.parallelism < 1 or args.max_tokens < 1:
        parser.error("--requests, --parallelism, and --max-tokens must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    return args


def check_server(args: argparse.Namespace) -> tuple[bool, dict[str, Any] | None]:
    try:
        status, _, body = request_json(
            args.base_url, "/ready", timeout=min(args.timeout, 5.0)
        )
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        print(f"[SKIP] Qarnyx server is not reachable at {args.base_url}")
        print(f"       {error}")
        print("       This program never starts or stops the server.")
        return False, None

    if status != 200:
        print(f"[SKIP] {args.base_url}/ready returned HTTP {status}; no load was sent.")
        return False, None
    try:
        ready = decode_object(body, "/ready")
    except ValueError as error:
        print(f"[SKIP] {error}; no load was sent.")
        return False, None
    if ready.get("status") != "ok":
        print("[SKIP] Qarnyx server is not ready; no load was sent.")
        return False, None
    return True, ready


def main() -> int:
    args = parse_args()
    is_ready, ready = check_server(args)
    if not is_ready or ready is None:
        return 0

    path, payload = make_payload(args)
    workers = min(args.requests, args.parallelism)
    configured_slots = ready.get("max_concurrent_requests", "unknown")
    print(
        f"[INFO] endpoint={args.endpoint} requests={args.requests} "
        f"client_workers={workers} server_slots={configured_slots}"
    )

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(run_probe, args, path, payload)
            for _ in range(args.requests)
        ]
        probes = [future.result() for future in concurrent.futures.as_completed(futures)]
    elapsed = time.monotonic() - started

    latencies = [probe.elapsed_ms for probe in probes]
    for status in sorted({probe.status for probe in probes}, key=lambda x: (-1 if x is None else x)):
        count = sum(1 for probe in probes if probe.status == status)
        label = "network error" if status is None else f"HTTP {status}"
        print(f"[RESULT] {label}: {count}")

    if latencies:
        print(
            "[LATENCY] "
            f"median={statistics.median(latencies):.1f}ms "
            f"mean={statistics.mean(latencies):.1f}ms "
            f"max={max(latencies):.1f}ms"
        )
    retry_hints = sorted(
        {probe.retry_after for probe in probes if probe.retry_after is not None}
    )
    if retry_hints:
        print(f"[INFO] Retry-After values observed: {', '.join(retry_hints)}")

    errors = [
        probe for probe in probes if probe.status not in (200, 429) or probe.error
    ]
    for probe in errors[:5]:
        print(
            f"[FAIL] status={probe.status} latency={probe.elapsed_ms:.1f}ms "
            f"detail={probe.error or 'unexpected HTTP status'}"
        )

    try:
        health_status, _, health_body = request_json(
            args.base_url, "/health", timeout=min(args.timeout, 5.0)
        )
        health = decode_object(health_body, "/health")
        ready_status, _, ready_body = request_json(
            args.base_url, "/ready", timeout=min(args.timeout, 5.0)
        )
        ready_after = decode_object(ready_body, "/ready")
        print(
            f"[HEALTH] HTTP {health_status}; "
            f"active_requests={health.get('active_requests', 'unknown')}; "
            f"ready HTTP {ready_status}"
        )
        if (
            health_status != 200
            or health.get("status") != "ok"
            or ready_status != 200
            or ready_after.get("status") != "ok"
            or health.get("active_requests") != 0
        ):
            errors.append(
                Probe(
                    status=health_status,
                    elapsed_ms=0,
                    error="server did not return to healthy/idle state after the load burst",
                )
            )
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        errors.append(Probe(None, 0, error=f"post-load health check failed: {error}"))

    print(
        f"[SUMMARY] {len(probes) - len(errors)} accepted/rejected as expected, "
        f"{len(errors)} failures, wall={elapsed:.2f}s"
    )
    if errors:
        return 1
    print("[PASS] server stayed healthy; HTTP 429 responses are expected overload rejection.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
