#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SKIP_BUILD=0
ARGS=()

for arg in "$@"; do
    case "$arg" in
        --skip-build)
            SKIP_BUILD=1
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
done

cd "$REPO_ROOT"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
    echo "[BUILD] cargo build --release -p qarnyx-api"
    cargo build --release -p qarnyx-api
    echo
fi

exec python3 "$REPO_ROOT/scripts/api_test.py" "${ARGS[@]}"
