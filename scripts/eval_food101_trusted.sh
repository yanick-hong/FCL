#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=()
[[ -n "${CACHE:-}" ]] && ARGS+=(--cache "$CACHE")
[[ -n "${CKPT:-}" ]] && ARGS+=(--ckpt "$CKPT")

python "${ROOT}/src/utils/eval/eval_acc.py" "${ARGS[@]}" "$@"
