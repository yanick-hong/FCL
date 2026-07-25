#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=()
[[ -n "${CACHE:-}" ]] && ARGS+=(--cache "$CACHE")
[[ -n "${OBS:-}" ]] && ARGS+=(--obs_labels_path "$OBS")
[[ -n "${CKPT:-}" ]] && ARGS+=(--save "$CKPT")
[[ -n "${BATCH_SIZE:-}" ]] && ARGS+=(--batch_size "$BATCH_SIZE")

python "${ROOT}/src/contrast/train_trusted.py" "${ARGS[@]}" "$@"
