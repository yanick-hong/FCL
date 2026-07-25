#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=()
[[ -n "${CACHE:-}" ]] && ARGS+=(--cache "$CACHE")
[[ -n "${OBS:-}" ]] && ARGS+=(--obs_labels_path "$OBS")
[[ -n "${VAL_DIR:-}" ]] && ARGS+=(--val_dir "$VAL_DIR")
[[ -n "${CKPT:-}" ]] && ARGS+=(--best_ckpt "$CKPT")
[[ -n "${P:-}" ]] && ARGS+=(--P "$P")
[[ -n "${K:-}" ]] && ARGS+=(--K "$K")

python "${ROOT}/src/fcl/train_auc_ce.py" "${ARGS[@]}" "$@"
