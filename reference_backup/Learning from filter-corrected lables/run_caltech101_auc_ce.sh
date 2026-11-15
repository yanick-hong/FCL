#!/usr/bin/env bash
set -euo pipefail

# 可选：如果你有特定 conda 环境，取消下一行注释并改成你的环境名
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate <your-env>

# 默认参数（可用环境变量覆盖）
CACHE="${CACHE:-./cache/caltech101/resnet.pt}"
OBS="${OBS:-./cache/caltech101/resnet_caltech101_obs_labels.pt}"
P="${P:-8}"
K="${K:-8}"
VAL_DIR="${VAL_DIR:-./val/caltech101/resnet_auc_ce}"
CKPT="${CKPT:-./outputs/caltech101/resnet_auc_ce.pt}"

# 保障目录存在
mkdir -p "${VAL_DIR}" "$(dirname "${CKPT}")"

# 基础检查
[[ -f "${CACHE}" ]] || { echo "Missing cache: ${CACHE}"; exit 1; }
[[ -f "${OBS}" ]]   || { echo "Missing obs labels: ${OBS}"; exit 1; }

# 执行训练；"$@" 可承接临时追加的其他参数
python train_auc_ce.py \
  --cache "${CACHE}" \
  --obs_labels_path "${OBS}" \
  --P "${P}" --K "${K}" \
  --val_dir "${VAL_DIR}" \
  --best_ckpt "${CKPT}" \
  "$@"
