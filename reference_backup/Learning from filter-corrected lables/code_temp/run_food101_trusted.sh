#!/usr/bin/env bash
set -euo pipefail

# 可选：如果你有特定 conda 环境，取消下一行注释并改成你的环境名
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate <your-env>

# 默认参数（可用环境变量覆盖）
CACHE="${CACHE:-./cache/food101/clip_ViTL14.pt}"
OBS="${OBS:-./cache/food101/food101_obs_labels.pt}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CKPT="${CKPT:-./outputs/food101/head_trusted.pt}"

# 保障目录存在
mkdir -p "$(dirname "${CKPT}")"

# 基础检查
[[ -f "${CACHE}" ]] || { echo "Missing cache: ${CACHE}"; exit 1; }
[[ -f "${OBS}" ]]   || { echo "Missing obs labels: ${OBS}"; exit 1; }

# 执行训练；"$@" 可承接临时追加的其他参数
python train_trusted.py \
  --cache "${CACHE}" \
  --obs_labels_path "${OBS}" \
  --batch_size "${BATCH_SIZE}" \
  --save "${CKPT}" \
  "$@"
