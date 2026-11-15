#!/usr/bin/env bash
set -euo pipefail

# 可选：如果你有特定 conda 环境，取消下一行注释并改成你的环境名
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate <your-env>

# 默认参数（可用环境变量覆盖）
CACHE="${CACHE:-./cache/food101/clip_ViTL14.pt}"
CKPT="${CKPT:-./outputs/food101/head_trusted.pt}"

# 保障目录存在
mkdir -p "$(dirname "${CKPT}")"

# 基础检查
[[ -f "${CACHE}" ]] || { echo "Missing cache: ${CACHE}"; exit 1; }

# 执行训练；"$@" 可承接临时追加的其他参数
python ./utils/eval_acc.py \
  --cache "${CACHE}" \
  --ckpt "${CKPT}" \
  "$@"
