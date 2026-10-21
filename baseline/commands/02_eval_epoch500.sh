#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate faceflux

CKPT="$ROOT/checkpoints/repro_p05_full_exact/epoch_500.pth"
DATASET_DIR="$ROOT/datasets/FacescapeRender1"
OUTPUT_DIR="$ROOT/outputs/repro_from_scratch_epoch500_seed42"

if [ ! -f "$CKPT" ]; then
  echo "错误：尚未找到从头训练产生的 checkpoint："
  echo "$CKPT"
  echo
  echo "请先完成 commands/01_train_from_scratch.sh 启动的训练。"
  exit 1
fi

if [ ! -d "$DATASET_DIR" ]; then
  echo "错误：评估数据集不存在：$DATASET_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "正在评估从头训练得到的 epoch_500 checkpoint："
echo "$CKPT"
sha256sum "$CKPT"

CUDA_VISIBLE_DEVICES=0 python \
  scripts/eval/eval_facescape_modes.py \
  --model unet \
  --unet_size B \
  --ckpt "$CKPT" \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --eval_size 256 \
  --steps 1 \
  --inference_mode no_tta \
  --model_name "Repro from scratch p05 full epoch500 seed42"

echo
echo "评估完成，结果目录：$OUTPUT_DIR"
echo "当前终端不会自动关闭。"
