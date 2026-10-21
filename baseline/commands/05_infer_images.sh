#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate faceflux

INPUT_DIR="${1:-$ROOT/data/test_images_1/imgs}"
OUTPUT_DIR="${2:-$ROOT/outputs/general_inference}"
CKPT="${3:-$ROOT/checkpoints/repro_p05_full_exact/epoch_500.pth}"

if [ ! -d "$INPUT_DIR" ]; then
  echo "错误：输入图片目录不存在：$INPUT_DIR"
  echo
  echo "用法："
  echo "bash commands/05_infer_images.sh [输入目录] [输出目录] [checkpoint]"
  exit 1
fi

if [ ! -f "$CKPT" ]; then
  echo "错误：checkpoint 不存在：$CKPT"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0 python   scripts/inference/inference_unet.py   --ckpt "$CKPT"   --input "$INPUT_DIR"   --output_dir "$OUTPUT_DIR"   --model_size B   --image_size 256   --steps 1   --inference_mode no_tta

echo
echo "通用推理完成，结果目录：$OUTPUT_DIR"
