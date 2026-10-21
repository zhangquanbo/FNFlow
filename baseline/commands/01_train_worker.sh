#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate faceflux

RGB_DIR="$ROOT/datasets/ffhq_rgb_crop256"
NORMAL_DIR="$ROOT/datasets/ffhq_rgb_crop256N"
OUTPUT_DIR="$ROOT/checkpoints/repro_p05_full_exact"
FINAL_CKPT="$OUTPUT_DIR/epoch_500.pth"

for path in "$RGB_DIR" "$NORMAL_DIR"; do
  if [ ! -e "$path" ]; then
    echo "错误：缺少必要路径：$path"
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR"

echo "开始从头训练……"

if CUDA_VISIBLE_DEVICES=0 python -u \
  train_unet.py \
  --model_size B \
  --rgb_dir "$RGB_DIR" \
  --normal_dir "$NORMAL_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --loss full \
  --drop_prob 0.5 \
  --weight_cos 0.25 \
  --image_size 256 \
  --batch_size 8 \
  --grad_accum_steps 2 \
  --lr 1e-4 \
  --epochs 500 \
  --start_save_epoch 200 \
  --save_interval 10 \
  --eval_steps 1
then
  echo
  echo "训练进程正常完成。"

  if [ -f "$FINAL_CKPT" ]; then
    echo "最终 checkpoint 已确认：$FINAL_CKPT"
    echo "正在删除中间 checkpoint……"

    find "$OUTPUT_DIR" \
      -maxdepth 1 \
      -type f \
      -name 'epoch_*.pth' \
      ! -name 'epoch_500.pth' \
      -print \
      -delete

    rm -f -- "$OUTPUT_DIR/latest.pth"
    rm -rf -- "$OUTPUT_DIR/eval_output_dir"

    echo "中间 checkpoint、latest.pth 和训练评估输出清理完成。"
    echo "保留：$FINAL_CKPT"
    sha256sum "$FINAL_CKPT"
  else
    echo "错误：训练结束，但未找到 epoch_500.pth。"
    echo "为安全起见，没有删除任何中间 checkpoint。"
    exit 1
  fi
else
  TRAIN_STATUS=$?
  echo
  echo "训练异常退出，状态码：$TRAIN_STATUS"
  echo "为便于排查和恢复，没有删除中间 checkpoint。"
  exit "$TRAIN_STATUS"
fi
