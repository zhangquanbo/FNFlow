#!/usr/bin/env bash
set -euo pipefail

BEST="${FACEFLUX_BEST_ROOT}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_CODE="$DIR/training_code/train_unet.py"
OUTPUT_DIR="$DIR/retrained"
FINAL_CKPT="$OUTPUT_DIR/epoch_500.pth"

cd "$BEST"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate faceflux

for path in   "$BEST/datasets/ffhq_rgb_crop256"   "$BEST/datasets/ffhq_rgb_crop256N"   "$BEST/data/test_images_1/imgs"   "$TRAIN_CODE"
do
  [ -e "$path" ] || {
    echo "错误：缺少必要路径：$path"
    exit 1
  }
done

if [ -f "$FINAL_CKPT" ]; then
  echo "错误：重训结果已经存在：$FINAL_CKPT"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

PYTHONPATH="$DIR/training_code:$BEST" CUDA_VISIBLE_DEVICES=0 python -u "$TRAIN_CODE"   --model_size B   --rgb_dir "$BEST/datasets/ffhq_rgb_crop256"   --normal_dir "$BEST/datasets/ffhq_rgb_crop256N"   --eval_img_dir "$BEST/data/test_images_1/imgs"   --output_dir "$OUTPUT_DIR"   --loss "l2_cos"   --weight_cos "0.5"   --drop_prob "0.1"   --image_size 256   --batch_size 8   --grad_accum_steps 2   --lr 1e-4   --epochs 500   --start_save_epoch 200   --save_interval 10   --eval_steps "20"

[ -f "$FINAL_CKPT" ] || {
  echo "错误：训练结束但没有找到 $FINAL_CKPT"
  exit 1
}

find "$OUTPUT_DIR"   -maxdepth 1   -type f   -name 'epoch_*.pth'   ! -name 'epoch_500.pth'   -print   -delete

rm -f -- "$OUTPUT_DIR/latest.pth"
rm -rf -- "$OUTPUT_DIR/eval_output_dir"

sha256sum "$FINAL_CKPT" > "$OUTPUT_DIR/epoch_500.sha256"

echo "训练完成：$FINAL_CKPT"
