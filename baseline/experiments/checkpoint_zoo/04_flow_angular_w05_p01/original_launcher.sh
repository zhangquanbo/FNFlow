#!/bin/bash
# ==============================================================
# 窗口1 — Fig2 Loss消融（共4个变体）
#   [1] w/o L_flow  : l2_only  → fig2/output/wof_flow
#   [2] w/o L_ang   : l2_norm  → fig2/output/wo_ang
#   [3] λ_cos=0.1   : l2_cos   → fig2/output/cos01
#   [4] λ_cos=0.5   : l2_cos   → fig2/output/cos05
# ==============================================================

set -e
cd $WORKSPACE_ROOT/FaceFlux-Clean

BASE="--model_size B --drop_prob 0.1 --epochs 500 --start_save_epoch 200 --save_interval 10 --image_size 256 --batch_size 8 --grad_accum_steps 2"

echo "====== 窗口1：Fig2 Loss消融（非-full组） ======"

echo "[1/4] w/o L_flow (l2_only)..."
python experiments/unet/train_unet.py $BASE \
    --loss l2_only \
    --output_dir experiments/fig2/output/wof_flow

echo "[2/4] w/o L_ang (l2_norm)..."
python experiments/unet/train_unet.py $BASE \
    --loss l2_norm --weight_cos 0.25 \
    --output_dir experiments/fig2/output/wo_ang

echo "[3/4] λ_cos=0.1 (l2_cos)..."
python experiments/unet/train_unet.py $BASE \
    --loss l2_cos --weight_cos 0.1 \
    --output_dir experiments/fig2/output/cos01

echo "[4/4] λ_cos=0.5 (l2_cos)..."
python experiments/unet/train_unet.py $BASE \
    --loss l2_cos --weight_cos 0.5 \
    --output_dir experiments/fig2/output/cos05

echo ""
echo "✅ 窗口1 完成！"
