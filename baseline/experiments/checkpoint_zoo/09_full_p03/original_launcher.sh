#!/bin/bash
# ==============================================================
# 窗口2 — Fig2 Loss消融（full组）+ Ours + Fig3 CFG消融（共5个）
#   [1] λ_cos=0.1 (full) → fig2/output/cos01_full   ← 公平对比 Ours
#   [2] λ_cos=0.5 (full) → fig2/output/cos05_full   ← 公平对比 Ours
#   [3] Ours (full,λ=0.25) → checkpoints/unet_base_new
#   [4] p=0 (CFG消融)      → fig3/output/p0_train
#   [5] p=0.3 (CFG消融)    → fig3/output/p03_train
# ==============================================================

set -e
cd $WORKSPACE_ROOT/FaceFlux-Clean

BASE="--model_size B --drop_prob 0.1 --epochs 500 --start_save_epoch 200 --save_interval 10 --image_size 256 --batch_size 8 --grad_accum_steps 2"

echo "====== 窗口2：Fig2 full组 + Ours + Fig3 CFG消融 ======"

# echo "[1/5] Ours (full, λ=0.25)..."
# python experiments/unet/train_unet.py $BASE \
#     --loss full --weight_cos 0.25 \
#     --output_dir checkpoints/unet_base_new

# echo "[2/5] λ_cos=0.1 (full)..."
# python experiments/unet/train_unet.py $BASE \
#     --loss full --weight_cos 0.1 \
#     --output_dir experiments/fig2/output/cos01_full

# echo "[3/5] λ_cos=0.5 (full)..."
# python experiments/unet/train_unet.py $BASE \
#     --loss full --weight_cos 0.5 \
#     --output_dir experiments/fig2/output/cos05_full

# echo "[4/5] Fig3 CFG消融: p=0 (无dropout，推理用 --no_cfg)..."
# python experiments/unet/train_unet.py $BASE \
#     --drop_prob 0.0 \
#     --loss full --weight_cos 0.25 \
#     --output_dir experiments/fig3/output/p0_train

# echo "[5/5] Fig3 CFG消融: p=0.3..."
# python experiments/unet/train_unet.py $BASE \
#     --drop_prob 0.3 \
#     --loss full --weight_cos 0.25 \
#     --output_dir experiments/fig3/output/p03_train

python experiments/unet/train_unet.py $BASE \
    --drop_prob 0.5 \
    --loss full --weight_cos 0.25 \
    --output_dir experiments/fig3/output/p05_train

echo ""
echo "✅ 窗口2 完成！"