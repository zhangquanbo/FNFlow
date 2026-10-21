#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZOO="$ROOT/experiments/checkpoint_zoo"
DATASET="$ROOT/datasets/FacescapeRender1"
EVAL="$ROOT/scripts/eval/eval_facescape_modes.py"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="$ROOT/outputs/checkpoint_zoo_finals_$STAMP"
MANIFEST="$OUTPUT/checkpoint_manifest.tsv"

cd "$ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate faceflux

for path in "$ZOO" "$DATASET" "$EVAL"; do
  [ -e "$path" ] || {
    echo "错误：缺少必要路径：$path"
    exit 1
  }
done

CHECKPOINT_COUNT="$(
  find "$ZOO" \
    -mindepth 2 \
    -maxdepth 2 \
    -type f \
    -name checkpoint.pth |
  wc -l
)"

[ "$CHECKPOINT_COUNT" -eq 13 ] || {
  echo "错误：checkpoint_zoo 应有13个checkpoint，实际为：$CHECKPOINT_COUNT"
  exit 1
}

mkdir -p "$OUTPUT"

printf \
  'model\tfolder\tcheckpoint\ttraining_script\tconfig_file\tconfig\tmode\tsha256\n' \
  > "$MANIFEST"

run_eval() {
  local name="$1"
  local folder="$2"
  local config="$3"
  local mode="$4"

  local experiment_dir="$ZOO/$folder"
  local checkpoint="$experiment_dir/checkpoint.pth"
  local training_script="$experiment_dir/train.sh"
  local config_file="$experiment_dir/config.yaml"

  for path in "$checkpoint" "$training_script" "$config_file"; do
    [ -f "$path" ] || {
      echo "错误：实验文件不存在：$path"
      exit 1
    }
  done

  local hash
  hash="$(sha256sum "$checkpoint" | awk '{print $1}')"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" \
    "$folder" \
    "$checkpoint" \
    "$training_script" \
    "$config_file" \
    "$config" \
    "$mode" \
    "$hash" \
    >> "$MANIFEST"

  echo
  echo "============================================================"
  echo "评测：$name"
  echo "实验目录：$experiment_dir"
  echo "Checkpoint：$checkpoint"
  echo "训练脚本：$training_script"
  echo "配置文件：$config_file"
  echo "配置：$config"
  echo "模式：$mode"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU:-0}" \
  python -u "$EVAL" \
    --model unet \
    --unet_size B \
    --ckpt "$checkpoint" \
    --dataset_dir "$DATASET" \
    --output_dir "$OUTPUT" \
    --eval_size 256 \
    --steps 1 \
    --inference_mode "$mode" \
    --model_name "$name"
}

run_eval \
  "flow_only_p01" \
  "01_flow_only_p01" \
  "loss=l2_only,weight_cos=unused,drop_prob=0.1" \
  "no_tta"

run_eval \
  "no_angular_p01" \
  "02_no_angular_p01" \
  "loss=l2_norm,weight_cos=0.25,drop_prob=0.1" \
  "no_tta"

run_eval \
  "flow_angular_w01_p01" \
  "03_flow_angular_w01_p01" \
  "loss=l2_cos,weight_cos=0.1,drop_prob=0.1" \
  "no_tta"

run_eval \
  "flow_angular_w05_p01" \
  "04_flow_angular_w05_p01" \
  "loss=l2_cos,weight_cos=0.5,drop_prob=0.1" \
  "no_tta"

run_eval \
  "full_wunit_wang_01_p01" \
  "05_full_wunit_wang_01_p01" \
  "loss=full,weight_unit=0.1,weight_ang=0.1,drop_prob=0.1" \
  "no_tta"

run_eval \
  "full_wunit_wang_05_p01" \
  "06_full_wunit_wang_05_p01" \
  "loss=full,weight_unit=0.5,weight_ang=0.5,drop_prob=0.1" \
  "no_tta"

run_eval \
  "true_no_flow_p01" \
  "07_no_flow_p01" \
  "loss=no_flow,weight_cos=0.25,drop_prob=0.1" \
  "no_tta"

run_eval \
  "full_p00" \
  "08_full_p00" \
  "loss=full,weight_cos=0.25,drop_prob=0.0" \
  "no_cfg"

run_eval \
  "full_p03" \
  "09_full_p03" \
  "loss=full,weight_cos=0.25,drop_prob=0.3" \
  "no_tta"

run_eval \
  "full_p05_original" \
  "10_full_p05_original" \
  "loss=full,weight_cos=0.25,drop_prob=0.5" \
  "no_tta"

run_eval \
  "full_p07" \
  "11_full_p07" \
  "loss=full,weight_cos=0.25,drop_prob=0.7" \
  "no_tta"

run_eval \
  "repro_l2cos_p05" \
  "12_repro_l2cos_p05" \
  "loss=l2_cos,weight_cos=0.25,drop_prob=0.5" \
  "no_tta"

run_eval \
  "repro_full_p05_best" \
  "13_repro_full_p05_best" \
  "loss=full,weight_cos=0.25,drop_prob=0.5" \
  "no_tta"

OUTPUT_DIR="$OUTPUT" python - <<'PY'
import json
import os
from pathlib import Path

output = Path(os.environ["OUTPUT_DIR"])
json_path = output / "eval_results.json"
results = json.loads(json_path.read_text(encoding="utf-8"))

if len(results) != 13:
    raise RuntimeError(
        f"结果数量错误：{len(results)} != 13"
    )

for result in results:
    if result["num_images"] != 1998:
        raise RuntimeError(
            f"{result['model']} 样本数错误："
            f"{result['num_images']} != 1998"
        )

    fallbacks = result.get("face_mask_fallbacks")

    if fallbacks != 38:
        raise RuntimeError(
            f"{result['model']} mask fallback数量错误："
            f"{fallbacks} != 38"
        )

results.sort(key=lambda item: item["mean"])

summary = output / "summary_sorted.md"

with summary.open("w", encoding="utf-8") as f:
    f.write("# Checkpoint Zoo Final Results\n\n")
    f.write(
        "统一协议：FacescapeRender1、UNet-B、256px、"
        "1 step、seed=42；p0=no_cfg，其余=no_tta。\n\n"
    )
    f.write(
        "| Rank | Model | Mean ± Std | <20° | <25° | "
        "<30° | Mask fallback |\n"
    )
    f.write("|---:|---|---:|---:|---:|---:|---:|\n")

    for rank, result in enumerate(results, 1):
        f.write(
            f"| {rank} | {result['model']} | "
            f"{result['mean']:.4f} ± {result['std']:.4f} | "
            f"{result['lt_20']:.2f}% | "
            f"{result['lt_25']:.2f}% | "
            f"{result['lt_30']:.2f}% | "
            f"{result['face_mask_fallbacks']} |\n"
        )

print()
print(f"结果 JSON：{json_path}")
print(f"排序报告：{summary}")
print()
print(
    f"最佳结果：{results[0]['model']} "
    f"{results[0]['mean']:.6f}°"
)
PY

echo
echo "全部评测完成：$OUTPUT"
