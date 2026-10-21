#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER="$ROOT/commands/01_train_worker.sh"
LOG_FILE="$ROOT/logs/repro_p05_full_exact.log"
PID_FILE="$ROOT/logs/repro_p05_full_exact.pid"
OUTPUT_DIR="$ROOT/checkpoints/repro_p05_full_exact"
FINAL_CKPT="$OUTPUT_DIR/epoch_500.pth"

mkdir -p "$ROOT/logs"

if [ -f "$PID_FILE" ]; then
  EXISTING_PID="$(cat "$PID_FILE")"

  if [[ "$EXISTING_PID" =~ ^[0-9]+$ ]] &&      kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "错误：训练进程已经在运行，PID=$EXISTING_PID"
    echo "查看进度：bash commands/04_follow_training_log.sh"
    exit 1
  fi
fi

if [ "${FORCE_RETRAIN:-0}" != "1" ]; then
  if [ -f "$FINAL_CKPT" ]; then
    echo "错误：最终 checkpoint 已存在：$FINAL_CKPT"
    echo "如需重新训练，请显式运行："
    echo "FORCE_RETRAIN=1 bash commands/01_train_from_scratch.sh"
    exit 1
  fi

  if compgen -G "$OUTPUT_DIR/epoch_*.pth" > /dev/null; then
    echo "错误：训练目录中存在历史 epoch checkpoint："
    echo "$OUTPUT_DIR"
    echo "请先检查历史文件，或显式设置 FORCE_RETRAIN=1。"
    exit 1
  fi
fi

nohup bash "$WORKER" > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"

echo "训练已在后台启动"
echo "PID=$TRAIN_PID"
echo "日志：$LOG_FILE"
echo
echo "训练成功后将自动删除中间 epoch checkpoint，"
echo "只保留 epoch_500.pth。"
echo
echo "查看训练进度："
echo "bash \"$ROOT/commands/04_follow_training_log.sh\""
echo
echo "当前终端不会自动关闭。"
