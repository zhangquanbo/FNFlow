#!/usr/bin/env bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$ROOT/logs/repro_p05_full_exact.log"
PID_FILE="$ROOT/logs/repro_p05_full_exact.pid"

if [ -f "$PID_FILE" ]; then
  TRAIN_PID="$(cat "$PID_FILE")"
  echo "记录的训练 PID：$TRAIN_PID"

  if kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "训练进程当前正在运行"
  else
    echo "训练进程当前未运行，可能已经完成或异常退出"
  fi

  echo
fi

if [ ! -f "$LOG_FILE" ]; then
  echo "训练日志尚不存在：$LOG_FILE"
  echo "请先运行：bash commands/01_train_from_scratch.sh"
else
  echo "正在持续查看日志：$LOG_FILE"
  echo "按 Ctrl+C 只会停止查看，不会终止后台训练。"
  echo
  tail -f "$LOG_FILE"
fi
