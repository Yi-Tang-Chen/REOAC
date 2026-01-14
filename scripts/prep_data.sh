#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-gsm8k}"
CONFIG_PATH="${2:-$ROOT_DIR/configs/reoac_default.yaml}"
SPLIT="${3:-all}"
WORK_DIR="${4:-}"

resolve_work_dir() {
  local arg="${1:-}"
  if [[ -z "$arg" ]]; then
    echo ""
    return
  fi
  if [[ "$arg" == "server" ]]; then
    echo "${REOAC_WORK_ROOT:-/work/$USER}"
    return
  fi
  echo "$arg"
}

WORK_DIR="$(resolve_work_dir "$WORK_DIR")"
DATA_ROOT="$ROOT_DIR"
if [[ -n "$WORK_DIR" ]]; then
  DATA_ROOT="$WORK_DIR"
fi

mkdir -p "$DATA_ROOT/dataset/raw" "$DATA_ROOT/dataset/processed" "$DATA_ROOT/runs"

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/third_party:${PYTHONPATH:-}"
export REOAC_DATA_ROOT="$DATA_ROOT"
export REOAC_RUNS_ROOT="$DATA_ROOT/runs"
python -m src.data.download --config "$CONFIG_PATH" --dataset "$DATASET" --split "$SPLIT" --save-raw
