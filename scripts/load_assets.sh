#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/mdlm_small_math.yaml}"
TRAIN_COUNT="${2:-32}"
EVAL_COUNT="${3:-16}"
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

if [[ -n "$WORK_DIR" ]]; then
  export REOAC_DATA_ROOT="$WORK_DIR"
  export REOAC_RUNS_ROOT="$WORK_DIR/runs"
  mkdir -p "$REOAC_DATA_ROOT/dataset/raw" "$REOAC_DATA_ROOT/dataset/processed" "$REOAC_RUNS_ROOT"
fi

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/third_party:${PYTHONPATH:-}"
python -m src.data.asset_loader --config "$CONFIG_PATH" --train-count "$TRAIN_COUNT" --eval-count "$EVAL_COUNT"
