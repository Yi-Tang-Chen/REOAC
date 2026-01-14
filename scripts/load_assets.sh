#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/mdlm_small_math.yaml}"
TRAIN_COUNT="${2:-32}"
EVAL_COUNT="${3:-16}"

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/third_party:${PYTHONPATH:-}"
python -m src.data.asset_loader --config "$CONFIG_PATH" --train-count "$TRAIN_COUNT" --eval-count "$EVAL_COUNT"
