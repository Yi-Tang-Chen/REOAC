#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-gsm8k}"
CONFIG_PATH="${2:-$ROOT_DIR/configs/reoac_default.yaml}"
SPLIT="${3:-all}"

mkdir -p "$ROOT_DIR/data/raw" "$ROOT_DIR/data/processed" "$ROOT_DIR/runs"

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/third_party:${PYTHONPATH:-}"
python -m src.data.download --config "$CONFIG_PATH" --dataset "$DATASET" --split "$SPLIT" --save-raw
