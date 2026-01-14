#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/mdlm_small_math.yaml}"
OUTPUT_PATH="${2:-$ROOT_DIR/runs/eval_math.json}"
CHECKPOINT_DIR="${3:-}"

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/third_party:${PYTHONPATH:-}"
bash "$ROOT_DIR/scripts/load_assets.sh" "$CONFIG_PATH"
if [[ -n "$CHECKPOINT_DIR" ]]; then
  python -m src.eval.metrics --config "$CONFIG_PATH" --task math --output "$OUTPUT_PATH" --checkpoint-dir "$CHECKPOINT_DIR"
else
  python -m src.eval.metrics --config "$CONFIG_PATH" --task math --output "$OUTPUT_PATH"
fi
