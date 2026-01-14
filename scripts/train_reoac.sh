#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/mdlm_small_math.yaml}"
FINETUNE_MODE="${2:-lora}"

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/third_party:${PYTHONPATH:-}"
bash "$ROOT_DIR/scripts/load_assets.sh" "$CONFIG_PATH"
python -m src.rl.trainer --config "$CONFIG_PATH" --finetune-mode "$FINETUNE_MODE"
