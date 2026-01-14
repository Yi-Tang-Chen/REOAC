#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_PATH="${1:-$ROOT_DIR/data/raw/input.json}"
OUTPUT_PATH="${2:-$ROOT_DIR/data/processed/processed.jsonl}"

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/third_party:${PYTHONPATH:-}"
python -m src.data.prep --input "$INPUT_PATH" --output "$OUTPUT_PATH"
