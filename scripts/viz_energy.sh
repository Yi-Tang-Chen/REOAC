#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_PATH="${1:-$ROOT_DIR/runs/sample_relation.json}"
OUTPUT_DIR="${2:-$ROOT_DIR/runs/viz}"

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/third_party:${PYTHONPATH:-}"
python -m src.viz.energy_maps --input "$INPUT_PATH" --output-dir "$OUTPUT_DIR"
