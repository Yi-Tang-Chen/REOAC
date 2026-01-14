#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/reoac_default.yaml}"
ARG2="${2:-}"
ARG3="${3:-}"
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

RUNS_ROOT="${REOAC_RUNS_ROOT:-$ROOT_DIR/runs}"
if [[ -n "$WORK_DIR" ]]; then
  export REOAC_DATA_ROOT="$WORK_DIR"
  export REOAC_RUNS_ROOT="$WORK_DIR/runs"
  RUNS_ROOT="$REOAC_RUNS_ROOT"
  mkdir -p "$REOAC_DATA_ROOT/dataset/raw" "$REOAC_DATA_ROOT/dataset/processed" "$REOAC_RUNS_ROOT"
fi

OUTPUT_DEFAULT="$RUNS_ROOT/eval_gsm8k.json"

if [[ -n "$ARG2" && -d "$ARG2" ]]; then
  CHECKPOINT_DIR="$ARG2"
  OUTPUT_PATH="${ARG3:-$OUTPUT_DEFAULT}"
else
  OUTPUT_PATH="${ARG2:-$OUTPUT_DEFAULT}"
  CHECKPOINT_DIR="$ARG3"
fi

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/third_party:${PYTHONPATH:-}"
bash "$ROOT_DIR/scripts/load_assets.sh" "$CONFIG_PATH" "" "" "$WORK_DIR"
if [[ -n "$CHECKPOINT_DIR" ]]; then
  python -m src.eval.metrics --config "$CONFIG_PATH" --task gsm8k --output "$OUTPUT_PATH" --checkpoint-dir "$CHECKPOINT_DIR"
else
  python -m src.eval.metrics --config "$CONFIG_PATH" --task gsm8k --output "$OUTPUT_PATH"
fi
