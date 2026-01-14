#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/mdlm_small_math.yaml}"
FINETUNE_MODE="${2:-lora}"
WORK_DIR="${3:-}"

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
bash "$ROOT_DIR/scripts/load_assets.sh" "$CONFIG_PATH" "" "" "$WORK_DIR"
detect_nproc() {
  local nproc="${REOAC_NPROC:-}"
  if [[ -n "$nproc" ]]; then
    echo "$nproc"
    return
  fi
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    echo "${SLURM_GPUS_ON_NODE}"
    return
  fi
  if [[ -n "${SLURM_GPUS_PER_NODE:-}" ]]; then
    echo "${SLURM_GPUS_PER_NODE%%(*}"
    return
  fi
  if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    local count
    count=$(awk -F',' '{print NF}' <<< "${SLURM_JOB_GPUS}")
    echo "$count"
    return
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local count
    count=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
    echo "$count"
    return
  fi
  echo ""
}

NPROC="$(detect_nproc)"
if [[ -n "${NPROC}" && "${NPROC}" -gt 1 ]]; then
  if ! command -v torchrun >/dev/null 2>&1; then
    echo "torchrun not found; install PyTorch with torchrun or unset REOAC_NPROC." >&2
    exit 1
  fi
  torchrun --nproc_per_node="${NPROC}" -m src.rl.trainer --config "$CONFIG_PATH" --finetune-mode "$FINETUNE_MODE"
else
  python -m src.rl.trainer --config "$CONFIG_PATH" --finetune-mode "$FINETUNE_MODE"
fi
