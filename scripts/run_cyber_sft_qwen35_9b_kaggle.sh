#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"

TRAINING_ROOT="build/cyber-training"
CORPUS="$TRAINING_ROOT/defense-reflex-v3"
EXPORT_ROOT="${KOSCHEI_KAGGLE_OUTPUT_ROOT:-/kaggle/working/koschei-sentinel-output}"
PROFILE_MODE="${KOSCHEI_KAGGLE_PROFILE:-auto}"

NORMAL_CONFIG="configs/training/cyber-sft.qwen3.5-9b.smoke.json"
NORMAL_PLAN="$TRAINING_ROOT/qwen35-9b-smoke.plan.json"
NORMAL_RUN_DIR="$TRAINING_ROOT/runs/qwen35-9b-smoke-001"

LOWMEM_CONFIG="configs/training/cyber-sft.qwen3.5-9b.smoke.lowmem.json"
LOWMEM_PLAN="$TRAINING_ROOT/qwen35-9b-smoke-lowmem.plan.json"
LOWMEM_RUN_DIR="$TRAINING_ROOT/runs/qwen35-9b-smoke-lowmem-001"

export HF_HOME="${HF_HOME:-/kaggle/working/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

mkdir -p "$EXPORT_ROOT"

printf '\n[Koschei] Kaggle GPU preflight\n'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "$EXPORT_ROOT/nvidia-smi.txt"
else
  echo "nvidia-smi is unavailable; Kaggle GPU accelerator may not be enabled" >&2
  exit 2
fi

python -m pip install -e '.[training]'

if [[ ! -d "$CORPUS" ]]; then
  sentinel-cyber-seed-curriculum --output-root "$TRAINING_ROOT"
fi

run_readiness() {
  local profile="$1"
  local config="$2"
  printf '\n[Koschei] Runtime + tokenizer readiness (%s)\n' "$profile"
  sentinel-cyber-training-readiness \
    --config "$config" \
    --check-runtime \
    --check-tokenization \
    | tee "$EXPORT_ROOT/readiness-${profile}.json"
}

run_training() {
  local profile="$1"
  local config="$2"
  local plan="$3"
  local log="$EXPORT_ROOT/training-${profile}.log"

  printf '\n[Koschei] Executing real QLoRA smoke run (%s)\n' "$profile"
  set +e
  sentinel-cyber-sft \
    --config "$config" \
    --plan-output "$plan" \
    --execute \
    2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  set -e
  return "$status"
}

verify_and_export() {
  local profile="$1"
  local config="$2"
  local plan="$3"
  local run_dir="$4"

  printf '\n[Koschei] Verifying adapter + training receipt (%s)\n' "$profile"
  sentinel-cyber-sft-verify \
    --run-dir "$run_dir" \
    | tee "$EXPORT_ROOT/verification.json"

  rm -rf "$EXPORT_ROOT/run"
  mkdir -p "$EXPORT_ROOT/run"
  cp -a "$run_dir"/. "$EXPORT_ROOT/run/"
  cp "$plan" "$EXPORT_ROOT/qwen35-9b-smoke.plan.json"
  cp "$config" "$EXPORT_ROOT/selected-training-config.json"
  cp "$CORPUS/manifest.json" "$EXPORT_ROOT/defense-reflex-v3.manifest.json"
  printf '%s\n' "$profile" > "$EXPORT_ROOT/selected-profile.txt"
  git rev-parse HEAD > "$EXPORT_ROOT/repository-commit.txt"
}

SELECTED_PROFILE=""
SELECTED_CONFIG=""
SELECTED_PLAN=""
SELECTED_RUN_DIR=""

case "$PROFILE_MODE" in
  normal)
    run_readiness normal "$NORMAL_CONFIG"
    run_training normal "$NORMAL_CONFIG" "$NORMAL_PLAN"
    SELECTED_PROFILE="normal"
    SELECTED_CONFIG="$NORMAL_CONFIG"
    SELECTED_PLAN="$NORMAL_PLAN"
    SELECTED_RUN_DIR="$NORMAL_RUN_DIR"
    ;;
  lowmem)
    run_readiness lowmem "$LOWMEM_CONFIG"
    run_training lowmem "$LOWMEM_CONFIG" "$LOWMEM_PLAN"
    SELECTED_PROFILE="lowmem"
    SELECTED_CONFIG="$LOWMEM_CONFIG"
    SELECTED_PLAN="$LOWMEM_PLAN"
    SELECTED_RUN_DIR="$LOWMEM_RUN_DIR"
    ;;
  auto)
    run_readiness normal "$NORMAL_CONFIG"
    if run_training normal "$NORMAL_CONFIG" "$NORMAL_PLAN"; then
      SELECTED_PROFILE="normal"
      SELECTED_CONFIG="$NORMAL_CONFIG"
      SELECTED_PLAN="$NORMAL_PLAN"
      SELECTED_RUN_DIR="$NORMAL_RUN_DIR"
    else
      NORMAL_LOG="$EXPORT_ROOT/training-normal.log"
      if grep -Eqi 'CUDA.*out of memory|out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUDA error:.*memory' "$NORMAL_LOG"; then
        printf '\n[Koschei] Normal profile hit a CUDA-memory failure; retrying low-memory profile.\n'
        run_readiness lowmem "$LOWMEM_CONFIG"
        run_training lowmem "$LOWMEM_CONFIG" "$LOWMEM_PLAN"
        SELECTED_PROFILE="lowmem"
        SELECTED_CONFIG="$LOWMEM_CONFIG"
        SELECTED_PLAN="$LOWMEM_PLAN"
        SELECTED_RUN_DIR="$LOWMEM_RUN_DIR"
      else
        echo "[Koschei] Normal profile failed for a non-memory reason; refusing automatic fallback." >&2
        exit 2
      fi
    fi
    ;;
  *)
    echo "KOSCHEI_KAGGLE_PROFILE must be auto, normal, or lowmem" >&2
    exit 2
    ;;
esac

verify_and_export "$SELECTED_PROFILE" "$SELECTED_CONFIG" "$SELECTED_PLAN" "$SELECTED_RUN_DIR"

python - "$EXPORT_ROOT" <<'PY'
from pathlib import Path
import shutil
import sys

root = Path(sys.argv[1]).resolve()
archive = root.parent / "koschei-sentinel-qwen35-9b-smoke"
zip_path = Path(shutil.make_archive(str(archive), "zip", root_dir=root))
print(f"\n[Koschei] Export archive: {zip_path}")
PY

printf '\n[Koschei] REAL QLoRA SMOKE RUN COMPLETE (%s profile)\n' "$SELECTED_PROFILE"
printf '[Koschei] This adapter remains SMOKE_ONLY / promotion_eligible=false by design.\n'
