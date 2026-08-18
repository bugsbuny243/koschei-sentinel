#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"

CONFIG="configs/training/cyber-sft.qwen3.5-9b.smoke.json"
TRAINING_ROOT="build/cyber-training"
CORPUS="$TRAINING_ROOT/defense-reflex-v3"
PLAN="$TRAINING_ROOT/qwen35-9b-smoke.plan.json"
RUN_DIR="$TRAINING_ROOT/runs/qwen35-9b-smoke-001"
EXPORT_ROOT="${KOSCHEI_KAGGLE_OUTPUT_ROOT:-/kaggle/working/koschei-sentinel-output}"

export HF_HOME="${HF_HOME:-/kaggle/working/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

mkdir -p "$EXPORT_ROOT"

printf '\n[Koschei] Kaggle GPU preflight\n'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi is unavailable; Kaggle GPU accelerator may not be enabled" >&2
  exit 2
fi

python -m pip install -e '.[training]'

if [[ ! -d "$CORPUS" ]]; then
  sentinel-cyber-seed-curriculum --output-root "$TRAINING_ROOT"
fi

printf '\n[Koschei] Runtime + tokenizer readiness\n'
sentinel-cyber-training-readiness \
  --config "$CONFIG" \
  --check-runtime \
  --check-tokenization \
  | tee "$EXPORT_ROOT/readiness.json"

printf '\n[Koschei] Executing real QLoRA smoke run\n'
sentinel-cyber-sft \
  --config "$CONFIG" \
  --plan-output "$PLAN" \
  --execute

printf '\n[Koschei] Verifying adapter + training receipt\n'
sentinel-cyber-sft-verify \
  --run-dir "$RUN_DIR" \
  | tee "$EXPORT_ROOT/verification.json"

rm -rf "$EXPORT_ROOT/run"
mkdir -p "$EXPORT_ROOT/run"
cp -a "$RUN_DIR"/. "$EXPORT_ROOT/run/"
cp "$PLAN" "$EXPORT_ROOT/qwen35-9b-smoke.plan.json"
cp "$CORPUS/manifest.json" "$EXPORT_ROOT/defense-reflex-v3.manifest.json"

git rev-parse HEAD > "$EXPORT_ROOT/repository-commit.txt"

python - "$EXPORT_ROOT" <<'PY'
from pathlib import Path
import shutil
import sys

root = Path(sys.argv[1]).resolve()
archive = root.parent / "koschei-sentinel-qwen35-9b-smoke"
zip_path = Path(shutil.make_archive(str(archive), "zip", root_dir=root))
print(f"\n[Koschei] Export archive: {zip_path}")
PY

printf '\n[Koschei] REAL QLoRA SMOKE RUN COMPLETE\n'
printf '[Koschei] This adapter remains SMOKE_ONLY / promotion_eligible=false by design.\n'
