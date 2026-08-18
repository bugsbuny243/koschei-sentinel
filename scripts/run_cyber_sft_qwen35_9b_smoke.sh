#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"

CONFIG="configs/training/cyber-sft.qwen3.5-9b.smoke.json"
TRAINING_ROOT="build/cyber-training"
CORPUS="$TRAINING_ROOT/defense-reflex-v3"
PLAN="$TRAINING_ROOT/qwen35-9b-smoke.plan.json"

python -m pip install -e '.[training]'

if [[ ! -d "$CORPUS" ]]; then
  sentinel-cyber-seed-curriculum --output-root "$TRAINING_ROOT"
fi

sentinel-cyber-training-readiness \
  --config "$CONFIG" \
  --check-runtime \
  --check-tokenization

sentinel-cyber-sft \
  --config "$CONFIG" \
  --plan-output "$PLAN" \
  --execute
