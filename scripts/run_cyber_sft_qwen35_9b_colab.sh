#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/content/drive/MyDrive/Koschei-Sentinel/runtime/koschei-sentinel}"
DRIVE_ROOT="${KOSCHEI_DRIVE_ROOT:-/content/drive/MyDrive/Koschei-Sentinel}"

if [[ ! -d /content/drive/MyDrive ]]; then
  echo "Google Drive is not mounted at /content/drive/MyDrive" >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "No NVIDIA runtime detected. In Colab select a GPU runtime before training." >&2
  exit 2
fi

mkdir -p "$DRIVE_ROOT/hf-cache"
export HF_HOME="${KOSCHEI_HF_HOME:-$DRIVE_ROOT/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "$ROOT"

echo "=== Koschei Sentinel Colab Cyber SFT ==="
echo "repo: $ROOT"
echo "HF_HOME: $HF_HOME"
nvidia-smi

bash scripts/run_cyber_sft_qwen35_9b_smoke.sh "$ROOT"

echo "=== Cyber SFT output ==="
find build/cyber-training/runs/qwen35-9b-smoke-001 -maxdepth 2 -type f -print | sort
