#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"

TRAINING_ROOT="build/cyber-training"
CORPUS="$TRAINING_ROOT/defense-reflex-v3"
CONFIG="configs/training/cyber-sft.qwen3.5-0.8b.micro.json"
TARGET_9B_NORMAL_CONFIG="configs/training/cyber-sft.qwen3.5-9b.smoke.json"
TARGET_9B_LOWMEM_CONFIG="configs/training/cyber-sft.qwen3.5-9b.smoke.lowmem.json"
PLAN="$TRAINING_ROOT/qwen35-08b-micro.plan.json"
RUN_DIR="$TRAINING_ROOT/runs/qwen35-08b-micro-001"
EXPORT_ROOT="${KOSCHEI_KAGGLE_OUTPUT_ROOT:-/kaggle/working/koschei-sentinel-micro-output}"

export HF_HOME="${HF_HOME:-/kaggle/working/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

EXPORT_ROOT="$(python - "$EXPORT_ROOT" "$PWD" "$HF_HOME" <<'PY'
from pathlib import Path
import sys

candidate = Path(sys.argv[1]).expanduser().resolve()
repo = Path(sys.argv[2]).resolve()
hf_home = Path(sys.argv[3]).expanduser().resolve()
working = Path("/kaggle/working").resolve()

if candidate == working or working not in candidate.parents:
    raise SystemExit("KOSCHEI_KAGGLE_OUTPUT_ROOT must be a child of /kaggle/working")
for protected, label in ((repo, "repository"), (hf_home, "HF_HOME")):
    overlaps = (
        candidate == protected
        or candidate in protected.parents
        or protected in candidate.parents
    )
    if overlaps:
        raise SystemExit(f"KOSCHEI_KAGGLE_OUTPUT_ROOT overlaps protected {label} path")
print(candidate)
PY
)"

rm -rf "$EXPORT_ROOT"
mkdir -p "$EXPORT_ROOT"

printf '\n[Koschei] Installing text-only Cyber SFT runtime\n'
python -m pip install -e '.[training]'

if [[ ! -d "$CORPUS" ]]; then
  sentinel-cyber-seed-curriculum --output-root "$TRAINING_ROOT"
fi

printf '\n[Koschei] Qwen3.5-0.8B exact model preflight\n'
sentinel-cyber-model-preflight \
  --config "$CONFIG" \
  | tee "$EXPORT_ROOT/model-preflight.json"

printf '\n[Koschei] Kaggle GPU preflight\n'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "$EXPORT_ROOT/nvidia-smi.txt"
else
  echo "nvidia-smi is unavailable; Kaggle GPU accelerator may not be enabled" >&2
  exit 2
fi

printf '\n[Koschei] Runtime + tokenizer readiness (micro)\n'
sentinel-cyber-training-readiness \
  --config "$CONFIG" \
  --check-runtime \
  --check-tokenization \
  | tee "$EXPORT_ROOT/readiness-micro.json"

printf '\n[Koschei] Executing real Qwen3.5-0.8B text-only QLoRA micro-smoke\n'
sentinel-cyber-sft \
  --config "$CONFIG" \
  --plan-output "$PLAN" \
  --execute \
  2>&1 | tee "$EXPORT_ROOT/training-micro.log"

printf '\n[Koschei] Verifying micro adapter + receipt + text runtime\n'
sentinel-cyber-sft-verify \
  --run-dir "$RUN_DIR" \
  | tee "$EXPORT_ROOT/verification.json"

REPOSITORY_COMMIT="$(git rev-parse HEAD)"
printf '\n[Koschei] Building fail-closed micro run attestation\n'
sentinel-cyber-sft-attest \
  --config "$CONFIG" \
  --plan "$PLAN" \
  --run-dir "$RUN_DIR" \
  --model-preflight "$EXPORT_ROOT/model-preflight.json" \
  --verification "$EXPORT_ROOT/verification.json" \
  --profile micro \
  --repository-commit "$REPOSITORY_COMMIT" \
  | tee "$EXPORT_ROOT/run-attestation.json"

mkdir -p "$EXPORT_ROOT/run"
cp -a "$RUN_DIR"/. "$EXPORT_ROOT/run/"
cp "$PLAN" "$EXPORT_ROOT/training-plan.json"
cp "$CONFIG" "$EXPORT_ROOT/training-config.json"
cp "$CORPUS/manifest.json" "$EXPORT_ROOT/corpus-manifest.json"
cp "$CORPUS/examples.jsonl" "$EXPORT_ROOT/corpus-examples.jsonl"
printf 'micro\n' > "$EXPORT_ROOT/selected-profile.txt"
printf '%s\n' "$REPOSITORY_COMMIT" > "$EXPORT_ROOT/repository-commit.txt"

printf '\n[Koschei] Offline-verifying portable micro bundle\n'
sentinel-cyber-sft-export-verify \
  --export-dir "$EXPORT_ROOT" \
  | tee "$EXPORT_ROOT/export-verification.json"

run_scale_gate() {
  local profile="$1"
  local config="$2"
  local output="$EXPORT_ROOT/scale-gate-${profile}.json"
  set +e
  sentinel-cyber-sft-scale-gate \
    --micro-export "$EXPORT_ROOT" \
    --target-config "$config" \
    | tee "$output"
  local status=${PIPESTATUS[0]}
  set -e
  return "$status"
}

printf '\n[Koschei] Evaluating micro-to-9B normal profile gate\n'
if run_scale_gate normal "$TARGET_9B_NORMAL_CONFIG"; then
  SCALE_STATUS=0
  cp "$EXPORT_ROOT/scale-gate-normal.json" "$EXPORT_ROOT/scale-gate.json"
  printf 'normal\n' > "$EXPORT_ROOT/recommended-9b-profile.txt"
else
  NORMAL_SCALE_STATUS=$?
  if [[ "$NORMAL_SCALE_STATUS" -eq 2 ]]; then
    echo "[Koschei] Normal scale gate execution failed; refusing ambiguous evidence." >&2
    exit 2
  fi
  printf '\n[Koschei] Normal 9B gate blocked; evaluating low-memory profile.\n'
  if run_scale_gate lowmem "$TARGET_9B_LOWMEM_CONFIG"; then
    SCALE_STATUS=0
    cp "$EXPORT_ROOT/scale-gate-lowmem.json" "$EXPORT_ROOT/scale-gate.json"
    printf 'lowmem\n' > "$EXPORT_ROOT/recommended-9b-profile.txt"
  else
    LOWMEM_SCALE_STATUS=$?
    if [[ "$LOWMEM_SCALE_STATUS" -eq 2 ]]; then
      echo "[Koschei] Low-memory scale gate execution failed; refusing ambiguous evidence." >&2
      exit 2
    fi
    SCALE_STATUS=1
    cp "$EXPORT_ROOT/scale-gate-lowmem.json" "$EXPORT_ROOT/scale-gate.json"
    printf 'blocked\n' > "$EXPORT_ROOT/recommended-9b-profile.txt"
  fi
fi

python - "$EXPORT_ROOT" <<'PY'
from pathlib import Path
import shutil
import sys

root = Path(sys.argv[1]).resolve()
archive = root.parent / "koschei-sentinel-qwen35-08b-micro"
zip_path = Path(shutil.make_archive(str(archive), "zip", root_dir=root))
print(f"\n[Koschei] Export archive: {zip_path}")
PY

printf '\n[Koschei] REAL QLoRA MICRO-SMOKE COMPLETE\n'
if [[ "$SCALE_STATUS" -eq 0 ]]; then
  printf '[Koschei] Micro evidence permits a 9B attempt; inspect scale-gate.json and recommended-9b-profile.txt.\n'
else
  printf '[Koschei] Micro run is valid, but both 9B profiles are blocked on this runtime.\n'
fi
printf '[Koschei] Pipeline proof only: promotion_eligible=false by design.\n'
