#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"

OUT="${KOSCHEI_KAGGLE_CPU_PREFLIGHT_ROOT:-/kaggle/working/koschei-sentinel-cpu-preflight}"
TRAINING_ROOT="build/cyber-training"
CORPUS="$TRAINING_ROOT/defense-reflex-v3"
MICRO="configs/training/cyber-sft.qwen3.5-0.8b.micro.json"
NINE_NORMAL="configs/training/cyber-sft.qwen3.5-9b.smoke.json"
NINE_LOWMEM="configs/training/cyber-sft.qwen3.5-9b.smoke.lowmem.json"

OUT="$(python - "$OUT" <<'PY'
from pathlib import Path
import sys

candidate = Path(sys.argv[1]).expanduser().resolve()
working = Path('/kaggle/working').resolve()
if candidate == working or working not in candidate.parents:
    raise SystemExit('CPU preflight output must be a child of /kaggle/working')
print(candidate)
PY
)"
rm -rf "$OUT"
mkdir -p "$OUT"

printf '\n[Koschei] Installing CPU preflight dependencies only\n'
python -m pip install -e . 'transformers>=5.12,<6' huggingface-hub

if [[ ! -d "$CORPUS" ]]; then
  sentinel-cyber-seed-curriculum --output-root "$TRAINING_ROOT"
fi

printf '\n[Koschei] Exact model-access preflight: 0.8B micro\n'
sentinel-cyber-model-preflight --config "$MICRO" | tee "$OUT/model-preflight-micro.json"

printf '\n[Koschei] Exact model-access preflight: 9B normal\n'
sentinel-cyber-model-preflight --config "$NINE_NORMAL" | tee "$OUT/model-preflight-9b-normal.json"

printf '\n[Koschei] Exact model-access preflight: 9B low-memory\n'
sentinel-cyber-model-preflight --config "$NINE_LOWMEM" | tee "$OUT/model-preflight-9b-lowmem.json"

python - "$MICRO" "$NINE_NORMAL" "$NINE_LOWMEM" "$OUT" <<'PY'
from __future__ import annotations

import json
import sys
from importlib import metadata
from pathlib import Path

from koschei_sentinel.cyber_sft_training import load_cyber_sft_config
from koschei_sentinel.cyber_training_readiness import (
    _transformers_version_blocker,
    audit_cyber_training_readiness,
)

config_paths = [Path(row) for row in sys.argv[1:4]]
out = Path(sys.argv[4]).resolve()
version = metadata.version('transformers')
blocker = _transformers_version_blocker(version)
if blocker is not None:
    raise SystemExit(blocker)

reports = []
for config_path in config_paths:
    config = load_cyber_sft_config(config_path)
    report = audit_cyber_training_readiness(
        config,
        check_runtime=False,
        check_tokenization=True,
    )
    if not report.static_plan_ready:
        raise SystemExit(f'static Cyber SFT plan failed for {config_path}')
    if report.tokenization_ready is not True:
        raise SystemExit(
            f'tokenization preflight failed for {config_path}: {report.blockers}'
        )
    if report.overlength_example_ids:
        raise SystemExit(
            f'overlength examples for {config_path}: {report.overlength_example_ids}'
        )
    payload = report.model_dump(mode='json')
    payload['config_path'] = str(config_path)
    reports.append(payload)

summary = {
    'schema_version': 'sentinel.cyber-sft-kaggle-cpu-preflight.v1',
    'transformers_version': version,
    'gpu_used': False,
    'model_weights_loaded': False,
    'reports': reports,
    'ready_for_gpu_micro_attempt': True,
}
(out / 'cpu-preflight-summary.json').write_text(
    json.dumps(summary, indent=2, sort_keys=True) + '\n',
    encoding='utf-8',
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

git rev-parse HEAD > "$OUT/repository-commit.txt"
cp "$CORPUS/manifest.json" "$OUT/corpus-manifest.json"

printf '\n[Koschei] CPU PREFLIGHT COMPLETE — GPU quota used: NO\n'
printf '[Koschei] Next step: run the micro-first GPU notebook only if cpu-preflight-summary.json says ready_for_gpu_micro_attempt=true.\n'
