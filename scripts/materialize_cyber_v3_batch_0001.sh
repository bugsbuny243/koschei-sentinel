#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)/build/cyber-v3-batch-0001}"
SNAPSHOTS="$ROOT/snapshots"
MATERIALIZED="$ROOT/materialized"
SEALED="$ROOT/sealed"
mkdir -p "$SNAPSHOTS" "$MATERIALIZED" "$SEALED"

clone_at() {
  local url="$1"
  local revision="$2"
  local dest="$3"
  if [[ ! -d "$dest/.git" ]]; then
    git clone --filter=blob:none "$url" "$dest"
  fi
  git -C "$dest" fetch --tags --force origin
  git -C "$dest" checkout --detach "$revision"
  git -C "$dest" clean -ffdqx
}

clone_at \
  https://github.com/RustSec/advisory-db.git \
  69f93e1d081d8b6fbee010e48f0b5e0d13661415 \
  "$SNAPSHOTS/rustsec-advisory-db"

clone_at \
  https://github.com/mitre-attack/attack-stix-data.git \
  v19.2 \
  "$SNAPSHOTS/attack-stix-data"

clone_at \
  https://github.com/kubernetes/website.git \
  2a031a5f0a9382d26c25cb1e58016001a21ce7b6 \
  "$SNAPSHOTS/kubernetes-website"

clone_at \
  https://github.com/VirusTotal/yara.git \
  604822da04103d13812dbcb08f4d7d42b61f94a8 \
  "$SNAPSHOTS/yara"

SPEC="$ROOT/materialization-spec.json"
cat > "$SPEC" <<JSON
{
  "schema_version": "sentinel.cyber-snapshot-materialization-spec.v3",
  "approved_catalog_path": "configs/corpus/cyber-v3.sources.approved.jsonl",
  "output_root": "$MATERIALIZED",
  "inputs": [
    {"source_id": "rustsec.advisory.database", "snapshot_path": "$SNAPSHOTS/rustsec-advisory-db"},
    {"source_id": "mitre.attack.knowledge", "snapshot_path": "$SNAPSHOTS/attack-stix-data"},
    {"source_id": "kubernetes.security.docs", "snapshot_path": "$SNAPSHOTS/kubernetes-website"},
    {"source_id": "yara.official.rules.docs", "snapshot_path": "$SNAPSHOTS/yara"}
  ]
}
JSON

sentinel-cyber-materialize --spec "$SPEC"

BATCH_SPEC="$ROOT/batch-spec.json"
cat > "$BATCH_SPEC" <<JSON
{
  "schema_version": "sentinel.cyber-collection-batch-spec.v3",
  "batch_id": "cyber-v3-batch-0001",
  "approved_catalog_path": "configs/corpus/cyber-v3.sources.approved.jsonl",
  "inputs": [
    {
      "source_id": "rustsec.advisory.database",
      "manifest_path": "$MATERIALIZED/rustsec.advisory.database/artifacts.jsonl",
      "corpus_path": "$MATERIALIZED/rustsec.advisory.database/training-corpus.jsonl"
    },
    {
      "source_id": "mitre.attack.knowledge",
      "manifest_path": "$MATERIALIZED/mitre.attack.knowledge/artifacts.jsonl",
      "corpus_path": "$MATERIALIZED/mitre.attack.knowledge/training-corpus.jsonl"
    },
    {
      "source_id": "kubernetes.security.docs",
      "manifest_path": "$MATERIALIZED/kubernetes.security.docs/artifacts.jsonl",
      "corpus_path": "$MATERIALIZED/kubernetes.security.docs/training-corpus.jsonl"
    },
    {
      "source_id": "yara.official.rules.docs",
      "manifest_path": "$MATERIALIZED/yara.official.rules.docs/artifacts.jsonl",
      "corpus_path": "$MATERIALIZED/yara.official.rules.docs/training-corpus.jsonl"
    }
  ]
}
JSON

sentinel-cyber-batch-seal --spec "$BATCH_SPEC" --output-dir "$SEALED"

python - <<PY
import json
from pathlib import Path
seal = json.loads(Path("$SEALED/seal.json").read_text())
print(json.dumps(seal, indent=2, sort_keys=True))
if not seal["ready_for_training_pipeline"]:
    raise SystemExit("Cyber v3 Batch 0001 seal FAILED")
PY
