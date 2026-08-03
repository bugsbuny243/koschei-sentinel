# Real dataset readiness gate

Koschei Sentinel v0.7 adds a fail-closed boundary between validated ARVIS exports and an actual model-training run. A syntactically valid dataset is not automatically large, diverse, or independent enough to justify training.

## One-shot private build

Keep raw ARVIS exports outside Git. Set the deployment-owned pseudonymization salt only in the runtime environment, then run:

```bash
export SENTINEL_DATASET_SALT='replace-in-a-secret-manager'

sentinel-dataset-build \
  --input /private/arvis-export-001.json \
  --input /private/arvis-export-002.jsonl \
  --output-dir build/releases/sentinel-v0.7-training \
  --salt-version production-2026-08 \
  --policy configs/readiness/training-ready.v1.json
```

The builder loads source records in memory, applies the existing strict `arvis.export.v1` validation and pseudonymization boundary, creates lineage-safe train/validation/test splits in a temporary directory, and runs the readiness gate. The final release is moved into place only when every readiness requirement passes. Raw source exports and the secret salt are never copied into the release.

Use `--dry-run` to perform the same validation without materializing a release.

## Default real-training policy

The checked-in `training-ready.v1.json` policy requires at least:

- 100 examples across 50 independent lineage groups;
- 80 train, 10 validation, and 10 test examples;
- three distinct deterministic grades;
- three distinct evidence kinds;
- at least one `VERIFIED` evidence item;
- no single lineage group representing more than 10% of examples.

These are minimum pipeline safeguards, not a claim that 100 examples produce a production-quality expert model. Larger and independently reviewed datasets remain preferable.

## Release contents

A ready release contains:

```text
build-manifest.json
quality-manifest.json
readiness-report.json
train.jsonl
validation.jsonl
test.jsonl
```

`readiness-report.json` binds the readiness decision to the exact SHA-256 digest of `quality-manifest.json`. The training planner revalidates the split files and refuses a readiness-required configuration when the report is missing, failed, malformed, outside the release directory, or bound to a different manifest.

## Inspect an existing release

```bash
sentinel-dataset-readiness \
  --release build/releases/sentinel-v0.7-training \
  --policy configs/readiness/training-ready.v1.json \
  --output build/releases/sentinel-v0.7-training/readiness-report.json
```

Exit code `0` means ready, `3` means structurally valid but below policy, and `2` means the release or policy was rejected.

## Train on a T4

The pinned T4 configuration uses the already verified base revision and `float16` compute:

```bash
sentinel-train \
  --config configs/training/qlora.t4.qwen2.5-1.5b.json \
  --plan-output build/training/qwen2.5-1.5b.plan.json
```

Inspect the network-free plan first. Add `--execute` only on a CUDA runtime after confirming the release and immutable base-model revision.

A passing readiness report authorizes training to start; it does not authorize model promotion. The produced adapter must still pass the independent authority, grounding, confidence, abstention, and privacy benchmark gates.
