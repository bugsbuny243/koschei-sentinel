# Koschei Sentinel

Evidence-grounded Web3 security model, dataset, training, evaluation, and inference platform for Koschei.

Koschei Sentinel is not allowed to replace a signed deterministic verdict. Its job is to explain bounded evidence, surface limitations, and produce structured commentary that can be checked automatically.

## v0.3.0 — Model Benchmark Gate

The repository now provides:

- strict `arvis.export.v1` ingestion;
- deterministic HMAC pseudonyms and private source fingerprints;
- PII, credential, JWT, bearer-token, address, and long-secret sanitization;
- canonical `sentinel.dataset.v1` JSONL;
- atomic fail-closed export batches and dry-run manifests;
- deterministic leakage-safe train/validation/test assignment by `group_ref`;
- duplicate, privacy, identifier-format, and malformed-row quality gates;
- atomic versioned release directories with per-split digests and statistics;
- versioned benchmark-case, prediction, and benchmark-report contracts;
- offline authority, grounding, confidence, abstention, and privacy evaluation;
- a strict model-promotion gate enforced in CI;
- a baseline inference engine and FastAPI boundary.

No model weights, production data, or production secrets belong in this repository.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
make check
make run
```

## Export source evidence

Set a deployment-owned salt. Never commit the real value.

```bash
export SENTINEL_DATASET_SALT='at-least-16-secret-characters'
sentinel-dataset-export \
  --input fixtures/arvis.source.safe.json \
  --output build/sentinel.dataset.jsonl \
  --manifest build/sentinel.manifest.json
```

Validate without writing training data:

```bash
sentinel-dataset-export \
  --input fixtures/arvis.source.safe.json \
  --manifest build/sentinel.dry-run.json \
  --dry-run
```

## Create a release

```bash
sentinel-dataset-split \
  --input fixtures/dataset.safe.jsonl \
  --output-dir build/releases/sentinel-v0.2
```

Dry-run release validation:

```bash
sentinel-dataset-split \
  --input fixtures/dataset.safe.jsonl \
  --dry-run
```

A rejected export or release exits with code `2` and writes no training release.

## Run the model benchmark

Run the deterministic baseline without network access or API keys:

```bash
sentinel-eval \
  --suite fixtures/evals/suite.safe.jsonl \
  --candidate sentinel-baseline-v0.3 \
  --output build/evals/baseline.json
```

Evaluate a model or provider prediction file:

```bash
sentinel-eval \
  --suite fixtures/evals/suite.safe.jsonl \
  --predictions build/evals/candidate.predictions.jsonl \
  --output build/evals/candidate.report.json
```

The command exits with code `3` when valid predictions fail the promotion gate.

## Inference API

The API starts on `http://127.0.0.1:8080`.

```bash
curl http://127.0.0.1:8080/health
curl -X POST http://127.0.0.1:8080/v1/opinions \
  -H 'content-type: application/json' \
  --data @fixtures/case.safe.json
```

## Non-negotiable contract

1. The signed deterministic verdict remains final.
2. Every factual claim must cite one or more known `evidence_id` values.
3. Claim confidence cannot exceed the weakest cited evidence.
4. Unknown or missing information must be reported as a limitation.
5. Raw personal data and secrets must not enter training exports or model outputs.
6. Related clusters and incident families must never cross dataset splits.
7. A model cannot be promoted when any hard benchmark gate fails.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/DATA_CARD.md`](docs/DATA_CARD.md), [`docs/DATASET_EXPORT.md`](docs/DATASET_EXPORT.md), [`docs/DATASET_RELEASE.md`](docs/DATASET_RELEASE.md), and [`docs/BENCHMARK.md`](docs/BENCHMARK.md).
