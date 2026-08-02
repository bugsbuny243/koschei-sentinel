# Koschei Sentinel

Evidence-grounded Web3 security model, dataset, training, evaluation, and inference platform for Koschei.

Koschei Sentinel is not allowed to replace a signed deterministic verdict. Its job is to explain bounded evidence, surface limitations, and produce structured commentary that can be checked automatically.

## v0.2.0 — Dataset Release Gate

The repository now provides:

- strict `arvis.export.v1` ingestion;
- deterministic HMAC pseudonyms and private source fingerprints;
- PII, credential, JWT, bearer-token, address, and long-secret sanitization;
- canonical `sentinel.dataset.v1` JSONL;
- atomic fail-closed export batches and dry-run manifests;
- deterministic leakage-safe train/validation/test assignment by `group_ref`;
- duplicate, privacy, identifier-format, and malformed-row quality gates;
- atomic versioned release directories with per-split digests and statistics;
- evidence-citation policy checks, a baseline inference engine, API boundary, and CI.

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
3. Unknown or missing information must be reported as a limitation.
4. Raw personal data and secrets must not enter training exports.
5. Related clusters and incident families must never cross dataset splits.
6. Model output must be rejected when policy checks fail.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/DATA_CARD.md`](docs/DATA_CARD.md), [`docs/DATASET_EXPORT.md`](docs/DATASET_EXPORT.md), and [`docs/DATASET_RELEASE.md`](docs/DATASET_RELEASE.md).
