# Koschei Sentinel

Evidence-grounded Web3 security model, dataset, training, evaluation, and inference platform for Koschei.

Koschei Sentinel is not allowed to replace a signed deterministic verdict. Its job is to explain bounded evidence, surface limitations, and produce structured commentary that can be checked automatically.

## v0.1.1 — Dataset Boundary

The repository provides:

- versioned case, opinion, dataset, and export-manifest contracts;
- deterministic HMAC pseudonyms for cases, targets, signatures, evidence, and lineage groups;
- PII, credential, JWT, bearer-token, and address removal from free text;
- canonical and sorted `sentinel.dataset.v1` JSONL;
- atomic writes with no partial dataset on any rejected record;
- dry-run manifests that write no training data;
- evidence-citation policy checks and a non-generative baseline engine;
- a FastAPI inference boundary, evaluation scaffolding, and CI.

No model weights, production data, or production secrets belong in this repository.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
make check
make run
```

The API starts on `http://127.0.0.1:8080`.

```bash
curl http://127.0.0.1:8080/health
curl -X POST http://127.0.0.1:8080/v1/opinions \
  -H 'content-type: application/json' \
  --data @fixtures/case.safe.json
```

## Dataset exporter

Set a deployment-owned salt. Never commit the real value.

```bash
export SENTINEL_DATASET_SALT='at-least-16-secret-characters'
sentinel-dataset-export \
  --input fixtures/arvis.source.safe.json \
  --output build/sentinel.dataset.jsonl \
  --manifest build/sentinel.manifest.json
```

Validate an export without writing training data:

```bash
sentinel-dataset-export \
  --input fixtures/arvis.source.safe.json \
  --manifest build/sentinel.dry-run.json \
  --dry-run
```

A rejected record makes the command exit with code `2`; no dataset file is written.

## Non-negotiable contract

1. The signed deterministic verdict remains final.
2. Every factual claim must cite one or more known `evidence_id` values.
3. Unknown or missing information must be reported as a limitation.
4. Raw personal data and secrets must not enter training exports.
5. Model output must be rejected when policy checks fail.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/DATA_CARD.md`](docs/DATA_CARD.md), and [`docs/DATASET_EXPORT.md`](docs/DATASET_EXPORT.md).
