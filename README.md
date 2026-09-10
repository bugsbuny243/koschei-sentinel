# Koschei Sentinel

## Product direction and joint packages

Koschei Sentinel is an independent cybersecurity model for code development, vulnerability research, attack-path reasoning and remediation. The single-model architecture target remains 397B total / 35B active parameters. ARVIS commentary is one application; trained-product readiness and competitive superiority require independent evaluation evidence.

**Koschei Lang and Koschei Sentinel are offered together in the same commercial packages.** Their runtime dependencies and release gates remain independent. See [product direction](docs/PRODUCT_DIRECTION_2026-09-10.md) and the [offline bundle contract](docs/COMMERCIAL_BUNDLE_V1.md).

Evidence-grounded Web3 security model, dataset, training, evaluation, and inference platform for Koschei.

Koschei Sentinel is not allowed to replace a signed deterministic verdict. Its job is to explain bounded evidence, surface limitations, and produce structured commentary that can be checked automatically.

## v0.8.0 — Read-Only Neon Dataset Source

The repository now provides:

- strict `arvis.export.v1` ingestion;
- deterministic HMAC pseudonyms and private source fingerprints;
- PII, credential, JWT, bearer-token, address, and long-secret sanitization;
- canonical `sentinel.dataset.v1` JSONL;
- atomic fail-closed export batches and dry-run manifests;
- deterministic leakage-safe train/validation/test assignment by `group_ref`;
- duplicate, privacy, identifier-format, and malformed-row quality gates;
- atomic versioned release directories with per-split digests and statistics;
- a one-shot private ARVIS dataset builder that never copies raw source data into releases;
- a direct read-only Neon source that keeps raw production identifiers in memory only;
- fixed-table and required-column validation before any Neon record is accepted;
- bounded Neon reads with a read-only transaction and statement timeout;
- versioned real-training readiness policies and deterministic readiness reports;
- minimum example, lineage, split, grade, evidence-kind, confidence, and group-concentration gates;
- training plans cryptographically bound to the exact passing readiness report and release manifest;
- versioned benchmark-case, prediction, benchmark-report, candidate, and matrix contracts;
- offline authority, grounding, confidence, abstention, and privacy evaluation;
- baseline, replay, OpenAI-compatible, and provider-locked Together adapters;
- network-deny-by-default provider execution and private-endpoint controls;
- preflight request-count and conservative USD budget enforcement;
- Together JSON Schema outputs with explicit reasoning controls;
- deterministic `sentinel.training-config.v1`, `sentinel.training-plan.v1`, and `sentinel.adapter-manifest.v1` contracts;
- an executable 4-bit or 8-bit QLoRA/LoRA trainer with prompt-token loss masking;
- a pinned T4-compatible Qwen2.5 1.5B training configuration using `float16`;
- atomic adapter output and deterministic adapter artifact digests;
- secrets-free CI planning and dataset-readiness checks;
- a baseline inference engine and FastAPI boundary.

No model weights, production data, raw ARVIS exports, private training releases, database URLs, salts, or production secrets belong in this repository.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
make check
make run
```

## Build directly from Neon

Store `NEON_DATABASE_URL` and `SENTINEL_DATASET_SALT` only in a secret manager or runtime environment. Install the optional database dependency:

```bash
pip install -e '.[dev,neon]'
```

Run the complete pipeline without preserving a release:

```bash
sentinel-neon-build \
  --salt-version production-2026-08 \
  --policy configs/readiness/training-ready.v1.json \
  --dry-run
```

Materialize only after the same checks report ready:

```bash
sentinel-neon-build \
  --output-dir build/releases/sentinel-v0.8-training \
  --salt-version production-2026-08 \
  --policy configs/readiness/training-ready.v1.json
```

The Neon lane reads only signed unified verdicts, latest signed module verdicts, and bounded holder snapshots through a fixed read-only query. It validates required tables and columns first, suppresses provider diagnostics, never writes raw source rows, and passes records directly into the existing pseudonymization and readiness gates. See [`docs/NEON_DATASET.md`](docs/NEON_DATASET.md).

## Build from private ARVIS export files

Keep raw ARVIS exports outside Git and set a deployment-owned salt through a secret manager or runtime environment:

```bash
export SENTINEL_DATASET_SALT='at-least-16-secret-characters'

sentinel-dataset-build \
  --input /private/arvis-export-001.json \
  --input /private/arvis-export-002.jsonl \
  --output-dir build/releases/sentinel-v0.8-training \
  --salt-version production-2026-08 \
  --policy configs/readiness/training-ready.v1.json
```

The final release is written only when strict export validation, pseudonymization, privacy checks, lineage-safe splitting, digest verification, and all readiness thresholds pass. Use `--dry-run` to validate without materializing any release.

Inspect an existing release independently:

```bash
sentinel-dataset-readiness \
  --release build/releases/sentinel-v0.8-training \
  --policy configs/readiness/training-ready.v1.json \
  --output build/releases/sentinel-v0.8-training/readiness-report.json
```

A readiness command exits with code `0` when ready, `3` when structurally valid but below policy, and `2` when rejected.

The lower-level `sentinel-dataset-export` and `sentinel-dataset-split` commands remain available for controlled workflows. See [`docs/DATASET_EXPORT.md`](docs/DATASET_EXPORT.md), [`docs/DATASET_RELEASE.md`](docs/DATASET_RELEASE.md), and [`docs/DATASET_READINESS.md`](docs/DATASET_READINESS.md).

## Plan or execute QLoRA training

The default training command is a network-free planner. It validates the immutable base-model reference, exact release manifest, split digests, every dataset row, group counts, readiness binding, output path, and effective batch plan without loading a model.

Fixture plan:

```bash
sentinel-train \
  --config fixtures/training/config.safe.json \
  --plan-output build/training/fixture.plan.json
```

Pinned T4 real-training plan:

```bash
pip install -e '.[dev,training]'
sentinel-train \
  --config configs/training/qlora.t4.qwen2.5-1.5b.json \
  --plan-output build/training/qwen2.5-1.5b.plan.json
```

Inspect the plan before adding `--execute`. The real-training configuration requires `build/releases/sentinel-v0.8-training/readiness-report.json` to be present, passing, stored inside the release, and bound to that release's exact quality-manifest digest.

The initial supervision target is generated by the deterministic baseline. This teaches the adapter Sentinel's output contract, evidence grounding, confidence ceiling, abstention behavior, and authority boundary. It does not yet add reviewed expert knowledge. See [`docs/TRAINING.md`](docs/TRAINING.md).

## Run the model benchmark

Run the deterministic baseline without network access or API keys:

```bash
sentinel-eval \
  --suite fixtures/evals/suite.safe.jsonl \
  --candidate sentinel-baseline-v0.4 \
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

## Compare model candidates

Run the secrets-free baseline and replay matrix:

```bash
sentinel-compare \
  --suite fixtures/evals/suite.safe.jsonl \
  --registry fixtures/models/candidates.safe.json \
  --output-dir build/comparisons/safe \
  --require-all
```

Network adapters remain disabled unless `--allow-network` is supplied. Local and private endpoints require the additional `--allow-local-network` override. API credentials are referenced only by environment-variable name in the candidate registry.

## Plan Together cost without calling Together

```bash
sentinel-compare \
  --suite fixtures/evals/suite.safe.jsonl \
  --registry fixtures/models/candidates.together.low-cost.json \
  --plan-only
```

The planner is intentionally conservative and performs no network request. Every provider response is constrained by a case-specific JSON Schema and revalidated by the deterministic policy and benchmark layers.

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
7. Provider access is denied unless explicitly enabled and safety-validated.
8. External calls are rejected before credentials or DNS when request or budget limits fail.
9. Provider output cannot omit required semantic fields or change case identity.
10. A model cannot be promoted when any hard benchmark gate fails.
11. Training cannot use a mutable base-model revision or execute remote model code.
12. Training cannot begin when a release split digest, count, group count, or row contract drifts.
13. Real training cannot begin without a passing readiness report bound to the exact release.
14. A dataset below minimum diversity, independence, confidence, or split thresholds is not materialized as training-ready.
15. Neon ingestion must remain read-only, schema-checked, bounded, and free of raw-source artifacts.
16. Existing adapter output directories cannot be overwritten.
17. A trained adapter remains untrusted until it passes the independent benchmark and comparison gates.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/DATA_CARD.md`](docs/DATA_CARD.md), [`docs/DATASET_READINESS.md`](docs/DATASET_READINESS.md), [`docs/NEON_DATASET.md`](docs/NEON_DATASET.md), [`docs/TRAINING.md`](docs/TRAINING.md), [`docs/BENCHMARK.md`](docs/BENCHMARK.md), [`docs/MODEL_ADAPTERS.md`](docs/MODEL_ADAPTERS.md), and [`docs/TOGETHER.md`](docs/TOGETHER.md).
