# Dataset export contract

## Input

The exporter accepts a JSON array or JSONL stream of strict `arvis.export.v1` objects. Unknown fields, missing signatures, nested attribute objects, duplicate evidence IDs, invalid confidence labels, and over-limit collections are rejected.

## Output

Each canonical JSONL row is a `sentinel.dataset.v1` object containing:

- a deterministic `example_id`;
- a deterministic `group_ref` for leakage-safe split assignment;
- an HMAC-SHA-256 `source_digest` for private provenance and revocation;
- a validated `sentinel.case.v1` model input.

Raw source identifiers are never retained. Case IDs, targets, verdict signatures, evidence IDs, addresses found in prose, and lineage keys are HMAC-pseudonymized with `SENTINEL_DATASET_SALT`.

## Fail-closed behavior

The batch is atomic. If any input record is invalid, the manifest reports all accepted and rejected counts but no training JSONL is written. Dry runs always skip the training-data write. Validation errors in manifests include field locations and error types but never rejected input values.

## Salt rotation

Changing the salt changes every pseudonym, source digest, and leakage group. A dataset release must record a non-secret salt version label outside the training rows. The salt itself belongs in a secret manager, never Git or logs.
