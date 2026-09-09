# PQ Reviewed Evidence Catalog Admission v1

## Purpose

This stage admits human-verified PQ evidence into a **reviewed research catalog**. It does not admit that evidence into a train/validation/test dataset and does not grant model-training authority.

The distinction is deliberate:

`research capture -> structured materialization -> signed human review -> reviewed research catalog -> later dataset/split/licensing gates`

A reviewed catalog entry is useful for curation, provenance inspection, deduplication, future licensing review, and later dataset construction without treating verified evidence as automatically trainable.

## Admission contract

The policy is:

`configs/corpus/pq-reviewed-evidence-admission.v1.json`

The catalog entry schema version is:

`sentinel.pq-reviewed-evidence-catalog-entry.v1`

An admission requires the complete prior trust chain to verify again:

- source/watch registry
- exact local snapshot bytes and snapshot receipt
- structured PQ claim
- materialized `sentinel.pq-network-intelligence.v1` record
- materialization receipt
- owner-authorized reviewer trust policy
- Ed25519-signed PQ evidence review proof

The review proof must have `decision=VERIFIED` and `evidence_verified=true`.

## License boundary

Catalog admission tracks license state but does not convert license state into training permission.

- `BLOCKED` is rejected before catalog admission.
- `REVIEW_REQUIRED` may enter the reviewed research catalog and remains unresolved.
- `EVAL_ONLY` requires a license reference but still receives no automatic evaluation authorization in this stage.
- `ALLOW_WITH_ATTRIBUTION` and `ALLOW_TRAINING` require a license reference and a resolved license scope.
- Even `ALLOW_TRAINING` does **not** set `training_authorization=true` here.

Training authorization is a separate later decision that must also account for split isolation, benchmark overlap, dataset versioning, licensing, and regression policy.

## Hard-closed fields

Every v1 catalog entry fixes these values:

- `reviewed_catalog_admitted=true`
- `dataset_admission_allowed=false`
- `split_assignment=null`
- `evaluation_authorization=false`
- `training_authorization=false`
- `gold_eligible=false`
- `production_activation_allowed=false`

The entry is self-hashed and written immutably. Verification rebuilds the entry from the current source artifacts, signed review proof, admission request, and policy.

## Deterministic identity

`admission_id` is derived from the admission-policy digest, record digest, and signed review-proof digest. Re-admitting the same verified identity produces the same admission id even if an operator changes the admission timestamp. The full entry hash still changes, preserving the exact admission event.

## Operator interface

Use the module CLI:

```text
python -m koschei_sentinel.pq_reviewed_evidence_admission_cli admit ...
python -m koschei_sentinel.pq_reviewed_evidence_admission_cli verify ...
```

Only public reviewer/owner keys are needed at this stage. Private keys are used earlier when producing the signed review proof and are never stored in the catalog entry.

## Non-goals

This stage does not fetch the network, resolve licensing automatically, create train/validation/test splits, authorize evaluation, authorize training, create Gold data, run a model, or promote a checkpoint.
