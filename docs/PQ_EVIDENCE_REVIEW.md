# PQ Evidence Review v1

## Purpose

The PQ research ingestion lane intentionally materializes evidence as unverified and non-training-authorizing.
This review stage adds a separate human trust artifact without mutating the immutable research record.

A reviewed evidence artifact is still not a training row, Gold example, production decision, or model-promotion authorization.

## Trust boundary

PQ evidence review uses a dedicated Ed25519 signature namespace and owner-signed reviewer policy:

- trust policy schema: `sentinel.pq-evidence-reviewer-trust-policy.v1`
- review input schema: `sentinel.pq-evidence-review-input.v1`
- review proof schema: `sentinel.pq-evidence-review-proof.v1`
- authority: `pq_evidence_review_signing_only`
- review scope: `sentinel.pq-network-intelligence.v1`

This authority is intentionally different from Gold reviewer authority. Gold review signatures cannot be replayed as PQ evidence reviews, and PQ evidence reviews do not authorize Gold release signing.

## Immutable review flow

1. Verify the PQ source watch registry and local snapshot receipt.
2. Re-hash the exact snapshot bytes.
3. Rebuild and verify the claim -> intelligence record materialization.
4. Verify the owner-signed PQ reviewer trust policy.
5. Require a human review decision for both source match and claim support.
6. Bind the review to the exact claim, record, materialization receipt, snapshot receipt, snapshot SHA, source-row SHA, canonical locator, reviewer identity, and trust-policy digest.
7. Sign that binding with the trusted reviewer Ed25519 key.
8. Re-verify the entire chain before accepting the review proof.

The original `sentinel.pq-network-intelligence.v1` record remains immutable. A verified review is represented by a separate proof rather than by rewriting the source record.

## VERIFIED decision

A `VERIFIED` review requires both:

- `source_match_verified=true`
- `claim_supported=true`

Only then does the signed review proof carry `evidence_verified=true`.

Even in this state, the following remain hard-coded false:

- `training_authorization`
- `dataset_admission_allowed`
- `gold_eligible`

Those are separate later gates.

## REJECTED decision

A reviewer may sign a `REJECTED` review to preserve negative provenance. Rejection never marks evidence verified and never authorizes training.

## Operator interface

Use the module CLI:

```text
python -m koschei_sentinel.pq_evidence_review_cli trust-create ...
python -m koschei_sentinel.pq_evidence_review_cli trust-verify ...
python -m koschei_sentinel.pq_evidence_review_cli sign ...
python -m koschei_sentinel.pq_evidence_review_cli verify ...
```

Private keys are operator-supplied local files and are never committed to the repository. The test suite generates ephemeral Ed25519 keys in temporary directories.

## Non-goals

This stage does not perform network fetching, automatically determine that a web page is authentic, authorize a dataset row, move an example into Gold, launch training, run HOLDOUT inference, or promote a model.
