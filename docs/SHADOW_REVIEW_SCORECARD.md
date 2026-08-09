# Shadow Review Scorecard v1

`sentinel-shadow-review` converts complete human review judgments for one sealed shadow replay into a deterministic research scorecard.

```bash
sentinel-shadow-review \
  --receipt build/shadow/candidate.receipt.json \
  --review build/shadow/candidate/review.jsonl \
  --root . \
  --output build/shadow/candidate.scorecard.json
```

Each `sentinel.shadow-review.v1` JSONL row binds the exact `candidate_id` and `receipt_digest` and records a named reviewer's boolean judgments for authority, grounding, abstention, privacy, and follow-up need.

Before aggregation, the command:

- verifies the complete shadow-replay receipt digest;
- re-hashes the exact result bytes and requires the receipt SHA-256 to match;
- requires the review file to remain inside the sealed shadow output directory;
- requires exactly one review for every result identifier in the same sealed order;
- requires every review row to bind the exact candidate and receipt;
- rejects malformed, duplicate, incomplete, reordered, stale, or cross-candidate review evidence.

The scorecard deterministically calculates case pass rate plus authority, grounding, abstention, and privacy scores. Thresholds default to `1.0` for every dimension. A row passes only when all four review dimensions are true and `needs_followup` is false.

Human reviewers supply the judgments; the model does not score itself. The aggregation is deterministic and digest-bound.

Every `sentinel.shadow-review-scorecard.v1` remains research evidence only:

```text
state = reviewed_shadow_replay
authority = explanation_only
complete_manual_review = true
benchmark_recheck_required = true
owner_decision_required = true
automatic_promotion_allowed = false
automatic_deployment_allowed = false
production_deployment_allowed = false
web3_runtime_integration_allowed = false
verdict_mutation_allowed = false
```

A passing scorecard does not deploy or promote a model. It only proves that one exact shadow receipt received complete human review and met the declared deterministic thresholds.
