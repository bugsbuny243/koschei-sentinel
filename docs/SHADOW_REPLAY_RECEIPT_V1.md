# Shadow Replay Receipt v1

`sentinel-shadow-receipt` seals the outputs of one owner-approved offline shadow replay. It does not execute inference and it does not grant deployment authority.

```bash
sentinel-shadow-receipt \
  --plan build/shadow/candidate.plan.json \
  --results build/shadow/candidate/results.jsonl \
  --root . \
  --output build/shadow/candidate.receipt.json
```

Before a receipt is created, the command revalidates:

- the complete shadow-plan digest;
- the exact replay bytes and replay SHA-256 recorded by the plan;
- the replay case count;
- that the results file is inside the plan's sealed output directory;
- every result row's `candidate_id` and `plan_digest`;
- unique result identifiers;
- exact result/replay identifier coverage in the same sealed order.

Each result JSONL row must contain `case_id` or `test_id`, the exact `candidate_id`, and the exact `plan_digest`. Additional model-output fields are allowed, but a row cannot enable production deployment, Web3 runtime integration, verdict mutation, live customer traffic, or automatic deployment. If an `authority` field is present it must remain `explanation_only`.

The receipt records the result file SHA-256 and is permanently bounded to:

```text
state = completed_shadow_replay
stage = shadow_research_candidate
authority = explanation_only
complete_case_coverage = true
ordered_case_identity_match = true
manual_review_required = true
benchmark_recheck_required = true
automatic_promotion_allowed = false
automatic_deployment_allowed = false
production_deployment_allowed = false
web3_runtime_integration_allowed = false
verdict_mutation_allowed = false
```

An existing receipt is never overwritten silently. Changing the replay, result bytes, case ordering, candidate binding, plan binding, or authority flags prevents a valid receipt from being produced.

This closes one more offline evidence link:

```text
owner approval
→ sealed replay plan
→ manually dispatched offline inference
→ exact result bytes
→ sealed shadow replay receipt
→ manual review and benchmark recheck
```

The receipt is research evidence only. It cannot alter ARVIS verdicts, serve customers, deploy the adapter, or connect Sentinel to Koschei Web3 Hub.
