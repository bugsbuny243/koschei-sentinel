# Shadow Regression Gate v1

`sentinel-shadow-regression` compares a newly reviewed shadow run with a passing baseline on the exact same sealed replay dataset.

```bash
sentinel-shadow-regression \
  --baseline-scorecard build/shadow/baseline.scorecard.json \
  --baseline-receipt build/shadow/baseline.receipt.json \
  --candidate-scorecard build/shadow/candidate.scorecard.json \
  --candidate-receipt build/shadow/candidate.receipt.json \
  --output build/shadow/candidate.regression.json \
  --history-out build/shadow/regression-history.json
```

Before comparison, the gate verifies both scorecard digests, both receipt digests, scorecard-to-receipt bindings, identical replay SHA-256 values, identical replay case counts, and identical review thresholds. A failing baseline is not accepted as a reference.

By default, any score drop, increase in failed cases, or increase in follow-up cases is a regression. Explicit non-negative tolerances can be supplied for research experiments.

A digest-bound `sentinel.shadow-regression-history.v1` can be extended one report at a time. Duplicate reports and modified history snapshots fail closed.

A passing regression report is research evidence only:

```text
manual_review_required = true
benchmark_recheck_required = true
owner_decision_required = true
automatic_promotion_allowed = false
automatic_deployment_allowed = false
production_deployment_allowed = false
web3_runtime_integration_allowed = false
verdict_mutation_allowed = false
```

It never deploys or promotes a model, changes deterministic verdicts, serves customer traffic, or connects Sentinel to the live Web3 runtime.
