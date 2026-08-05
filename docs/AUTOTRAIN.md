# Koschei Sentinel Autotrain

Status: `incubation_only`

`sentinel-autotrain` is the first control-plane step for background model training. It validates an existing `sentinel.training-config.v1` and deterministic training plan, then emits a versioned `sentinel.autotrain-plan.v1` decision.

The command does not load a model, start a GPU worker, promote a candidate or deploy anything to Web3 Hub.

## Plan an offline candidate

```bash
sentinel-autotrain \
  --policy configs/autotrain/incubation.v1.json \
  --config configs/training/qlora.t4.qwen2.5-1.5b.json \
  --output build/autotrain/incubation-plan.json
```

Exit codes:

- `0`: the release is ready for a separately executed offline training run;
- `2`: the policy, training config, dataset or digest binding is invalid;
- `3`: inputs are valid but the incubation policy blocked the run.

## Fail-closed properties

The v1 policy requires:

- a materialized dataset release;
- a passing readiness report bound to the exact quality manifest;
- a training config that sets `require_readiness`;
- no more warnings than the policy allows;
- an incubation-only candidate stage.

The following fields are literal `false` in the policy and output schema:

- automatic training execution;
- automatic candidate promotion;
- production deployment.

A later GPU worker may consume a passing plan, but it must remain a separate offline action. No passing plan creates a path into the Web3 runtime.
