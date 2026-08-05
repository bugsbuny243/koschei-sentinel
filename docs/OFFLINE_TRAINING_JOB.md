# Koschei Sentinel Offline Training Job v1

`sentinel-job-plan` converts a passing autotrain decision into a sealed job envelope for a separately operated GPU worker.

## Plan a job

```bash
sentinel-job-plan \
  --autotrain-plan build/autotrain/incubation-plan.json \
  --config configs/training/qlora.t4.qwen2.5-1.5b.json \
  --training-plan-output build/jobs/sentinel-v1.training-plan.json \
  --output build/jobs/sentinel-v1.job.json
```

The command re-plans the training config, verifies its dataset, readiness and config digests against the approved autotrain plan, then writes:

- the exact deterministic `sentinel.training-plan.v1`;
- a `sentinel.offline-training-job.v1` containing the worker command and all lineage digests.

## Dispatch boundary

The job envelope includes the command a worker may execute:

```text
sentinel-train --config <config> --plan-output <plan> --execute
```

It does not execute that command. The schema permanently records:

```text
state = planned_offline
candidate_stage = incubation_candidate
secret_values_included = false
automatic_dispatch_allowed = false
automatic_promotion_allowed = false
production_deployment_allowed = false
```

A scheduler or human operator must explicitly dispatch the job in an isolated GPU environment. Completion still produces only an adapter manifest; benchmark and incubation-registry gates remain mandatory afterward.
