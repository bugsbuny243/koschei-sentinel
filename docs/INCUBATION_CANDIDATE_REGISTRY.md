# Koschei Sentinel Incubation Candidate Registry

`sentinel-register-candidate` binds three independently produced artifacts before a trained adapter can be retained as an incubation candidate:

1. a passing `sentinel.autotrain-plan.v1`;
2. the exact `sentinel.adapter-manifest.v1` produced by training;
3. a `sentinel.comparison-matrix.v1` in which the same candidate passes every benchmark hard gate.

The resulting `sentinel.incubation-candidate.v1` record contains SHA-256 identities for the dataset, training config, adapter, autotrain plan, comparison matrix and benchmark report.

## Register a candidate

```bash
sentinel-register-candidate \
  --candidate-id sentinel-incubation-v1 \
  --autotrain-plan build/autotrain/incubation-plan.json \
  --adapter-manifest build/training/sentinel-incubation-v1/adapter-manifest.json \
  --comparison-matrix build/comparisons/incubation/comparison-matrix.json \
  --registry build/models/incubation-registry.json
```

The registry is append-only at the logical level:

- an existing candidate ID cannot be replaced;
- an adapter digest cannot be registered twice;
- every record has its own digest;
- the complete registry has a digest covering all records;
- loading fails if a record or registry digest was altered.

## Authority boundary

Registration does not deploy, promote or serve a model. Every record is permanently marked:

```text
stage = incubation_candidate
authority = explanation_only
automatic_promotion_allowed = false
production_deployment_allowed = false
```

A candidate may be retained for historical replay, benchmark comparison and later research. There is still no automatic path into Koschei Web3 Hub.
