# Koschei Sentinel Candidate Finalization v1

`sentinel-finalize-candidate` closes the offline incubation evidence chain without granting production authority.

```bash
sentinel-finalize-candidate \
  --candidate-id sentinel-incubation-v1 \
  --receipt build/receipts/sentinel-incubation-v1.receipt.json \
  --autotrain-plan build/autotrain/sentinel-incubation-v1.json \
  --adapter-manifest build/training/sentinel-incubation-v1/adapter-manifest.json \
  --comparison-matrix build/comparisons/sentinel-incubation-v1/comparison-matrix.json \
  --registry build/models/incubation-registry.json \
  --output-dir build/finalizations/sentinel-incubation-v1
```

The command requires all of the following to agree:

- the sealed offline-training receipt and its own digest;
- the exact adapter manifest and adapter digest;
- base model, pinned revision, dataset and training-config lineage;
- the approved autotrain plan;
- a comparison matrix whose own digest is valid;
- a passing benchmark report for the same candidate;
- the previous append-only incubation registry.

The output directory is created atomically and contains:

```text
candidate-finalization.json
incubation-registry.json
```

The second file is a new registry snapshot. The command never replaces the input registry automatically. The finalization record is permanently marked:

```text
state = finalized_incubation
authority = explanation_only
automatic_registry_replacement_allowed = false
automatic_promotion_allowed = false
production_deployment_allowed = false
```

This closes the research evidence chain:

```text
autotrain plan
→ offline GPU job
→ verified adapter receipt
→ hard-gate benchmark
→ sealed incubation finalization bundle
```

It does not promote, deploy, serve, or connect the candidate to Koschei Web3 Hub.
