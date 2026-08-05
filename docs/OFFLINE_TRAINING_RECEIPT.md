# Koschei Sentinel Offline Training Receipt v1

`sentinel-job-receipt` verifies that the adapter files produced by an offline worker match both their adapter manifest and the sealed job that authorized the run.

```bash
sentinel-job-receipt \
  --job build/jobs/sentinel-v1.job.json \
  --adapter-manifest build/training/sentinel-v1/adapter-manifest.json \
  --output build/receipts/sentinel-v1.receipt.json
```

The command checks:

- the offline job's own digest;
- run ID, base model and pinned revision;
- dataset and training-config lineage;
- expected output directory;
- every listed adapter file;
- the exact directory digest recorded by the adapter manifest.

The receipt deliberately does not claim that the model is good or production-ready. It is permanently marked:

```text
state = completed_offline
candidate_stage = incubation_candidate
adapter_files_verified = true
benchmark_required = true
benchmark_passed = false
automatic_registration_allowed = false
automatic_promotion_allowed = false
production_deployment_allowed = false
```

The next independent step remains benchmark generation and comparison. Only afterward may the adapter be entered into the separate incubation candidate registry.
