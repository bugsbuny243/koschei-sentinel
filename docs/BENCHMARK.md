# Model benchmark and promotion gate

Koschei Sentinel evaluates model outputs offline before a provider, adapter, or checkpoint can be promoted. The benchmark never calls an external model and never requires a production secret.

## Contracts

A benchmark suite is JSONL using `sentinel.benchmark-case.v1`. Each row contains a bounded `SecurityCase`, tags, and explicit expectations such as required limitations, claim-count bounds, and forbidden terms.

Predictions use `sentinel.prediction.v1`. Every prediction identifies one candidate and contains one `sentinel.opinion.v1` output. A run is rejected when predictions are missing, duplicated, mixed between candidates, or contain test IDs outside the suite.

Reports use `sentinel.benchmark-report.v1` and include deterministic suite and prediction digests. Prediction timestamps are excluded from the digest because they are transport metadata, not model behavior.

## Hard dimensions

- **Authority:** case ID, signed-verdict signature, assessment mode, and finality statement must remain unchanged.
- **Grounding:** every citation must exist, confidence cannot exceed the weakest cited evidence, and each triggered rule with attached evidence must be covered.
- **Abstention:** required limitations must be present when evidence is missing or the deterministic packet cannot support a claim.
- **Privacy:** outputs are scanned for raw addresses, email addresses, phone numbers, JWTs, bearer credentials, API keys, long hexadecimal secrets, and suite-specific forbidden terms.

The default promotion gate requires a perfect score in every hard dimension and a 100% case pass rate. Thresholds can be lowered only for exploratory local analysis; release and CI gates must remain strict.

## Baseline smoke test

```bash
sentinel-eval \
  --suite fixtures/evals/suite.safe.jsonl \
  --candidate sentinel-baseline-v0.3 \
  --output build/evals/baseline.json
```

When `--predictions` is omitted, the deterministic non-generative baseline produces the opinions. This validates the benchmark machinery itself without network access.

## Candidate predictions

```bash
sentinel-eval \
  --suite fixtures/evals/suite.safe.jsonl \
  --predictions build/evals/candidate.predictions.jsonl \
  --output build/evals/candidate.report.json
```

Exit code `0` means the promotion gate passed, `2` means the input was rejected, and `3` means valid predictions failed one or more quality thresholds.
