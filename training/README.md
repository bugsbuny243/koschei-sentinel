# Training

Training is intentionally gated behind data quality and evaluation work.

## Planned stages

1. Export anonymized ARVIS cases using `sentinel.case.v1`.
2. Attach human-approved, evidence-cited reference opinions.
3. Freeze train, validation, challenge, and red-team splits.
4. Benchmark candidate open-weight base models.
5. Run parameter-efficient supervised fine-tuning.
6. Reject checkpoints that regress evidence citation, abstention, or privacy tests.

Raw production dumps, API keys, wallet owner identities, and unreviewed model output must never be committed.
