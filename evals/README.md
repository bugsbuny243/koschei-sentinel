# Evaluations

Every candidate model is evaluated through the deterministic `sentinel-eval` gate before promotion.

The current suite measures:

- **Authority:** the model cannot replace or alter the signed deterministic verdict.
- **Grounding:** claims cite known evidence, cover supported triggered rules, and preserve the weakest cited confidence.
- **Abstention:** missing evidence and no-rule cases produce explicit limitations instead of invented claims.
- **Privacy:** raw identifiers, personal data, credentials, and suite-specific forbidden terms are rejected.

The built-in baseline is the contract oracle. CI runs it without API keys against `fixtures/evals/suite.safe.jsonl`. External providers and trained checkpoints must emit the same versioned prediction contract and pass the same gate.

See [`docs/BENCHMARK.md`](../docs/BENCHMARK.md).
