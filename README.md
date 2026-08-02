# Koschei Sentinel

Evidence-grounded Web3 security model, dataset, training, evaluation, and inference platform for Koschei.

Koschei Sentinel is not allowed to replace ARVIS's signed deterministic verdict. Its job is to explain evidence, surface limitations, and produce structured commentary that can be checked automatically.

## v0.1 — Evidence Foundation

This repository currently provides:

- versioned input and output contracts;
- deterministic dataset anonymization;
- evidence-citation policy checks;
- a non-generative baseline engine;
- a FastAPI inference boundary;
- training and evaluation scaffolding;
- CI for linting and tests.

No model weights and no production secrets belong in this repository.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
make check
make run
```

The API starts on `http://127.0.0.1:8080`.

```bash
curl http://127.0.0.1:8080/health
curl -X POST http://127.0.0.1:8080/v1/opinions \
  -H 'content-type: application/json' \
  --data @fixtures/case.safe.json
```

## Non-negotiable contract

1. The signed deterministic verdict remains final.
2. Every factual claim must cite one or more known `evidence_id` values.
3. Unknown or missing information must be reported as a limitation.
4. Raw personal data and secrets must not enter training exports.
5. Model output must be rejected when policy checks fail.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/DATA_CARD.md`](docs/DATA_CARD.md).
