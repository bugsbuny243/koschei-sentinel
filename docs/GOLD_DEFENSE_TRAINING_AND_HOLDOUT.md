# Koschei Sentinel Gold Defense Training and HOLDOUT

This document defines the promotion-eligible Defense Reflex training and unseen evaluation path.

## 1. Pre-review split assignment

Gold scenarios are assigned to `TRAIN`, `VALIDATION`, or `HOLDOUT` before human review. The split is deterministic and bound to the scenario/report identity and a versioned split policy.

```text
Cyber Range scenario/report
        ↓
sentinel-defense-reflex-gold-queue
        ↓
TRAIN / VALIDATION / HOLDOUT review packets
```

Changing a review result cannot move the scenario to another split.

## 2. Human review

Human review is fail-closed. Reviewed targets and supporting evidence must exist in the scenario graph. `HOLDOUT` can never receive training authorization.

```text
TRAIN       → human-approved + training-authorized
VALIDATION  → human-approved + training-authorized
HOLDOUT     → human-approved + evaluation-authorized only
```

## 3. Split-safe Gold release

`sentinel-defense-reflex-gold-release` writes physically separate artifacts:

```text
gold-defense-release/
  train/
    examples.jsonl
    manifest.json
  validation/
    examples.jsonl
    manifest.json
  holdout/
    cases.jsonl
    manifest.json
  release-manifest.json
```

The HOLDOUT schema is intentionally different from the training-example schema.

## 4. Gold release audit

Run before promotion-eligible training:

```bash
sentinel-defense-reflex-gold-audit \
  --release-dir build/gold-defense-release \
  --output build/gold-defense-release-audit.json
```

The audit re-parses every artifact and verifies:

- TRAIN and VALIDATION are human-reviewed and promotion-eligible.
- Synthetic-policy examples are absent.
- HOLDOUT is evaluation-only and contains no `examples.jsonl`.
- TRAIN, VALIDATION, and HOLDOUT scenario IDs do not overlap.
- Split manifests, case self-hashes, release manifest hashes, and release digest verify.
- HOLDOUT cases cannot parse as Defense Reflex training examples.

Promotion-eligible `sentinel-cyber-training-readiness` automatically runs this audit. `sentinel-cyber-sft --execute` also refuses promotion-eligible Defense Reflex execution when the audited Gold release is invalid.

## 5. Explicit TRAIN and VALIDATION

The promotion-eligible Qwen3.5 9B example config is:

```text
configs/training/cyber-sft.qwen3.5-9b.gold.example.json
```

It uses:

```text
corpus_dir            = build/gold-defense-release/train
validation_corpus_dir = build/gold-defense-release/validation
validation_ratio      = 0.0
```

The trainer never re-splits these datasets. Validation hashes are included in the training plan, resume binding, source binding, and run attestation.

## 6. Export answer-key-isolated HOLDOUT inputs

Never give the inference runtime `holdout/cases.jsonl` directly. Export an answer-key-free inference pack instead:

```bash
sentinel-gold-holdout-eval export-inputs \
  --release-dir build/gold-defense-release \
  --output-dir build/gold-holdout-inference
```

The inference pack contains only:

```text
scenario_id
critical_entity_ids
graph_snapshots
```

It does not contain the expected interpretation, expected defense sequence, range truth, simulated outcomes, or reviewer-only context.

## 7. Run the trained adapter against HOLDOUT

First create a CPU-side inference plan. The plan re-verifies the Cyber SFT run and binds the candidate identity to the adapter digest.

```bash
sentinel-gold-holdout-infer \
  --inference-pack build/gold-holdout-inference \
  --run-dir build/cyber-training/runs/qwen35-9b-gold-defense-v1 \
  --training-config configs/training/cyber-sft.qwen3.5-9b.gold.example.json \
  --model-ref koschei-sentinel:qwen35-9b-gold-defense-v1 \
  --generation-policy configs/training/gold-holdout-generation-policy.v1.json \
  --plan-output build/gold-holdout-inference-plan.json
```

Then execute on the GPU host:

```bash
sentinel-gold-holdout-infer \
  --inference-pack build/gold-holdout-inference \
  --run-dir build/cyber-training/runs/qwen35-9b-gold-defense-v1 \
  --training-config configs/training/cyber-sft.qwen3.5-9b.gold.example.json \
  --model-ref koschei-sentinel:qwen35-9b-gold-defense-v1 \
  --generation-policy configs/training/gold-holdout-generation-policy.v1.json \
  --output-dir build/gold-holdout-inference-output \
  --execute
```

Generation is deterministic: sampling is disabled and the model must return exactly one JSON object with `interpretation` and `defense_sequence`. The runner does not repair malformed model output. Parse failures are recorded as failed cases.

For real runner outputs, `model_revision` is the verified adapter digest. Operators do not supply a free-form model revision.

## 8. Offline-verify inference output

Before evaluation, independently re-verify the inference artifacts:

```bash
sentinel-gold-holdout-infer-verify \
  --inference-pack build/gold-holdout-inference \
  --output-dir build/gold-holdout-inference-output
```

Verification checks:

- inference plan and receipt self-hashes,
- exact input-pack binding,
- prediction self-hashes,
- model/adapter identity consistency,
- prediction and failure input-context bindings,
- prediction/failure SHA values,
- complete case accounting.

Every HOLDOUT case must appear exactly once: either as a valid prediction or as an inference failure.

## 9. Evaluate and bind evidence

The production evaluation path consumes the verified inference output rather than an operator-authored prediction file:

```bash
sentinel-gold-holdout-eval evaluate-output \
  --release-dir build/gold-defense-release \
  --inference-pack build/gold-holdout-inference \
  --inference-output build/gold-holdout-inference-output \
  --policy configs/training/gold-holdout-evaluation-policy.v1.json \
  --output build/gold-holdout-report.json
```

The evaluator measures:

- structural sequence exactness,
- mode accuracy,
- action accuracy,
- target accuracy,
- evidence grounding,
- target grounding,
- outcome-verification discipline.

Evidence or targets absent from the model-visible graph fail grounding. Missing HOLDOUT predictions also fail the evaluation.

Then bind the release audit, input pack, verified inference run, generation policy, evaluation policy, and evaluation report into one evidence object:

```bash
sentinel-gold-holdout-evidence \
  --release-dir build/gold-defense-release \
  --inference-pack build/gold-holdout-inference \
  --inference-output build/gold-holdout-inference-output \
  --policy configs/training/gold-holdout-evaluation-policy.v1.json \
  --output build/gold-holdout-evidence.json
```

The Gold evidence passes only when the evaluation passes and the inference run contains zero failed cases.

## 10. Promotion v4

Cyber Defense Promotion v4 requires four independent gate families:

```text
Single-Incident Cyber Range
        +
Multi-Incident / World-Line Range
        +
Defense Load Range
        +
Verified Unseen Gold HOLDOUT Evidence
        ↓
CyberDefensePromotionEvidence v4
```

Promotion requires the Gold evidence to belong to the exact candidate model reference and revision, re-verifies the evidence self-hash, requires the same versioned evaluation policy used to build the evidence, and binds the Gold source audit and inference verification digests into the promotion receipt.

If any gate fails, `ready_for_promotion=false`.
