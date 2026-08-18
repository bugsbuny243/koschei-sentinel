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

## 6. Answer-key-isolated HOLDOUT

Never give the model `holdout/cases.jsonl` directly. Export an answer-key-free inference pack instead:

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

## 7. Evaluate predictions

Predictions are evaluated after inference against the protected Gold answer key:

```bash
sentinel-gold-holdout-eval evaluate \
  --release-dir build/gold-defense-release \
  --predictions build/gold-holdout-predictions.jsonl \
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

Evidence or targets absent from the model-visible graph fail grounding.

## 8. Promotion v3

Cyber Defense Promotion v3 requires four independent gate families:

```text
Single-Incident Cyber Range
        +
Multi-Incident / World-Line Range
        +
Defense Load Range
        +
Unseen Gold HOLDOUT
        ↓
CyberDefensePromotionEvidence v3
```

Promotion re-verifies the Gold HOLDOUT report self-hash and re-evaluates its metrics against the explicitly supplied versioned HOLDOUT policy. The HOLDOUT report must belong to the exact candidate model reference and revision.

If any gate fails, `ready_for_promotion=false`.
