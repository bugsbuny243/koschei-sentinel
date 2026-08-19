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

The GPU inference host should receive the answer-key-isolated inference pack plus the verified promotion-eligible Cyber SFT candidate export. The candidate export provides the exact training config, run attestation, adapter, and portable provenance evidence required to verify the candidate before model loading. The full Gold release, especially `holdout/cases.jsonl`, belongs on the evaluation side and should not be mounted into the model-inference environment.

## 7. Run the trained adapter against HOLDOUT

First verify the portable candidate export independently:

```bash
sentinel-cyber-sft-export-verify \
  --export-dir build/cyber-training/exports/qwen35-9b-gold-defense-v1
```

Then create a CPU-side inference plan. The plan re-verifies the complete candidate export and binds the adapter digest, canonical training-config SHA, run-attestation SHA, candidate-export verification digest, input pack, and generation policy.

```bash
sentinel-gold-holdout-infer \
  --inference-pack build/gold-holdout-inference \
  --candidate-export build/cyber-training/exports/qwen35-9b-gold-defense-v1 \
  --model-ref koschei-sentinel:qwen35-9b-gold-defense-v1 \
  --generation-policy configs/training/gold-holdout-generation-policy.v1.json \
  --plan-output build/gold-holdout-inference-plan.json
```

Then execute on the GPU host:

```bash
sentinel-gold-holdout-infer \
  --inference-pack build/gold-holdout-inference \
  --candidate-export build/cyber-training/exports/qwen35-9b-gold-defense-v1 \
  --model-ref koschei-sentinel:qwen35-9b-gold-defense-v1 \
  --generation-policy configs/training/gold-holdout-generation-policy.v1.json \
  --output-dir build/gold-holdout-inference-output \
  --execute
```

Generation is deterministic: sampling is disabled and the model must return exactly one JSON object with `interpretation` and `defense_sequence`. The runner does not repair malformed model output. Parse failures are recorded as failed cases.

The output persists `generation-policy.json`, `training-config.json`, `run-attestation.json`, and `candidate-export-verification.json` alongside the inference plan, receipt, predictions, and failures. These are audit snapshots, not the final trust anchor.

For real runner outputs, `model_revision` is the verified adapter digest. Operators do not supply a free-form model revision. A smoke-only, non-promotion-eligible, config-drifted, attestation-drifted, or otherwise invalid candidate export is rejected before model loading.

## 8. Offline-verify inference output

Before evaluation, independently re-verify the inference artifacts and the original candidate export on the evaluation host:

```bash
sentinel-gold-holdout-infer-verify \
  --inference-pack build/gold-holdout-inference \
  --output-dir build/gold-holdout-inference-output \
  --candidate-export build/cyber-training/exports/qwen35-9b-gold-defense-v1
```

Verification checks:

- inference plan and receipt self-hashes,
- exact input-pack binding,
- persisted generation-policy schema and SHA,
- canonical training-config SHA against the plan and run attestation,
- run-attestation self-hash, candidate identity, and promotion eligibility,
- a fresh independent verification of the original portable candidate export,
- persisted candidate-export verification snapshot against that fresh result,
- exact candidate run/base/adapter identity against the inference plan,
- prediction self-hashes,
- model/adapter identity consistency,
- prediction and failure input-context bindings,
- prediction/failure SHA values,
- complete case accounting.

The evaluation host does not trust a GPU-produced `valid=true` snapshot by itself. The original candidate export must still verify independently and match the plan-bound provenance digests.

Every HOLDOUT case must appear exactly once: either as a valid prediction or as an inference failure.

## 9. Evaluate and bind evidence

The production evaluation path consumes the verified inference output rather than an operator-authored prediction file:

```bash
sentinel-gold-holdout-eval evaluate-output \
  --release-dir build/gold-defense-release \
  --inference-pack build/gold-holdout-inference \
  --inference-output build/gold-holdout-inference-output \
  --candidate-export build/cyber-training/exports/qwen35-9b-gold-defense-v1 \
  --policy configs/training/gold-holdout-evaluation-policy.v1.json \
  --output build/gold-holdout-report.json
```

The evaluator measures:

- structural sequence exactness,
- mode accuracy,
- action accuracy,
- target accuracy,
- evidence-selection accuracy,
- evidence grounding,
- target grounding,
- outcome-verification discipline.

Evidence grounding and evidence selection are intentionally different. Grounding asks whether cited evidence exists in the model-visible graph. Selection accuracy asks whether the model selected the same supporting evidence set as the reviewed Gold defense step. A visible but wrong or incomplete evidence set can therefore be fully grounded and still fail Gold evidence-selection accuracy.

Evidence or targets absent from the model-visible graph fail grounding. Missing HOLDOUT predictions also fail the evaluation.

If every HOLDOUT answer fails parsing, the verified runner output is not discarded as an exception-only failure. Evaluation emits a formal report with the exact adapter identity, zero predictions, all HOLDOUT cases marked missing, all quality metrics at `0.0`, and `passed=false`. The same failure is then bindable into Gold evaluation evidence, preserving why the candidate failed.

Then bind the release audit, input pack, independently reverified candidate export, verified inference run, generation policy, evaluation policy, and evaluation report into one evidence object:

```bash
sentinel-gold-holdout-evidence \
  --release-dir build/gold-defense-release \
  --inference-pack build/gold-holdout-inference \
  --inference-output build/gold-holdout-inference-output \
  --candidate-export build/cyber-training/exports/qwen35-9b-gold-defense-v1 \
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

The production promotion path does not trust the supplied Gold evidence JSON by itself. Before creating Promotion v4, it rebuilds Gold evaluation evidence from the original Gold release, answer-key-isolated inference pack, verified inference output, original portable candidate export, and exact evaluation policy. The freshly rebuilt evidence must be semantically identical to the supplied evidence or promotion fails closed.

The source-rebuilt evidence is then checked against the exact candidate model reference and adapter-digest revision. Promotion re-applies the Gold minimum case-count requirement and every quality threshold, including evidence-selection accuracy, and binds the Gold source audit and inference-verification digests into the promotion receipt.

The production command therefore requires the original source artifacts:

```bash
sentinel-cyber-defense-promotion \
  --promotion-id promotion:qwen35-9b-gold-defense-v1 \
  --candidate-model koschei-sentinel:qwen35-9b-gold-defense-v1 \
  --candidate-revision <verified-adapter-digest> \
  --training-bundle build/cyber-training-bundle.json \
  --cyber-range-report build/cyber-range-report.json \
  --multi-incident-range-report build/multi-incident-range-report.json \
  --defense-load-range-report build/defense-load-range-report.json \
  --gold-holdout-evidence build/gold-holdout-evidence.json \
  --gold-holdout-policy configs/training/gold-holdout-evaluation-policy.v1.json \
  --gold-release-dir build/gold-defense-release \
  --gold-inference-pack build/gold-holdout-inference \
  --gold-inference-output build/gold-holdout-inference-output \
  --gold-candidate-export build/cyber-training/exports/qwen35-9b-gold-defense-v1 \
  --output build/cyber-defense-promotion-v4.json
```

The repository production Gold evaluation policy requires at least 50 unseen HOLDOUT cases. A smaller test-only policy can exercise artifact plumbing in unit tests, but it does not weaken the production policy file used by the promotion command.

If any source artifact, candidate provenance binding, inference verification, Gold threshold, or independent defense gate fails, `ready_for_promotion=false`.
