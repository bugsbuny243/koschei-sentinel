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

## 2. Signed human review

Human review is fail-closed. Reviewed targets and supporting evidence must exist in the scenario graph. `HOLDOUT` can never receive training authorization.

```text
TRAIN       → human-approved + training-authorized
VALIDATION  → human-approved + training-authorized
HOLDOUT     → human-approved + evaluation-authorized only
```

Production Gold reviews are also Ed25519-signed. The signature uses a domain-separated Gold-review context and binds the reviewer ID, reviewer key fingerprint, pre-review packet SHA, scenario ID, assigned split, and final review SHA. A plain SHA-256 review self-hash is not treated as reviewer authentication.

The reviewer private key is supplied only to the review command and must remain outside Git and build artifacts:

```bash
sentinel-defense-reflex-gold-review \
  --packet build/gold-review-packets/case-001.json \
  --scenario build/cyber-range/case-001.json \
  --review-spec /secure/gold-reviews/case-001.review.json \
  --reviewer-private-key /secure/keys/gold-reviewer-private.pem \
  --output build/gold-reviewed/case-001.json \
  --signature-output build/gold-reviewed/case-001.signature.json
```

The resulting signature proof can be verified only against the separately supplied trusted reviewer public key. A public key found inside a release is not accepted as its own trust root.

## 3. Split-safe signed Gold release

The production release command requires one Ed25519 signature proof for every reviewed packet and the trusted reviewer public key. Repeat `--scenario`, `--packet`, `--reviewed`, and `--review-signature` in the same order for every Gold row.

```bash
sentinel-defense-reflex-gold-release \
  --scenario build/cyber-range/case-001.json \
  --packet build/gold-review-packets/case-001.json \
  --reviewed build/gold-reviewed/case-001.json \
  --review-signature build/gold-reviewed/case-001.signature.json \
  --reviewer-public-key /secure/keys/gold-reviewer-public.pem \
  --output-dir build/gold-defense-release
```

The release keeps TRAIN, VALIDATION, and HOLDOUT physically isolated and carries the exact signed-review proof set:

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
  review-signatures.jsonl
  release-manifest.json
```

The HOLDOUT schema is intentionally different from the training-example schema. The signed-review proof set must match the release review set exactly; missing, duplicate, extra, re-bound, or wrong-key proofs fail verification.

## 4. Gold release audit

Run before promotion-eligible training or evaluation:

```bash
sentinel-defense-reflex-gold-audit \
  --release-dir build/gold-defense-release \
  --reviewer-public-key /secure/keys/gold-reviewer-public.pem \
  --output build/gold-defense-release-audit.json
```

The production audit runs two independent checks:

- structural Gold release audit, including split separation, manifest hashes, release digest, human-review-only status, and HOLDOUT isolation;
- Ed25519 review-signature audit against the externally supplied trusted reviewer public key and the exact release review set.

TRAIN and VALIDATION must be promotion-eligible and human-reviewed, synthetic-policy examples are forbidden, HOLDOUT must be evaluation-only, all split scenario IDs must be disjoint, and HOLDOUT cases must not parse as Defense Reflex training examples.

The low-level structural audit remains useful internally, but a structural self-hash alone does not prove reviewer identity. Production evidence and Promotion v4 re-run the signed-review audit independently.

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

On `sentinel-cyber-sft --execute`, the CLI revalidates the planned TRAIN and VALIDATION hashes, split disjointness, example counts, and promotion eligibility before entering the GPU executor. The text trainer performs its own execution-time checks again as an independent inner gate.

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

It does not contain the expected interpretation, expected defense sequence, range truth, simulated outcomes, reviewer-only context, or review signatures.

The GPU inference host should receive the answer-key-isolated inference pack plus the verified promotion-eligible Cyber SFT candidate export. The full Gold release, trusted reviewer key, and `holdout/cases.jsonl` belong on the evaluation side and should not be mounted into the model-inference environment.

## 7. Run the trained adapter against HOLDOUT

Create the portable candidate export from the exact training artifacts first. The producer fresh-rebuilds the run attestation, revalidates the planned TRAIN/VALIDATION sources, copies exact artifact bytes into a staging directory, verifies the staged package independently, and publishes it atomically only when verification succeeds.

```bash
sentinel-cyber-sft-export \
  --config configs/training/cyber-sft.qwen3.5-9b.gold.example.json \
  --plan <training-plan.json> \
  --training-source <training-source.json> \
  --model-preflight <model-preflight.json> \
  --verification <verification.json> \
  --attestation <run-attestation.json> \
  --output-dir build/cyber-training/exports/qwen35-9b-gold-defense-v1
```

The exporter refuses stale attestation state, source drift, TRAIN/VALIDATION count drift, symlink-traversed source artifacts, duplicate adapter paths, non-portable adapter paths, an existing destination, or a staged package that fails fresh verification. It does not rewrite provenance to make a bad run look valid.

Then verify the portable candidate export independently:

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

- inference plan and receipt self-hashes;
- exact input-pack binding;
- persisted generation-policy schema and SHA;
- canonical training-config SHA against the plan and run attestation;
- run-attestation self-hash, candidate identity, and promotion eligibility;
- a fresh independent verification of the original portable candidate export;
- persisted candidate-export verification snapshot against that fresh result;
- exact candidate run/base/adapter identity against the inference plan;
- prediction self-hashes;
- prediction/failure input-context bindings and SHA values;
- complete case accounting.

The evaluation host does not trust a GPU-produced `valid=true` snapshot by itself. Every HOLDOUT case must appear exactly once as either a valid prediction or an inference failure.

## 9. Evaluate and bind signed evidence

The production evaluation path consumes verified inference output rather than an operator-authored prediction file:

```bash
sentinel-gold-holdout-eval evaluate-output \
  --release-dir build/gold-defense-release \
  --inference-pack build/gold-holdout-inference \
  --inference-output build/gold-holdout-inference-output \
  --candidate-export build/cyber-training/exports/qwen35-9b-gold-defense-v1 \
  --policy configs/training/gold-holdout-evaluation-policy.v1.json \
  --output build/gold-holdout-report.json
```

The evaluator measures structural sequence exactness, mode/action/target accuracy, evidence-selection accuracy, evidence grounding, target grounding, and outcome-verification discipline.

Production Gold evidence additionally re-verifies every human-review signature against the external trust root and binds the resulting review-signature-audit SHA into the evidence self-hash. It also binds the independent candidate-training verification proving that the evaluated adapter is tied to the same signed Gold TRAIN and VALIDATION release bytes.

```bash
sentinel-gold-holdout-evidence \
  --release-dir build/gold-defense-release \
  --inference-pack build/gold-holdout-inference \
  --inference-output build/gold-holdout-inference-output \
  --candidate-export build/cyber-training/exports/qwen35-9b-gold-defense-v1 \
  --reviewer-public-key /secure/keys/gold-reviewer-public.pem \
  --policy configs/training/gold-holdout-evaluation-policy.v1.json \
  --output build/gold-holdout-evidence.json
```

The Gold evidence passes only when evaluation passes and inference contains zero failed cases. A wrong reviewer key, missing proof, extra proof, modified review binding, invalid Ed25519 signature, or candidate trained against different TRAIN/VALIDATION bytes prevents production evidence creation.

## 10. Promotion v4

Cyber Defense Promotion v4 requires four independent gate families:

```text
Single-Incident Cyber Range
        +
Multi-Incident / World-Line Range
        +
Defense Load Range
        +
Verified Unseen Signed Gold HOLDOUT Evidence
        ↓
CyberDefensePromotionEvidence v4
```

The production promotion path does not trust the supplied Gold evidence JSON by itself. It rebuilds Gold evaluation evidence from the original Gold release, answer-key-isolated inference pack, verified inference output, original portable candidate export, exact evaluation policy, and external reviewer public key. The freshly rebuilt evidence must be semantically identical to the supplied evidence or promotion fails closed.

Promotion re-applies the minimum case-count and every Gold quality threshold. The Promotion v4 receipt binds the Gold source audit, reviewer-signature audit, candidate-training binding, inference verification, evaluation report, and evaluation evidence digests.

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
  --gold-reviewer-public-key /secure/keys/gold-reviewer-public.pem \
  --output build/cyber-defense-promotion-v4.json
```

The repository production Gold evaluation policy requires at least 50 unseen HOLDOUT cases. Smaller policies are test-only and do not weaken the production policy file.

If any source artifact, reviewer signature, trust-root match, candidate provenance binding, inference verification, Gold threshold, or independent defense gate fails, promotion fails closed.

## 11. Portable artifact identity and migration note

Gold provenance identity is content-based rather than host-path-based. The Gold structural audit digest excludes diagnostic `release_dir`, and the inference-verification digest excludes diagnostic `output_dir`. The same sealed release and inference output therefore retain their provenance digests after relocation to another mount or evaluation host.

The trusted reviewer public key is external and is not part of the relocated artifact directory. Full-artifact relocation is valid only when the same trusted reviewer public key is supplied again.

Artifacts produced before the host-path-independent digest, signed-review, candidate-training binding, or deterministic candidate-export changes must not be mixed with the current Promotion v4 path. Regenerate the portable candidate export, Gold release audit/inference pack, signed Gold evidence, and downstream promotion evidence from the current code before a real HOLDOUT promotion attempt.
