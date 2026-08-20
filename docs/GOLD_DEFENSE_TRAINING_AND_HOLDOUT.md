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

## 2. Canonical model-visible context

The model-visible context is not trusted merely because a packet SHA verifies. It is independently re-derived from the source `CyberRangeScenario` and must contain exactly:

```text
scenario_id
critical_entity_ids
graph_snapshots
```

Human review and signed-release construction both re-check this canonical view. A packet whose context was modified and then fully re-hashed is rejected.

A recursive fail-closed guard also rejects answer-key or review-only fields anywhere inside model-visible data, including nested graph objects. Examples include `truth`, `expected_sequence`, `expected_interpretation`, `range_report`, simulated outcomes, failure candidates, review results, and authorization fields.

## 3. Signed human review

Human review is fail-closed. Reviewed targets and supporting evidence must exist in the scenario graph. `HOLDOUT` can never receive training authorization.

```text
TRAIN       → human-approved + training-authorized
VALIDATION  → human-approved + training-authorized
HOLDOUT     → human-approved + evaluation-authorized only
```

Production reviews are Ed25519-signed with a domain-separated Gold-review message. The proof binds reviewer identity, trusted key fingerprint, packet SHA, scenario ID, assigned split, and final review SHA.

```bash
sentinel-defense-reflex-gold-review \
  --packet build/gold-review-packets/case-001.json \
  --scenario build/cyber-range/case-001.json \
  --review-spec /secure/gold-reviews/case-001.review.json \
  --reviewer-private-key /secure/keys/gold-reviewer-private.pem \
  --output build/gold-reviewed/case-001.json \
  --signature-output build/gold-reviewed/case-001.signature.json
```

The private key must stay outside Git and model/GPU artifacts. The public key is the external trust root.

## 4. Split-safe signed Gold release

The production release requires one trusted signature proof for every reviewed packet.

```bash
sentinel-defense-reflex-gold-release \
  --scenario build/cyber-range/case-001.json \
  --packet build/gold-review-packets/case-001.json \
  --reviewed build/gold-reviewed/case-001.json \
  --review-signature build/gold-reviewed/case-001.signature.json \
  --reviewer-public-key /secure/keys/gold-reviewer-public.pem \
  --output-dir build/gold-defense-release
```

The release is physically split:

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

The signed-release builder re-checks canonical model-visible context even if the reviewed packet and its Ed25519 signature are otherwise cryptographically valid. The exact signature-proof set must match the exact release review set.

## 5. Gold release audit

Production verification uses both structural and signature audits:

```bash
sentinel-defense-reflex-gold-audit \
  --release-dir build/gold-defense-release \
  --reviewer-public-key /secure/keys/gold-reviewer-public.pem \
  --output build/gold-defense-release-audit.json
```

The structural audit checks split isolation, hashes, release identity, human-review-only status and HOLDOUT training exclusion. The signature audit verifies every Ed25519 proof against the external reviewer public key and computes an audit SHA for the exact signed-review proof set.

That review-signature-audit SHA becomes part of downstream HOLDOUT pack identity.

## 6. Explicit TRAIN and VALIDATION

Promotion-eligible Defense Reflex training uses explicit TRAIN and VALIDATION directories. The trainer does not re-split them.

```text
corpus_dir            = build/gold-defense-release/train
validation_corpus_dir = build/gold-defense-release/validation
validation_ratio      = 0.0
```

TRAIN/VALIDATION hashes and counts are bound through the training plan, source binding, run attestation and adapter metadata. `sentinel-cyber-sft --execute` revalidates those sources before entering the GPU executor, and the trainer independently re-checks them again.

## 7. Portable candidate export

A real HOLDOUT never consumes a loose run directory. First build the promotion-eligible portable candidate export:

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

The export is staged and independently verified before atomic publication. It rejects stale provenance, source drift, count drift, symlinks, unexpected files, duplicate/non-portable adapter paths, empty adapters and incomplete PEFT adapters.

A real adapter must include the PEFT core artifacts:

```text
adapter/adapter_config.json
adapter/adapter_model.safetensors
```

Verify the portable export independently:

```bash
sentinel-cyber-sft-export-verify \
  --export-dir build/cyber-training/exports/qwen35-9b-gold-defense-v1
```

## 8. Export and sign answer-key-isolated HOLDOUT inputs

Never mount `holdout/cases.jsonl` on the inference host. Export only the model-visible HOLDOUT input pack.

Production export requires the trusted reviewer private key because the detached pack proof is signed under a separate domain from human-review signatures:

```bash
sentinel-gold-holdout-eval export-inputs \
  --release-dir build/gold-defense-release \
  --output-dir build/gold-holdout-inference \
  --reviewer-private-key /secure/keys/gold-reviewer-private.pem \
  --signature-output build/gold-holdout-inference.signature.json
```

The pack directory is sealed and contains exactly two files:

```text
gold-holdout-inference/
  inputs.jsonl
  manifest.json
```

The detached signature stays outside the pack so the pack inventory remains exact.

The exporter:

- audits the signed Gold release;
- creates the pack in staging;
- requires the exact two-file inventory with no symlinks;
- recursively rejects answer-key/review-only fields;
- binds the exact `inputs.jsonl` SHA and raw manifest SHA;
- binds the source Gold structural-audit SHA;
- binds the exact signed-review proof-set audit SHA;
- signs that identity with domain-separated Ed25519;
- verifies the new proof;
- publishes the detached proof first;
- publishes the final pack directory last.

Therefore a final pack path is never intentionally exposed without an already-published trusted detached proof.

The reviewer private key belongs only on trusted review/export infrastructure. It must never be copied to the GPU host. The reviewer **public** key is safe and required there.

## 9. Run the trained adapter against signed HOLDOUT

Both planning and execution require the detached pack proof and trusted reviewer public key.

```bash
sentinel-gold-holdout-infer \
  --inference-pack build/gold-holdout-inference \
  --inference-pack-signature build/gold-holdout-inference.signature.json \
  --reviewer-public-key /secure/keys/gold-reviewer-public.pem \
  --candidate-export build/cyber-training/exports/qwen35-9b-gold-defense-v1 \
  --model-ref koschei-sentinel:qwen35-9b-gold-defense-v1 \
  --generation-policy configs/training/gold-holdout-generation-policy.v1.json \
  --plan-output build/gold-holdout-inference-plan.json
```

GPU execution uses the same authenticated inputs:

```bash
sentinel-gold-holdout-infer \
  --inference-pack build/gold-holdout-inference \
  --inference-pack-signature build/gold-holdout-inference.signature.json \
  --reviewer-public-key /secure/keys/gold-reviewer-public.pem \
  --candidate-export build/cyber-training/exports/qwen35-9b-gold-defense-v1 \
  --model-ref koschei-sentinel:qwen35-9b-gold-defense-v1 \
  --generation-policy configs/training/gold-holdout-generation-policy.v1.json \
  --output-dir build/gold-holdout-inference-output \
  --execute
```

Before candidate admission or model loading, the CLI recursively preflights the pack and verifies its detached Ed25519 proof. Rewriting `inputs.jsonl`, the manifest, or every internal SHA cannot produce a valid proof without the trusted private key.

Generation is deterministic. Sampling is disabled. The model must emit exactly one JSON object with `interpretation` and `defense_sequence`; malformed output is recorded as an inference failure rather than repaired.

The inference result is written to a hidden staging directory, independently offline-verified, and published atomically only when valid.

## 10. Offline verification

```bash
sentinel-gold-holdout-infer-verify \
  --inference-pack build/gold-holdout-inference \
  --inference-pack-signature build/gold-holdout-inference.signature.json \
  --reviewer-public-key /secure/keys/gold-reviewer-public.pem \
  --output-dir build/gold-holdout-inference-output \
  --candidate-export build/cyber-training/exports/qwen35-9b-gold-defense-v1
```

Verification re-checks the signed pack, candidate export, plan, receipt, generation policy, training config, run attestation, candidate-export verification snapshot, prediction hashes, model/adapter identity and complete case accounting.

The inference output itself also has an exact file inventory. Extra files or symlinks fail closed.

## 11. Evaluate and build signed evidence

Production evaluation requires the same detached pack proof:

```bash
sentinel-gold-holdout-eval evaluate-output \
  --release-dir build/gold-defense-release \
  --inference-pack build/gold-holdout-inference \
  --inference-pack-signature build/gold-holdout-inference.signature.json \
  --reviewer-public-key /secure/keys/gold-reviewer-public.pem \
  --inference-output build/gold-holdout-inference-output \
  --candidate-export build/cyber-training/exports/qwen35-9b-gold-defense-v1 \
  --policy configs/training/gold-holdout-evaluation-policy.v1.json \
  --output build/gold-holdout-report.json
```

Evidence creation re-runs the source checks and requires the detached proof explicitly:

```bash
sentinel-gold-holdout-evidence \
  --release-dir build/gold-defense-release \
  --inference-pack build/gold-holdout-inference \
  --inference-pack-signature build/gold-holdout-inference.signature.json \
  --inference-output build/gold-holdout-inference-output \
  --candidate-export build/cyber-training/exports/qwen35-9b-gold-defense-v1 \
  --reviewer-public-key /secure/keys/gold-reviewer-public.pem \
  --policy configs/training/gold-holdout-evaluation-policy.v1.json \
  --output build/gold-holdout-evidence.json
```

Production Gold evidence requires all of these to agree:

- structural Gold release audit;
- fresh exact signed-review audit;
- the review-signature-audit SHA signed inside the HOLDOUT pack proof;
- detached pack proof SHA;
- candidate TRAIN/VALIDATION binding;
- inference verification;
- evaluation policy and evaluation report.

A pack proof signed by the trusted key but issued against a different signed-review audit is rejected.

## 12. Promotion v4

Promotion v4 requires four independent gate families:

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

The production promotion path rebuilds Gold evidence from source artifacts and requires exact semantic equality with the supplied evidence.

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
  --gold-inference-pack-signature build/gold-holdout-inference.signature.json \
  --gold-inference-output build/gold-holdout-inference-output \
  --gold-candidate-export build/cyber-training/exports/qwen35-9b-gold-defense-v1 \
  --gold-reviewer-public-key /secure/keys/gold-reviewer-public.pem \
  --output build/cyber-defense-promotion-v4.json
```

The Promotion v4 receipt binds:

- source Gold structural-audit SHA;
- exact review-signature-audit SHA;
- candidate TRAIN/VALIDATION binding SHA;
- detached HOLDOUT pack proof SHA;
- inference-verification SHA;
- evaluation policy/report/evidence identities;
- all independent range gates.

The repository production Gold policy requires at least 50 unseen HOLDOUT cases. Smaller policies are test-only plumbing policies.

## 13. Portability and migration

Artifact identity is content-based rather than host-path-based. Sealed release, candidate export, HOLDOUT pack, detached proof and inference output may be relocated without changing their content identities.

The detached pack proof remains valid after relocation because it binds content hashes, not mount paths. The same external reviewer public key must be supplied again.

Legacy artifacts produced before canonical visible-context revalidation, exact file inventories, signed-review audit binding, detached HOLDOUT pack signing, candidate-training binding or current Promotion v4 contracts must be regenerated. Do not mix legacy artifacts with a real promotion attempt.

## 14. Real GPU boundary

CPU fixture tests may synthesize deterministic inference artifacts solely to exercise plumbing. A real HOLDOUT must use actual inference from the verified trained adapter. Synthetic predictions must never be used as real promotion evidence.

No real GPU HOLDOUT and no Promotion v4 should run until the complete repository CI and named Gold fail-closed regression gate execute successfully.
