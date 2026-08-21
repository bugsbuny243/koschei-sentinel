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

## 3. Owner-rooted reviewer trust

The external production trust root is the **Gold owner public key**, not a loose reviewer public key. A reviewer key is accepted for production only when an owner-signed `GoldReviewerTrustPolicy` delegates the narrow `gold_review_signing_only` authority to that exact reviewer-key fingerprint.

Issue the policy on isolated owner/admin infrastructure:

```bash
sentinel-gold-reviewer-trust issue \
  --policy-id gold-reviewer-primary-v1 \
  --reviewer-public-key /secure/keys/gold-reviewer-public.pem \
  --owner-private-key /secure/keys/gold-owner-private.pem \
  --output /secure/policies/gold-reviewer-trust.json
```

Verify it independently before use:

```bash
sentinel-gold-reviewer-trust verify \
  --policy /secure/policies/gold-reviewer-trust.json \
  --reviewer-public-key /secure/keys/gold-reviewer-public.pem \
  --owner-public-key /secure/keys/gold-owner-public.pem
```

The owner private key is only needed to issue trust policy and must stay off review, training, inference, and GPU hosts. The reviewer private key is restricted to trusted review/export infrastructure. The owner public key, reviewer public key, and owner-signed reviewer policy are verification artifacts and may be distributed to production verifiers.

The policy is fail-closed: its digest, owner fingerprint, reviewer fingerprint, domain-separated Ed25519 owner signature, and restricted authority all verify before a reviewer key is admitted.

## 4. Signed human review

Human review is fail-closed. Reviewed targets and supporting evidence must exist in the scenario graph. `HOLDOUT` can never receive training authorization.

```text
TRAIN       → human-approved + training-authorized
VALIDATION  → human-approved + training-authorized
HOLDOUT     → human-approved + evaluation-authorized only
```

Production review signing requires both the reviewer private key and the owner-rooted trust policy:

```bash
sentinel-defense-reflex-gold-review \
  --packet build/gold-review-packets/case-001.json \
  --scenario build/cyber-range/case-001.json \
  --review-spec /secure/gold-reviews/case-001.review.json \
  --reviewer-private-key /secure/keys/gold-reviewer-private.pem \
  --reviewer-trust-policy /secure/policies/gold-reviewer-trust.json \
  --owner-public-key /secure/keys/gold-owner-public.pem \
  --output build/gold-reviewed/case-001.json \
  --signature-output build/gold-reviewed/case-001.signature.json
```

The human-review proof binds reviewer identity, reviewer-key fingerprint, packet SHA, scenario ID, assigned split, and final review SHA. The reviewer private key is not considered trusted merely because it is present; its public half must match the owner-signed policy.

## 5. Split-safe signed Gold release

The production release requires one valid trusted signature proof for every reviewed packet.

```bash
sentinel-defense-reflex-gold-release \
  --scenario build/cyber-range/case-001.json \
  --packet build/gold-review-packets/case-001.json \
  --reviewed build/gold-reviewed/case-001.json \
  --review-signature build/gold-reviewed/case-001.signature.json \
  --reviewer-public-key /secure/keys/gold-reviewer-public.pem \
  --reviewer-trust-policy /secure/policies/gold-reviewer-trust.json \
  --owner-public-key /secure/keys/gold-owner-public.pem \
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

The signed-release builder re-checks canonical model-visible context even when the reviewed packet and signature are otherwise internally consistent. The exact signature-proof set must match the exact release review set.

## 6. Gold release audit

Production verification uses both structural and trusted-signature audits:

```bash
sentinel-defense-reflex-gold-audit \
  --release-dir build/gold-defense-release \
  --reviewer-public-key /secure/keys/gold-reviewer-public.pem \
  --reviewer-trust-policy /secure/policies/gold-reviewer-trust.json \
  --owner-public-key /secure/keys/gold-owner-public.pem \
  --output build/gold-defense-release-audit.json
```

The structural audit checks split isolation, hashes, release identity, human-review-only status, and HOLDOUT training exclusion. The signature audit verifies every review proof with the reviewer key **after** that key has been admitted through the owner-signed trust policy. The exact review-signature proof-set audit SHA becomes downstream HOLDOUT identity.

## 7. Explicit TRAIN and VALIDATION

Promotion-eligible Defense Reflex training uses explicit TRAIN and VALIDATION directories. The trainer does not re-split them.

```text
corpus_dir            = build/gold-defense-release/train
validation_corpus_dir = build/gold-defense-release/validation
validation_ratio      = 0.0
```

TRAIN/VALIDATION hashes and counts are bound through the training plan, source binding, run attestation, adapter metadata, and candidate export. `sentinel-cyber-sft --execute` revalidates those sources before entering the GPU executor, and the trainer independently re-checks them again.

## 8. Portable candidate export

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

The export is staged and independently verified before atomic publication. It rejects stale provenance, source drift, count drift, symlinks, unexpected files, duplicate/non-portable adapter paths, empty adapters, and incomplete PEFT adapters.

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

## 9. Export and sign answer-key-isolated HOLDOUT inputs

Never mount `holdout/cases.jsonl` on the inference host. Export only the model-visible HOLDOUT input pack.

Production export requires an owner-trusted reviewer private key because the detached pack proof is signed under a separate domain from human-review signatures:

```bash
sentinel-gold-holdout-eval export-inputs \
  --release-dir build/gold-defense-release \
  --output-dir build/gold-holdout-inference \
  --reviewer-private-key /secure/keys/gold-reviewer-private.pem \
  --reviewer-trust-policy /secure/policies/gold-reviewer-trust.json \
  --owner-public-key /secure/keys/gold-owner-public.pem \
  --signature-output build/gold-holdout-inference.signature.json
```

The pack directory is sealed and contains exactly:

```text
gold-holdout-inference/
  inputs.jsonl
  manifest.json
```

The detached signature stays outside the pack. The exporter snapshots and re-audits the Gold release, rejects answer-key/review-only fields recursively, binds the exact inputs SHA, raw manifest SHA, source Gold structural-audit SHA, and signed-review proof-set audit SHA, signs that identity, verifies it, publishes the detached proof first, and publishes the final pack directory last.

The reviewer private key must never be copied to the GPU host.

## 10. Run the trained adapter against signed HOLDOUT

Planning and execution require the detached pack proof plus the complete owner-rooted reviewer trust chain.

```bash
sentinel-gold-holdout-infer \
  --inference-pack build/gold-holdout-inference \
  --inference-pack-signature build/gold-holdout-inference.signature.json \
  --reviewer-public-key /secure/keys/gold-reviewer-public.pem \
  --reviewer-trust-policy /secure/policies/gold-reviewer-trust.json \
  --owner-public-key /secure/keys/gold-owner-public.pem \
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
  --reviewer-trust-policy /secure/policies/gold-reviewer-trust.json \
  --owner-public-key /secure/keys/gold-owner-public.pem \
  --candidate-export build/cyber-training/exports/qwen35-9b-gold-defense-v1 \
  --model-ref koschei-sentinel:qwen35-9b-gold-defense-v1 \
  --generation-policy configs/training/gold-holdout-generation-policy.v1.json \
  --output-dir build/gold-holdout-inference-output \
  --execute
```

Before candidate admission or model loading, the CLI preflights the pack and verifies the detached proof through the owner-trusted reviewer key. Rewriting inputs, manifest, proof metadata, or internal SHA values cannot authorize a different reviewer without a valid owner-signed trust policy.

Generation is deterministic. Sampling is disabled. Malformed model output is recorded as failure rather than repaired. The inference result is written to hidden staging, independently offline-verified, and published atomically only when valid.

## 11. Offline verification

```bash
sentinel-gold-holdout-infer-verify \
  --inference-pack build/gold-holdout-inference \
  --inference-pack-signature build/gold-holdout-inference.signature.json \
  --reviewer-public-key /secure/keys/gold-reviewer-public.pem \
  --reviewer-trust-policy /secure/policies/gold-reviewer-trust.json \
  --owner-public-key /secure/keys/gold-owner-public.pem \
  --output-dir build/gold-holdout-inference-output \
  --candidate-export build/cyber-training/exports/qwen35-9b-gold-defense-v1
```

Verification re-checks the owner-rooted signed pack, candidate export, plan, receipt, generation policy, training config, run attestation, candidate-export verification snapshot, prediction hashes, model/adapter identity, and complete case accounting. Extra files or symlinks fail closed.

## 12. Evaluate and build production evidence

The `sentinel-gold-holdout-eval evaluate` subcommand is a direct/research evaluation utility. It can score supplied predictions, but by itself it does **not** establish the owner-rooted pack/candidate/inference provenance required by Promotion v4. Production flow uses `evaluate-output` and then source-rebuilt evidence.

```bash
sentinel-gold-holdout-eval evaluate-output \
  --release-dir build/gold-defense-release \
  --inference-pack build/gold-holdout-inference \
  --inference-pack-signature build/gold-holdout-inference.signature.json \
  --reviewer-public-key /secure/keys/gold-reviewer-public.pem \
  --reviewer-trust-policy /secure/policies/gold-reviewer-trust.json \
  --owner-public-key /secure/keys/gold-owner-public.pem \
  --inference-output build/gold-holdout-inference-output \
  --candidate-export build/cyber-training/exports/qwen35-9b-gold-defense-v1 \
  --policy configs/training/gold-holdout-evaluation-policy.v1.json \
  --output build/gold-holdout-report.json
```

Production evidence creation re-runs the source checks and requires the complete trust chain:

```bash
sentinel-gold-holdout-evidence \
  --release-dir build/gold-defense-release \
  --inference-pack build/gold-holdout-inference \
  --inference-pack-signature build/gold-holdout-inference.signature.json \
  --inference-output build/gold-holdout-inference-output \
  --candidate-export build/cyber-training/exports/qwen35-9b-gold-defense-v1 \
  --reviewer-public-key /secure/keys/gold-reviewer-public.pem \
  --reviewer-trust-policy /secure/policies/gold-reviewer-trust.json \
  --owner-public-key /secure/keys/gold-owner-public.pem \
  --policy configs/training/gold-holdout-evaluation-policy.v1.json \
  --output build/gold-holdout-evidence.json
```

Production Gold evidence requires all of these to agree:

- structural Gold release audit;
- exact signed-review audit;
- detached HOLDOUT pack proof and its bound review-audit SHA;
- candidate TRAIN/VALIDATION binding;
- offline inference verification and complete case accounting;
- exact evaluation policy/report identity;
- owner-signed reviewer trust-policy verification.

Its self-hash binds `reviewer_trust_policy_sha256` and `owner_key_fingerprint` alongside the review, pack, candidate, inference, and evaluation identities. Changing the delegated policy therefore changes Gold evidence identity even when the reviewer key is unchanged.

## 13. Promotion v4

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
  --gold-reviewer-trust-policy /secure/policies/gold-reviewer-trust.json \
  --gold-owner-public-key /secure/keys/gold-owner-public.pem \
  --output build/cyber-defense-promotion-v4.json
```

Promotion v4 fails closed even when a caller imports the low-level Python builder directly. The low-level builder requires the reviewer public key, owner-signed reviewer trust policy, and owner public key; it cryptographically verifies the policy's domain-separated owner signature, owner/reviewer fingerprints, active state, and restricted authority. It then requires the Gold evidence `reviewer_trust_policy_sha256` and `owner_key_fingerprint` to match that verified policy exactly. Self-hashing forged trust fields therefore cannot authorize Promotion v4.

The source-rebuild path performs the same owner-policy verification **before** rebuilding Gold evidence from release, pack, candidate, or inference artifacts. A wrong owner root fails before the trusted source rebuild begins. Fresh rebuilt evidence must still match the supplied evidence exactly.

The Promotion receipt binds Gold evidence SHA, so the owner policy digest and owner-key fingerprint are transitively part of Promotion identity together with structural audit, review-signature audit, candidate TRAIN/VALIDATION binding, HOLDOUT pack proof, inference verification, evaluation policy/report, and the independent range gates.

The repository production Gold policy requires at least 50 unseen HOLDOUT cases. Smaller policies are test-only plumbing policies.

## 14. Portability, migration, and real GPU boundary

Artifact identity is content-based rather than host-path-based. Sealed release, candidate export, HOLDOUT pack, detached proof, inference output, reviewer trust policy, and owner public trust root may be supplied again after relocation without weakening verification.

Legacy artifacts produced before canonical visible-context revalidation, exact file inventories, signed-review audit binding, detached HOLDOUT signing, candidate-training binding, or owner-rooted reviewer trust must be regenerated. Do not mix legacy artifacts with a real Promotion v4 attempt.

CPU fixture tests may synthesize deterministic inference artifacts solely to exercise plumbing. A real HOLDOUT must use actual inference from the verified trained adapter. Synthetic predictions must never be used as real promotion evidence.

No real GPU HOLDOUT and no Promotion v4 should run until the complete repository suite and named Gold fail-closed regression gate actually execute successfully.
