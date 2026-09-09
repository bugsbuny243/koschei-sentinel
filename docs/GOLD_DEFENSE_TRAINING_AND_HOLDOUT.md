# Koschei Sentinel Gold Defense Training and HOLDOUT

This document defines the promotion-eligible Defense Reflex training and unseen evaluation trust path.

The only active training target is:

```text
Qwen/Qwen3.5-397B-A17B@8472618112abcbd45acbcdc58436aff4233c23f7
```

`3.5` is the Qwen generation. There is no separate active 35B development model. Historical dense-PEFT fixtures remain only for regression coverage and must not be presented as the active Sentinel training path.

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

The model-visible context is not trusted merely because a packet SHA verifies. It is independently re-derived from the source `CyberRangeScenario` and must contain exactly the intended visible fields:

```text
scenario_id
critical_entity_ids
graph_snapshots
```

Human review and signed-release construction both re-check this canonical view. A packet whose visible context was modified and then fully re-hashed is rejected.

A recursive fail-closed guard rejects answer-key or review-only fields anywhere inside model-visible data, including nested graph objects. This includes truth, expected sequence/interpretation, range reports, simulated outcomes, failure candidates, review results, and authorization fields.

## 3. Owner-rooted reviewer trust

The production trust root is the **Gold owner public key**, not a loose reviewer key. A reviewer key is accepted only when an owner-signed `GoldReviewerTrustPolicy` delegates the narrow `gold_review_signing_only` authority to that exact reviewer-key fingerprint.

Issue the policy on isolated owner/admin infrastructure:

```bash
sentinel-gold-reviewer-trust issue \
  --policy-id gold-reviewer-primary-v1 \
  --reviewer-public-key /secure/keys/gold-reviewer-public.pem \
  --owner-private-key /secure/keys/gold-owner-private.pem \
  --output /secure/policies/gold-reviewer-trust.json
```

Verify it independently:

```bash
sentinel-gold-reviewer-trust verify \
  --policy /secure/policies/gold-reviewer-trust.json \
  --reviewer-public-key /secure/keys/gold-reviewer-public.pem \
  --owner-public-key /secure/keys/gold-owner-public.pem
```

The owner private key stays off review, training, inference, and GPU hosts. The reviewer private key is restricted to trusted review/export infrastructure. Public verification material may be distributed to verifiers.

## 4. Signed human review

Human review is fail-closed. Reviewed targets and supporting evidence must exist in the scenario graph. `HOLDOUT` can never receive training authorization.

```text
TRAIN       → human-approved + training-authorized
VALIDATION  → human-approved + training-authorized
HOLDOUT     → human-approved + evaluation-authorized only
```

Production review signing requires the reviewer private key and the owner-rooted trust policy:

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

The proof binds reviewer identity, reviewer-key fingerprint, packet SHA, scenario ID, assigned split, and final review SHA.

## 5. Split-safe signed Gold release

A production release requires one valid trusted signature proof for every reviewed packet.

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

The builder re-checks canonical model-visible context even when packet and signature hashes are internally consistent. The signature-proof set must exactly match the release review set.

## 6. Gold release audit

Production verification uses structural and trusted-signature audits:

```bash
sentinel-defense-reflex-gold-audit \
  --release-dir build/gold-defense-release \
  --reviewer-public-key /secure/keys/gold-reviewer-public.pem \
  --reviewer-trust-policy /secure/policies/gold-reviewer-trust.json \
  --owner-public-key /secure/keys/gold-owner-public.pem \
  --output build/gold-defense-release-audit.json
```

The audit verifies split isolation, hashes, release identity, human-review status, HOLDOUT training exclusion, reviewer signatures, owner delegation, and the exact proof set. The signed-review audit identity is carried downstream.

## 7. Active 397B TRAIN and VALIDATION path

Promotion-eligible training uses the single-model Gold config:

```text
configs/training/cyber-sft.qwen3.5-397b-a17b.gold.example.json
```

Its explicit sources are:

```text
corpus_dir            = build/gold-defense-release/train
validation_corpus_dir = build/gold-defense-release/validation
validation_ratio      = 0.0
```

Materialize and seal the Megatron-SWIFT dataset without re-splitting:

```bash
sentinel-cyber-megatron-sft \
  --config configs/training/cyber-sft.qwen3.5-397b-a17b.gold.example.json \
  --materialize-dataset \
  --plan-output build/cyber-training/megatron/397b-gold-plan.json
```

The 397B renderer binds the source release/audit, TRAIN and VALIDATION bytes and counts, model ID/revision, rendered JSONL digests, tokenizer preflight, and run identity into the static plan. Any drift blocks execution before the paid process starts.

The paid run remains separately guarded by the distributed topology checks and explicit launch approval/session variables documented in `CYBER_SFT_397B_MEGATRON.md`.

## 8. Signed answer-key-isolated HOLDOUT pack

Never mount `holdout/cases.jsonl` on the inference host. Export only model-visible HOLDOUT input data.

```bash
sentinel-gold-holdout-eval export-inputs \
  --release-dir build/gold-defense-release \
  --output-dir build/gold-holdout-inference \
  --reviewer-private-key /secure/keys/gold-reviewer-private.pem \
  --reviewer-trust-policy /secure/policies/gold-reviewer-trust.json \
  --owner-public-key /secure/keys/gold-owner-public.pem \
  --signature-output build/gold-holdout-inference.signature.json
```

The pack contains exactly:

```text
gold-holdout-inference/
  inputs.jsonl
  manifest.json
```

The detached signature remains outside the pack. Export snapshots and re-audits the source release, recursively rejects answer-key/review-only fields, binds input/manifest/audit identities, signs the pack identity, verifies it, and publishes atomically. The reviewer private key must never be copied to the GPU host.

## 9. Legacy PEFT candidate path is not the 397B bridge

The existing `sentinel-cyber-sft-export`, `sentinel-cyber-sft-export-verify`, and `sentinel-gold-holdout-infer` path is retained because PR regression tests exercise a real fail-closed candidate/export/HOLDOUT contract for dense PEFT artifacts.

That path enforces exact candidate inventories, no symlinks, source stability, TRAIN/VALIDATION binding, adapter identity, detached HOLDOUT proof verification, deterministic generation, complete case accounting, offline verification, evaluation evidence, and relocation safety.

It **must not** be used to claim that a Qwen3.5-397B-A17B Megatron-SWIFT/MCore checkpoint has passed HOLDOUT. The current dense-PEFT candidate loader and portable adapter assumptions are not equivalent to a 397B Megatron checkpoint.

Historical 9B examples in tests or old artifacts are fixtures/provenance only. They are not an active development tier and cannot satisfy the single-model 397B policy.

## 10. Required 397B candidate/HOLDOUT bridge

Before a real 397B HOLDOUT or Promotion v4 attempt, Sentinel needs a dedicated content-addressed bridge from the 397B training output to the signed HOLDOUT path.

At minimum that bridge must bind and independently verify:

- exact `Qwen/Qwen3.5-397B-A17B` revision;
- Megatron-SWIFT/MCore model and LoRA checkpoint identity;
- the self-hashed 397B run-identity manifest;
- training config and final plan digest;
- signed Gold release/audit identity;
- rendered TRAIN and VALIDATION digests/counts;
- checkpoint inventory or content-addressed storage manifest with no path escape/symlink ambiguity;
- deterministic generation policy;
- signed answer-key-isolated HOLDOUT pack and owner-rooted reviewer trust;
- complete prediction/failure accounting; and
- an offline verification receipt consumed by evaluation and Promotion v4.

No free-form model revision and no loose run directory may substitute for this binding.

**Boundary:** 397B training may be authorized only after its Gold TRAIN/VALIDATION and cluster gates pass. A successful 397B training run remains promotion-blocked until this dedicated candidate/inference binding exists and passes its own regression gate.

## 11. Offline verification and evaluation trust contract

For the legacy PEFT fixture path, offline verification re-checks signed pack admission, candidate export, plan/receipt, generation policy, training config, attestation, prediction hashes, model/adapter identity, and complete case accounting. Extra files or symlinks fail closed.

Production evaluation evidence must bind all of the following, irrespective of the candidate storage backend:

- structural Gold release audit;
- exact signed-review audit;
- owner-signed reviewer trust policy and owner fingerprint;
- detached HOLDOUT pack proof;
- candidate TRAIN/VALIDATION binding;
- independently verified inference output;
- complete case accounting; and
- exact evaluation policy/report identity.

Malformed output is failure evidence, not something to repair silently. Missing HOLDOUT predictions fail evaluation. A candidate cannot be promoted from training loss alone.

The production policy requires at least **50 unseen HOLDOUT cases**. Smaller policies are test-only plumbing policies.

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

`build_cyber_defense_promotion_evidence_from_sources` is the authoritative production trust boundary. It verifies the owner-signed reviewer policy, revalidates source artifacts, rebuilds Gold evidence, and requires semantic equality with supplied evidence before promotion can become ready.

`build_cyber_defense_promotion_evidence` is only a low-level assembly primitive for already-verified evidence. It is not a replacement for source revalidation.

For the active 397B model, Promotion v4 remains fail-closed until the dedicated 397B candidate/HOLDOUT bridge described above produces source-rebuildable evidence.

## 13. Portability and migration

Artifact identity must be content-based rather than host-path-based. Release, HOLDOUT pack, detached proof, trust policy, owner public root, inference evidence, and the future 397B checkpoint manifest must survive relocation without weakening verification.

Legacy artifacts produced before canonical visible-context revalidation, exact inventories, signed-review binding, detached HOLDOUT signing, candidate-training binding, or owner-rooted reviewer trust must be regenerated. Do not mix legacy artifacts with a real Promotion v4 attempt.

CPU fixture tests may synthesize deterministic inference artifacts only to exercise plumbing. Synthetic predictions must never be used as real promotion evidence.

## 14. Current execution boundary

No real GPU HOLDOUT and no Promotion v4 should run until:

1. the complete repository suite and named Gold fail-closed regression gate execute successfully on a real runner;
2. a signed/audited Gold release with enough unseen HOLDOUT cases exists;
3. the 397B Megatron training output binding is implemented and verified; and
4. real 397B inference is connected to the signed answer-key-isolated HOLDOUT pack without exposing answer keys to the model host.

Until then, Sentinel remains fail-closed rather than treating historical PEFT plumbing as proof for the 397B model.
