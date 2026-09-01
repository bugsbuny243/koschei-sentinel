# Koschei Sentinel — REACTIVATED

**Status:** REACTIVATED / engineering resumed  
**Frozen on:** 2026-08-24  
**Reactivated on:** 2026-09-01  
**Reason for prior freeze:** Preserve the security-intelligence work without spending compute or engineering capacity until the project had a justified restart condition.

Koschei Sentinel was frozen without deleting repository history, model contracts, dataset/evaluation work, or checkpoint/provenance design. The project is now reactivated from the exact preserved refs below.

## Reactivation rules

- Do not launch paid GPU training or HOLDOUT inference without explicit approval and funded/free compute.
- Do not merge unverified Sentinel pull requests solely to tidy the repository.
- Keep paid 397B launch gates fail-closed.
- Preserve branches, commits, manifests, hashes, provenance contracts, evaluation code, and Gold HOLDOUT work.
- Do not treat external models as Koschei Sentinel itself.
- Resume from the exact recorded refs below rather than rebuilding infrastructure from scratch.

## MODEL STATE

Single long-term training target:

- Model: `Qwen/Qwen3.5-397B-A17B`
- Immutable revision: `8472618112abcbd45acbcdc58436aff4233c23f7`
- Architecture target: sparse/MoE, 397B total parameters, 17B activated parameters for this concrete base model.
- Training backend: Megatron-SWIFT / MCore
- Pinned `ms-swift==4.5.2`
- Planned training topology: 4 nodes x 8 GPUs, TP=4, PP=1, CP=1, EP=8, DP=8, LoRA, bf16.

No paid 397B training run has been launched.

## DATASET STATE

The repository contains deterministic dataset, provenance, review, Gold release/HOLDOUT, and evaluation infrastructure.

Production Gold remains blocked because there is no confirmed committed bulk human-reviewed, owner-trusted, signed Gold corpus that deterministically yields at least 50 unseen HOLDOUT cases. The existing seed curriculum has 32 synthetic, non-production scenarios and cannot satisfy that requirement.

## TRAINING STATE

`main` at freeze decision before the freeze marker:

- `7581292de5f7800e6f14f9670ec34015ff294a8d` — `Train only Qwen3.5-397B-A17B with Megatron-SWIFT (#68)`

Paid launch remains fail-closed behind the explicit 397B approval/session gates. No paid GPU job was started during the freeze.

## VERIFIED / PRESERVED

Preserve the following completed or substantially implemented infrastructure:

- deterministic dataset/readiness and provenance contracts;
- checkpoint/content hashing and immutable candidate identity;
- signed Gold review and owner-rooted reviewer trust chain;
- answer-key-isolated HOLDOUT packaging;
- fail-closed 397B candidate/HOLDOUT planning;
- SWIFT/vLLM/Ray worker contracts and offline replay verification;
- 397B-native Gold evaluation evidence;
- Promotion v4 regression path and Promotion v5 design/work;
- paid-compute safety gates.

## REACTIVATED WORK

### PR #67

- Branch: `ci/gold-holdout-fail-closed-gate`
- Preserved head at reactivation: `fde029e0b4a164b5dc71a0301ce5abe14cbee95a`
- Purpose: harden Gold HOLDOUT provenance and Promotion v4 fail-closed chain.
- Reactivation action: reopen and re-audit against current `main` before merge.

### PR #69

- Branch: `feat/397b-gold-holdout-binding`
- Preserved head at reactivation: `c3869449e87ecaa1e13a34e60d77c6f923bc9f00`
- Base: PR #67 branch.
- Purpose: bind Qwen3.5-397B Megatron checkpoints to signed Gold HOLDOUT and Promotion v5 evidence.
- Reactivation action: reopen as stacked draft and continue the in-progress production Gold capacity/release layer.

## FAILURES / BLOCKERS AT REACTIVATION

- No confirmed production Gold dataset with deterministic >=50 unseen HOLDOUT cases.
- GitHub-hosted CI had not produced a trustworthy complete Ruff + full pytest + named Gold gate execution for the trust-chain work.
- Real multi-node 397B runtime compatibility has not been proven on the target GPU environment.
- Real 397B merged checkpoint, real HOLDOUT inference evidence, and Promotion v5 production evidence do not exist yet.

## ARTIFACT / REF CHECKPOINT

Restart anchors:

- `main` before freeze marker: `7581292de5f7800e6f14f9670ec34015ff294a8d`.
- Freeze marker commit: `f3027e1cbaff0cad08b80614fd96d60d18fa867c`.
- Reactivation marker commit: `c6f0c1809d8c8cd5a990da3af11f8154e126f4c8`.
- PR #67 preserved head: `fde029e0b4a164b5dc71a0301ce5abe14cbee95a`.
- PR #69 preserved head: `c3869449e87ecaa1e13a34e60d77c6f923bc9f00`.
- Base-model revision: `8472618112abcbd45acbcdc58436aff4233c23f7`.

## NEXT

1. re-open #67 and #69;
2. audit #67 and #69 against current `main`;
3. finish the production Gold capacity/release layer on #69;
4. execute trustworthy Ruff + full pytest + named Gold gates before any merge;
5. build a genuinely human-reviewed signed Gold corpus that yields >=50 deterministic HOLDOUT cases;
6. only then consider an explicitly funded/approved 397B run.
