# Koschei Sentinel — FROZEN

**Status:** FROZEN / research preserved  
**Frozen on:** 2026-08-24  
**Reason:** Preserve the security-intelligence work without spending compute or engineering capacity until the project has a justified restart condition.

Koschei Sentinel is not cancelled and no repository history, model contract, dataset/evaluation work, or checkpoint/provenance design should be deleted as part of this freeze.

## Freeze rules

- Do not launch paid GPU training or HOLDOUT inference.
- Do not merge unverified Sentinel pull requests solely to tidy the repository.
- Keep paid 397B launch gates fail-closed.
- Preserve branches, commits, manifests, hashes, provenance contracts, evaluation code, and Gold HOLDOUT work.
- Do not treat external models as Koschei Sentinel itself.
- On reactivation, resume from the exact recorded refs below rather than rebuilding infrastructure from scratch.

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

`main` at freeze decision before this marker:

- `7581292de5f7800e6f14f9670ec34015ff294a8d` — `Train only Qwen3.5-397B-A17B with Megatron-SWIFT (#68)`

Paid launch remains fail-closed behind the explicit 397B approval/session gates. No paid GPU job was started during this project freeze.

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

## OPEN WORK PRESERVED AT FREEZE

### PR #67

- Branch: `ci/gold-holdout-fail-closed-gate`
- Head: `fde029e0b4a164b5dc71a0301ce5abe14cbee95a`
- Purpose: harden Gold HOLDOUT provenance and Promotion v4 fail-closed chain.
- State at freeze: unmerged; GitHub Actions had not provided a trustworthy full runner execution proving the complete gate.

### PR #69

- Branch: `feat/397b-gold-holdout-binding`
- Head: `c3869449e87ecaa1e13a34e60d77c6f923bc9f00`
- Base: PR #67 branch.
- Purpose: bind Qwen3.5-397B Megatron checkpoints to signed Gold HOLDOUT and Promotion v5 evidence.
- State at freeze: draft/unmerged; intentionally stacked on #67.

The #69 head also preserves the in-progress production Gold capacity/release work. Do not assume that layer is complete without re-auditing it on restart.

## FAILURES / BLOCKERS

- No confirmed production Gold dataset with deterministic >=50 unseen HOLDOUT cases.
- GitHub-hosted CI had not produced a trustworthy complete Ruff + full pytest + named Gold gate execution for the open trust-chain work.
- Real multi-node 397B runtime compatibility has not been proven on the target GPU environment.
- Real 397B merged checkpoint, real HOLDOUT inference evidence, and Promotion v5 production evidence do not exist yet.

## ARTIFACT / REF CHECKPOINT

Restart anchors:

- `main`: `7581292de5f7800e6f14f9670ec34015ff294a8d` before this freeze marker commit.
- PR #67 head: `fde029e0b4a164b5dc71a0301ce5abe14cbee95a`.
- PR #69 head: `c3869449e87ecaa1e13a34e60d77c6f923bc9f00`.
- Base-model revision: `8472618112abcbd45acbcdc58436aff4233c23f7`.

## REACTIVATION CONDITIONS

Reactivate Sentinel only when at least one of these makes the work economically justified:

1. dedicated grant/cloud credits/sponsored compute sufficient for the required experiments;
2. proprietary real security-event data materially improves the training corpus;
3. a concrete product dependency requires Sentinel-specific intelligence;
4. funding explicitly covers the model/evaluation program.

## NEXT AFTER REACTIVATION

Do not start with a GPU run. First:

1. re-read repository, branches, commits, CI status, and this freeze checkpoint;
2. audit #67 and #69 against current `main`;
3. finish and execute real CI gates;
4. build a genuinely human-reviewed signed Gold corpus that yields >=50 deterministic HOLDOUT cases;
5. only then consider an explicitly funded/approved 397B run.
