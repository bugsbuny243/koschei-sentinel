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
- Fabric integration starts in observe/contract-only mode and must not bypass model, Gold, HOLDOUT, promotion, or paid-compute gates.

## MODEL STATE

Koschei Sentinel long-term architecture target:

- Product target ID: `koschei-sentinel-moe-397b-a35b`
- Product identity: `KOSCHEI_SENTINEL`
- Architecture target: sparse/MoE, 397B total parameters, 35B active parameters.
- Target status: `ARCHITECTURE_TARGET`; this is not evidence that a trained 397B/35B production checkpoint exists.

External bootstrap / trainer-proof model:

- Model: `Qwen/Qwen3.5-397B-A17B`
- Immutable revision: `8472618112abcbd45acbcdc58436aff4233c23f7`
- Role: external bootstrap and trainer proof only; it does not define Koschei Sentinel product identity.
- Concrete external base architecture: 397B total / 17B active.

Training infrastructure retained from the earlier plan:

- Training backend: Megatron-SWIFT / MCore
- Pinned `ms-swift==4.5.2`
- Previously planned external-bootstrap topology: 4 nodes x 8 GPUs, TP=4, PP=1, CP=1, EP=8, DP=8, LoRA, bf16.

No paid 397B training run has been launched. Real candidate-bound training evidence, checkpoint identity, and independent HOLDOUT evaluation are still required before any production promotion claim.

## DATASET STATE

The repository contains deterministic dataset, provenance, review, Gold release/HOLDOUT, and evaluation infrastructure.

Production Gold remains blocked because there is no confirmed committed bulk human-reviewed, owner-trusted, signed Gold corpus that deterministically yields at least 50 unseen HOLDOUT cases. The existing seed curriculum has 32 synthetic, non-production scenarios and cannot satisfy that requirement.

## TRAINING STATE

`main` at freeze decision before the freeze marker:

- `7581292de5f7800e6f14f9670ec34015ff294a8d` — `Train only Qwen3.5-397B-A17B with Megatron-SWIFT (#68)`

That historical training branch concerns the external bootstrap model, not the final Koschei Sentinel 397B/35B product target. Paid launch remains fail-closed behind the explicit 397B approval/session gates. No paid GPU job was started during the freeze.

## VERIFIED / PRESERVED

Preserve the following completed or substantially implemented infrastructure:

- deterministic dataset/readiness and provenance contracts;
- checkpoint/content hashing and immutable candidate identity;
- signed Gold review and owner-rooted reviewer trust chain;
- answer-key-isolated HOLDOUT packaging;
- fail-closed 397B candidate/HOLDOUT planning;
- SWIFT/vLLM/Ray worker contracts and offline replay verification;
- 397B-native Gold evaluation evidence infrastructure;
- Promotion v4 regression path and Promotion v5 design/work;
- paid-compute safety gates;
- Web4 security-event and agent-trust-chain research infrastructure;
- PQ research/evidence/watch infrastructure.

The presence of these files, schemas, gates, or research paths is not by itself proof of a trained or production-promoted Sentinel model.

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
- Purpose: bind the external Qwen3.5-397B bootstrap/trainer-proof work to signed Gold HOLDOUT and Promotion v5 evidence without redefining the final Sentinel architecture target.
- Reactivation action: reopen as stacked draft and continue the in-progress production Gold capacity/release layer.

## FAILURES / BLOCKERS AT REACTIVATION

- No confirmed production Gold dataset with deterministic >=50 unseen HOLDOUT cases.
- Real multi-node 397B runtime compatibility has not been proven on the target GPU environment.
- A real Koschei Sentinel 397B/35B trained checkpoint, independent HOLDOUT inference evidence, and production promotion evidence do not exist yet.
- External 397B-A17B bootstrap/trainer proof must not be reported as the Koschei Sentinel 397B/35B product model.

The integration branch has since produced trustworthy Ruff, full pytest, named Gold/Web4/PQ gate executions in CI, but that does not remove the model/data/runtime blockers above.

## ARTIFACT / REF CHECKPOINT

Restart anchors:

- `main` before freeze marker: `7581292de5f7800e6f14f9670ec34015ff294a8d`.
- Freeze marker commit: `f3027e1cbaff0cad08b80614fd96d60d18fa867c`.
- Reactivation marker commit: `c6f0c1809d8c8cd5a990da3af11f8154e126f4c8`.
- PR #67 preserved head: `fde029e0b4a164b5dc71a0301ce5abe14cbee95a`.
- PR #69 preserved head: `c3869449e87ecaa1e13a34e60d77c6f923bc9f00`.
- External bootstrap model revision: `8472618112abcbd45acbcdc58436aff4233c23f7`.
- Koschei Sentinel architecture target: `configs/model/sentinel-moe-397b-a35b.target.json`.

## NEXT

1. keep the current full CI / Gold / Web4 / PQ gates green on integration work;
2. finish the production Gold capacity/release layer without weakening HOLDOUT isolation;
3. build a genuinely human-reviewed signed Gold corpus that yields >=50 deterministic HOLDOUT cases;
4. prove runtime/trainer compatibility with explicitly authorized compute;
5. bind real candidate checkpoint identity, data version, tokenizer, training receipt and independent HOLDOUT result;
6. only then consider an explicitly funded/approved 397B run or production promotion.
