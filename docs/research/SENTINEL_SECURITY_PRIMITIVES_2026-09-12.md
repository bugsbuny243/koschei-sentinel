# Sentinel Security Primitives Intake — 2026-09-12

Purpose: convert current Web3→Web10 research signals into defensive Sentinel training/evaluation primitives without changing Sentinel Driver.

## 1. Ethereum L1 hardening

Current signal:
- Ethereum Protocol’s September 2026 priorities explicitly target full L1 post-quantum resistance across execution, consensus, and data by December 2029.
- Hegotá scope emphasizes Frame Transactions / native account abstraction, censorship resistance, transaction assertions, and the migration path away from vulnerable legacy key authentication.
- zkEVM work is moving execution proofs toward mandatory protocol use, with formal verification becoming cross-cutting infrastructure.

Sentinel primitives:
- cryptographic-agility reasoning
- signature-migration risk analysis
- account-abstraction trust-boundary review
- transaction-assertion / post-state invariant review
- consensus/data/execution dependency mapping
- proof-system assumption tracking

Eval targets:
- distinguish application risk from protocol risk
- detect unsafe auth fallback / legacy-key retention
- identify when PQ migration creates compatibility or downgrade risk
- reason about proof-system assumptions without treating proof existence as correctness

## 2. Solana / Agave rollout security

Current signal:
- Agave v4.3 is in staged September 2026 rollout: mainnet-beta volunteer adoption began at 10% stake on Sep 8, with 25% targeted Sep 14 and general adoption targeted Sep 21.

Sentinel primitives:
- release-diff security triage
- validator/runtime change review
- staged-rollout risk reasoning
- cluster-version divergence detection
- feature-activation dependency analysis

Eval targets:
- identify rollout risks without assuming unreleased activation is live
- separate testnet/devnet/mainnet evidence
- reason about validator-version skew and feature gates

## 3. Cross-chain / bridge trust analysis

Current signal:
- Cross-chain security fundamentally depends on who attests source-chain state to a destination chain and what must be compromised to forge that attestation.

Sentinel primitives:
- trust-model extraction
- validator/oracle/relayer quorum reasoning
- replay/domain-separation review
- message ordering / finality assumptions
- asset-accounting invariant review
- upgrade-key / admin-key risk

Eval targets:
- produce a trust graph before severity claims
- state minimum compromise set needed for forged cross-chain state
- distinguish protocol failure from integration/configuration failure
- recommend evidence needed to confirm an exploit path

## 4. TEE / confidential-compute attestation

Current signal:
- Confidential-computing ecosystems increasingly treat remote attestation as the trust anchor for autonomous systems.

Sentinel primitives:
- attestation-chain validation reasoning
- measurement / expected-code identity mapping
- freshness / replay-risk checks
- key-release policy review
- TCB / supply-chain assumption tracking

Eval targets:
- never treat “runs in a TEE” as sufficient proof of safety
- require verifiable measurement and policy binding
- identify stale-attestation, rollback, signer, and dependency risks

## 5. ZK / verifiable-compute security

Current signal:
- Ethereum is pushing zkEVM proving toward protocol-critical use; EF allocations also support operationally resilient proving infrastructure.

Sentinel primitives:
- statement-vs-witness boundary review
- circuit/spec equivalence reasoning
- verifier correctness review
- trusted-setup / cryptographic-assumption inventory
- prover operational-resilience review
- proof version / parameter binding

Eval targets:
- distinguish “valid proof” from “correctly specified statement”
- catch verifier/circuit mismatch and unconstrained-value classes conceptually
- require version/parameter provenance
- include availability and prover-centralization risks where relevant

## 6. Agent wallet / delegated-account security

Current signal:
- EIP-7702-style delegated account behavior enables batching, gas sponsorship, session-key-like flows and smart-account behavior, but delegated code becomes a direct security boundary.
- Native account abstraction is being advanced partly as a path to cryptographic agility.

Sentinel primitives:
- delegation-scope review
- session-key lifetime / privilege reasoning
- signer hierarchy and recovery-path review
- delegated-code provenance
- approval / transfer-effect assertions
- revoke / reset path analysis

Eval targets:
- detect over-broad delegated permissions
- identify unsafe master-key fallback
- test expiry / revocation / replay / chain-domain handling
- verify that requested wallet effects match observed post-state

## 7. Curriculum insertion

Add these case families to future expert-reviewed training releases:
- `pq_migration_and_crypto_agility`
- `account_abstraction_and_delegation`
- `solana_runtime_and_release_security`
- `cross_chain_trust_graphs`
- `tee_attestation_and_tcb`
- `zk_statement_and_verifier_security`
- `wallet_effect_and_post_state_assertions`

Each example must contain:
1. evidence packet,
2. trust assumptions,
3. known facts,
4. hypotheses,
5. missing evidence,
6. bounded conclusion,
7. remediation or verification next step,
8. confidence ceiling.

## 8. Research ingestion rule

Sentinel should rank inputs by evidence quality:
1. deployed protocol / implementation evidence,
2. standards-track or official protocol documentation,
3. audited public code / reproducible research,
4. experimental draft / devnet evidence,
5. conceptual Web4–Web10 naming or marketing.

Do not promote speculative generation labels into model knowledge as if they were protocol facts.

## 9. Driver boundary

Sentinel Driver remains unchanged by this intake. Research material feeds curriculum, datasets, evals, and future model training only.
