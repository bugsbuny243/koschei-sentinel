# Sentinel Daily Research — 2026-09-12 Addendum 5

## Scope

This addendum deepens the September 12 research map around wallet capability negotiation, account abstraction, data availability, native/based rollup sequencing, and chain-scoped authorization.

Research labels remain evidence metadata. Sentinel must not treat a proposal, roadmap item, draft, or capability advertisement as deployment truth without independent evidence.

## 1. Wallet capability negotiation is now a first-class security boundary

EIP-5792 is Final and defines wallet_sendCalls, wallet_getCallsStatus, wallet_showCallsStatus, and wallet_getCapabilities.

Security primitives for Sentinel:
- capability discovery is chain-scoped;
- live wallet capability responses take precedence over cached or out-of-band capability metadata;
- unsupported non-optional capabilities must fail closed;
- atomicRequired=true and atomicRequired=false have materially different security properties;
- a batch can have partial on-chain effects when execution is non-atomic;
- receipt/log scoping must exclude unrelated bundled operations;
- duplicate/guessable batch identifiers and privacy leakage are wallet-layer risks.

Sentinel evaluation cases:
- app assumes atomic execution when wallet reports unsupported;
- capability advertised on chain A is incorrectly reused on chain B;
- paymaster/session-key capability is cached after revocation or downgrade;
- partial batch execution is summarized as full success;
- unrelated ERC-4337 bundle logs are incorrectly attributed to the user operation;
- optional capability silently disappears and materially changes post-state.

## 2. Account abstraction must be modeled as multiple trust layers

Ethereum currently supports parallel account-abstraction paths: EIP-7702 delegated EOA code and ERC-4337 UserOperations/bundlers/paymasters.

Sentinel must separate:
1. account validation code;
2. bundler behavior and simulation;
3. paymaster sponsorship policy;
4. aggregator/signature validation;
5. delegated-code provenance;
6. recovery/master-key authority;
7. chain-specific capability state.

A transaction path being 'account abstracted' is not itself a security conclusion.

## 3. Permission requests remain distinct from capability discovery

ERC-7715 defines wallet execution-permission requests, revocation, supported-permission discovery, and granted-permission discovery. Its own security guidance emphasizes narrow scope, reasonable expiry, wallet enforcement correctness, and phishing risk.

Sentinel must distinguish:
- capability: what a wallet can support;
- permission: what a wallet has actually granted;
- authority: what the account/module can enforce;
- request: what an application asked for;
- execution: what happened on-chain.

Evaluation cases:
- requested permission broader than user-approved permission;
- expired permission accepted as current;
- revoked permission remains usable through stale context;
- wallet capability exists but permission was never granted;
- wallet UI says narrow scope while delegation manager enforces broader scope.

## 4. PeerDAS changes the data-availability evidence model

PeerDAS (EIP-7594) lets nodes verify blob data availability through sampling rather than full download. Ethereum reports PeerDAS shipped with Fusaka in December 2025 and continues blob scaling in 2026.

Sentinel must not confuse:
- data availability;
- transaction validity;
- execution correctness;
- settlement finality.

Security primitives:
- custody/sampling assumptions;
- erasure-code reconstruction thresholds;
- KZG commitment binding;
- peer discovery/distribution failures;
- sampling success versus full-data reconstruction;
- L2 dependence on L1 blob availability.

Evaluation cases:
- proof/commitment valid but expected blob unavailable;
- blob available but rollup state transition invalid;
- sampling evidence stale relative to challenged state;
- L2 incorrectly attributes settlement failure to DA.

## 5. Sequencing and preconfirmation must remain separate from finality

Ethereum's 2026 roadmap emphasizes fast finality research and ongoing rollup interoperability. Native-rollup proposals also make sequencing policy programmable: permissioned sequencer, based sequencing, or staked sequencer networks can have different trust models.

Sentinel rule:

> Preconfirmation is a latency/trust promise, not settlement finality.

Required evidence fields:
- sequencer identity/model;
- slashable or enforceable commitment, if any;
- expiry/freshness;
- L1 inclusion path;
- censorship/forced-inclusion path;
- reorg/finality boundary;
- conflicting-preconfirmation handling.

Evaluation cases:
- preconfirmation reported as final settlement;
- sequencer promise conflicts with L1 canonical outcome;
- user has no escape path under sequencer censorship;
- shared sequencer compromise affects multiple rollups simultaneously.

## 6. Cross-chain intent security remains resolver-and-settlement dependent

ERC-7683 standardizes a solver-facing order representation but explicitly does not guarantee the underlying protocol's settlement security.

Sentinel must track:
- resolver implementation provenance;
- settlement contracts;
- assets/token behavior;
- off-chain and cross-chain dependencies;
- solver capital-at-risk window;
- payment finality;
- permission/approval exposure;
- changing oracle, chain, token, or message-delivery state.

A solver following a resolver correctly can still be exposed if the resolver omits a required security assumption.

## 7. New curriculum families

Add or extend:
- wallet_capability_negotiation_and_atomicity
- account_abstraction_multi_layer_trust
- wallet_permission_request_vs_enforced_authority
- peerdas_data_availability_evidence
- sequencing_preconfirmation_and_finality
- cross_chain_solver_settlement_risk

Every training item should carry:
- chain id / environment;
- protocol or wallet version;
- capability state;
- granted permission state;
- authority source;
- freshness timestamp or block reference;
- expected post-state;
- evidence quality;
- abstention condition.

## 8. Core Sentinel rule

Sentinel must not infer security from feature presence.

It should reason:

`advertised capability -> current chain-scoped capability -> granted permission -> enforced authority -> execution semantics -> observed post-state -> settlement/finality`

A break in any edge must lower confidence or force abstention.

## Maturity notes

- EIP-5792: Final, Standards Track Interface.
- EIP-7594 PeerDAS: deployed with Fusaka according to Ethereum's 2026 protocol update; continuing blob scaling is roadmap work.
- EIP-7702: deployed with Pectra according to Ethereum's 2026 protocol update.
- ERC-4337: established higher-layer account abstraction path; runtime/bundler/paymaster implementation details remain independent trust surfaces.
- ERC-7715: useful permission model but should be tracked by its live EIP status rather than treated as universal wallet behavior.
- Native/based rollup and preconfirmation designs: research/proposal/implementation maturity must be recorded separately.

Driver remains out of scope and unchanged.
