# Sentinel Research Signal Map — Web3 to Web10

Date: 2026-09-12
Scope: research-only intelligence feed for Koschei Sentinel. Sentinel Driver is out of scope and must not be modified by this document or derived training work.

## Evidence grading

- A — standards body / protocol foundation / canonical project source
- B — active public implementation with reproducible code and dated releases
- C — individual Internet-Draft or ecosystem draft; useful but not standards consensus
- D — project-defined generation label / speculative framing; mine technical primitives only

## Web3 — production / highest priority

Status: A/B.

Current signals:

- Ethereum is explicitly prioritizing L1 hardening, post-quantum readiness, privacy, state, zkEVM/formal verification and native account abstraction.
- Ethereum's September 2026 protocol update elevates post-quantum work into near-fork delivery planning, with Frames/native account abstraction providing signature agility for accounts.
- Ethereum identifies ECDSA, BLS, KZG and some ZK systems as migration surfaces for post-quantum security.
- Solana's Agave v4.x release cadence is active in 2026; v4.3 staged adoption is in progress during September. Validator/client diversity and runtime/signature-verification work remain important reliability signals.

Sentinel training candidates:

1. cryptographic-agility reasoning;
2. account-abstraction security review;
3. validator/client diversity and consensus-risk analysis;
4. smart-contract/program secure-code review;
5. bridge/cross-domain trust-boundary reasoning;
6. formal-verification evidence interpretation;
7. privacy/security trade-off analysis;
8. deterministic remediation and regression-test generation.

## Web4 — agentic + verifiable web

Status: C.

A July 2026 individual IETF Internet-Draft, `draft-reilly-web4-orion-00`, defines a Web4 architecture centered on autonomous agents and cryptographically verifiable origin, integrity and time. It is explicitly an individual Internet-Draft and is not IETF endorsement or a standards-track result.

Sentinel should not learn "Web4" as an established Internet version. It should learn the underlying primitives:

- agent identity;
- signed provenance;
- verifiable artifacts;
- delegation boundaries;
- policy-constrained orchestration;
- auditable agent actions.

## Web5 — identity / decentralized data lineage

Status: mixed; terminology fragmented.

Treat "Web5" as a historical/project label rather than a single current standard. Prefer standards-level primitives such as W3C Verifiable Credentials and DIDs when building training material.

W3C Verifiable Credentials Data Model 2.0 is a Recommendation. Useful Sentinel concepts:

- issuer / holder / verifier trust separation;
- credential integrity and status;
- correlation/privacy risks from identifiers;
- DID resolution as an optional identity mechanism, not a mandatory dependency.

## Web6 — active project-defined architecture

Status: D/B depending on component.

OASIS WEB6 currently presents itself as an AI abstraction/orchestration layer combining agent protocols, DID/VC identity, memory providers, telemetry and multiple model/provider adapters. The "Web6" label is project-defined, not a generally accepted Internet generation standard.

Mine only reusable security concepts:

- multi-provider trust boundaries;
- tool/agent authorization;
- memory isolation and retention;
- loop/cycle detection;
- telemetry integrity;
- credential-scoped capabilities;
- protocol adapter validation.

## Web7 — agent-native / accountable intelligence experiments

Status: C/D with active implementations.

Public Web7 material in 2026 includes draft protocol work around agent identity, delegation, proof-of-outcome, accountable intelligence graphs and governance. Independent IETF Internet-Drafts also exist for DID7-related schemes. These drafts are work in progress and do not imply IETF approval.

High-value Sentinel primitives:

- accountable intelligence graphs;
- signed delegation chains;
- outcome attestation;
- agent reputation as untrusted evidence, not authority;
- policy/governance traces;
- provenance-preserving memory;
- deterministic verification around probabilistic models.

## Web8 / Web9 / Web10 — speculative namespace

Status: D unless a concrete implementation or recognized draft is independently verified.

Do not teach Sentinel that Web8, Web9 or Web10 are established Internet generations. Use these labels only as discovery buckets. Any candidate technology must be reduced to its concrete primitive and independently graded before it enters training.

Potential discovery themes:

- post-quantum identity and signatures;
- verifiable autonomous agents;
- decentralized compute and storage;
- machine-to-machine economic authorization;
- cryptographic provenance;
- privacy-preserving computation;
- formal verification of AI/security actions;
- multi-agent governance and fail-safe controls.

## Admission rule for Sentinel training

A research item may enter the training-candidate pool only if all of the following are recorded:

1. canonical source URL;
2. publication/update date;
3. evidence grade A-D;
4. concrete security primitive;
5. defensive learning objective;
6. known limitations / draft status;
7. no unsupported claim that a numbered "Web" generation is an accepted standard.

Research labels are never authority. Deterministic evidence and source quality remain authoritative.

## Canonical sources checked in this pass

- https://blog.ethereum.org/2026/09/07/protocol-priorities
- https://ethereum.org/roadmap/security/
- https://ethereum.org/roadmap/security/quantum-resistance/
- https://ethereum.org/roadmap/account-abstraction
- https://github.com/anza-xyz/agave/wiki/v4.3-Release-Schedule
- https://solana.com/news/solana-changelog-august-20-2026
- https://datatracker.ietf.org/doc/draft-reilly-web4-orion/
- https://www.w3.org/TR/vc-data-model-2.0/
- https://datatracker.ietf.org/doc/draft-herman-did-web7-urn/
- https://web7.foundation/rfcs/
- https://web6.oasisomniverse.one/
