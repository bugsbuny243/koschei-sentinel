# Koschei Sentinel — 12 September 2026 Research Pass

Date: 2026-09-12
Scope: Web3 through Web10 security/intelligence signals for Sentinel training, evaluation and architecture.
Driver: explicitly out of scope; unchanged.

## Evidence maturity classes

- A — deployed protocol / finalized standard / canonical implementation evidence
- B — official protocol or foundation roadmap / active implementation
- C — active Internet-Draft / draft standard / experimental protocol
- D — project-defined generation label, conceptual architecture, marketing, or speculative future framing

Sentinel must never infer that a numbered Web generation is an accepted standard merely because a project uses that label.

## Priority 0 — Ethereum protocol security and post-quantum migration

### Observed signal
Ethereum Foundation's 7 September 2026 protocol priorities make post-quantum resistance a near-fork engineering program rather than only a long-horizon research topic. The program spans execution, consensus and data. Native account abstraction / Frames is explicitly positioned as execution-layer cryptographic agility: account signature schemes can be changed without a hard fork for each scheme. Formal verification, privacy, state and zkEVM are parallel multi-fork arcs.

### Sentinel primitives
- cryptographic agility inventory
- legacy-key fallback detection
- downgrade-path reasoning
- account-abstraction signer hierarchy
- nonce/replay/domain separation
- transaction effect assertions and post-state invariants
- consensus-vs-execution cryptography distinction
- proof-system assumption and version binding

### Training/eval ideas
- distinguish protocol-roadmap capability from already-live mainnet behavior
- detect unsafe fallback to secp256k1/ECDSA master keys after delegated authorization
- require evidence before asserting a PQ migration is complete
- separate proof validity from statement/specification correctness

Maturity: B, with individual EIPs varying by status.

## Priority 1 — Solana Agave v4.3 staged rollout

### Observed signal
The Agave v4.3 schedule was updated 8 September 2026. Mainnet-beta upgrade candidate was tagged 4 September and 10% stake volunteers were requested 8 September. The next tentative milestones are 25% stake on 14 September, general adoption on 21 September and feature activation on 28 September.

### Sentinel primitives
- release-diff and feature-gate reasoning
- validator-version skew
- testnet/devnet/mainnet separation
- activation dependency mapping
- staged-rollout / rollback evidence

### Training/eval ideas
- never claim a tentative activation is already live
- require cluster and version evidence
- detect mismatches between client version, feature activation and runtime behavior

Maturity: A/B depending on delivered milestone; future schedule entries are not live facts.

## Priority 2 — ERC-8004 trustless agents

### Observed signal
Ethereum's AI-agent documentation states ERC-8004 is deployed across multiple networks and defines registries for agent identity, reputation and validation. Validation can use mechanisms including zkML, TEE and staked re-execution.

### Sentinel primitives
- agent identity vs authority
- reputation as evidence, never final authority
- validation-method provenance
- Sybil/reputation poisoning resistance
- verifier independence
- TEE/ZK/stake trust-model extraction

### Training/eval ideas
- high reputation must not override contradictory primary evidence
- require the validation method and verifier assumptions before trusting a claim
- compare minimum compromise sets across trust models

Maturity: A for deployed registry implementations; protocol trust claims still require implementation-specific evidence.

## Priority 3 — MCP 2026-07-28

### Observed signal
MCP 2026-07-28 moved to a stateless protocol core, formal Extensions framework and hardened authorization. The release adds issuer validation and tighter OAuth/OIDC alignment and restructures long-running Tasks to avoid unsafe global task listing in a stateless model.

### Sentinel primitives
- OAuth issuer/audience/resource binding
- confused-deputy detection
- task-handle authorization
- extension trust boundaries
- tool capability minimization
- credential isolation
- deprecation/downgrade safety

### Training/eval ideas
- over-broad token/resource grants
- issuer confusion
- cross-tool credential reuse
- task/artifact access without proper binding
- stale clients using deprecated semantics

Maturity: A/B — released protocol specification with production ecosystem adoption.

## Priority 4 — IETF AI-agent identity and delegation research

### Observed signal
An August 2026 Internet-Draft on AI-agent Internet architecture requires agent identity to be distinct from the human/organization principal, authorization evidence to be audience-restricted and narrowly scoped, multi-hop delegation to preserve original principal and authorized delegation steps, and delegated authority to be revocable or short-lived.

### Sentinel primitives
- principal → agent → sub-agent delegation graph
- time/value/context/redelegation constraints
- resource-server effective-authority calculation
- revocation freshness
- provenance-preserving delegation

### Training/eval ideas
- reject unauthenticated intermediary authority claims
- detect delegated privilege amplification
- require explicit scope and expiry
- require original-principal preservation for multi-hop actions

Maturity: C — active individual Internet-Draft, not an IETF standard.

## Priority 5 — NIST PQC migration

### Observed signal
NIST states finalized PQC standards are ready to implement now. June 2026 PIV working drafts describe a dual-stack migration model combining classical credentials with ML-DSA/ML-KEM additions for backward-compatible staged transition. NIST also distinguishes finalized standards from withdrawn candidates such as HAWK.

### Sentinel primitives
- algorithm inventory
- finalized-vs-candidate status
- dual-stack downgrade resistance
- long-lived key/data exposure
- migration completeness
- key/certificate object lineage

### Training/eval ideas
- never treat withdrawn or experimental candidates as finalized standards
- detect silent classical fallback
- flag mixed-mode systems where policy says PQ-required but classical auth remains authoritative

Maturity: A for finalized NIST algorithms; B/C for working PIV transition drafts.

## Priority 6 — Chain-agnostic identity and interoperability

### Observed signal
CAIP-2/CAIP-10 remain useful final standards for chain/account identifiers, while newer proposals add interoperable-address serialization and multichain/wildcard concepts. Maturity differs significantly across individual CAIPs.

### Sentinel primitives
- chain/account namespace binding
- canonicalization and homograph risks
- chain-context confusion
- cross-chain replay/domain separation
- asset-ID ambiguity
- session authorization scope

### Training/eval ideas
- same address bytes on multiple chains must not imply same authority context
- require chain ID in signatures and authorization evidence
- distinguish Final CAIPs from Draft/Review items

Maturity: A for CAIP-2/10; C for draft extensions.

## Priority 7 — Solana Token-2022 security surfaces

### Observed signal
Current Solana documentation highlights Transfer Hooks and Confidential Transfers. Transfer Hooks add external program logic and extra account dependencies to token transfers; clients are advised to simulate before signing. Confidential Transfers use client-generated proofs and ZK ElGamal verification, while account/mint/owner relationships remain public.

### Sentinel primitives
- transfer-hook program provenance
- extra-account mutation and dependency review
- simulation-vs-execution drift
- proof-context binding
- mint extension compatibility
- privacy claim boundaries

### Training/eval ideas
- a successful base token transfer assumption must not bypass hook logic
- detect unexpected mutable extra-account metadata
- distinguish amount privacy from sender/receiver anonymity
- ensure proof context is bound to intended transfer state

Maturity: A — deployed protocol/documented implementation behavior.

## Priority 8 — TEE remote attestation

### Observed signal
Confidential Computing Consortium material frames remote attestation as the trust anchor for autonomous systems: a TEE produces cryptographic evidence of hardware/software state and remote verifiers compare that evidence against expected measurements.

### Sentinel primitives
- attestation measurement identity
- freshness/replay protection
- expected-code policy binding
- key-release conditions
- TCB/supply-chain assumptions

### Training/eval ideas
- never infer "TEE therefore safe"
- require measurement, verifier policy and freshness
- detect stale attestation and rollback

Maturity: B — established confidential-computing primitive; individual platform guarantees vary.

## Web4 signal

The July 2026 `draft-reilly-web4-orion-00` describes an agentic, cryptographically verifiable Internet with verifiable origin/integrity/time and autonomous agents as first-class actors.

Useful Sentinel primitives:
- signed provenance
- agent identity
- delegation boundaries
- verifiable artifacts
- auditable action chains

Maturity: C. This is an individual Internet-Draft with no IETF endorsement or formal standards standing.

## Web5 signal

Do not model Web5 as a current universal generation standard. Prefer concrete decentralized-identity primitives and W3C VC/DID standards rather than the generation label itself.

Maturity: D for the generation label; standards underneath it must be tracked independently.

## Web6 signal

OASIS WEB6 v3.0 describes a project-defined AI abstraction/orchestration layer integrating many model providers, MCP/A2A and DID/VC identity.

Useful Sentinel primitives:
- multi-provider trust boundaries
- agent/tool authorization
- memory isolation
- orchestration adapter validation
- telemetry integrity
- credential-scoped capabilities

Maturity: D/B: active implementation/project source, but `Web6` is not a generally accepted Internet standard. Project claims require independent verification.

## Web7 signal

Web7 Foundation publishes draft RFCs around Accountable Intelligence Graph (AIG), Proof-of-Outcome, did:w7 and agent-mesh concepts. AIG models signed Intent → Delegation → Inference → Outcome causation as a DAG.

Useful Sentinel primitives:
- signed reasoning/evidence DAGs
- delegation lineage
- outcome attestation
- provenance-preserving subgraph export
- deterministic checks around probabilistic inference

Strong architectural relevance: Sentinel can adopt the *security principle* of a signed causal/evidence graph without accepting Web7's project-specific claims as authority.

Maturity: D/C — project-defined draft protocol family, not a broadly accepted Internet generation standard.

## Web8–Web10 signal

No broadly accepted Web8, Web9 or Web10 Internet standards were identified in this 12 September pass.

Rules for Sentinel:
- treat Web8–Web10 as discovery namespaces only
- never train "Web8/Web9/Web10 is an established Internet generation"
- admit only underlying concrete primitives with independent evidence

Watch themes:
- post-quantum identity and signatures
- verifiable autonomous agents
- proof-carrying delegated authority
- decentralized compute/storage trust
- privacy-preserving computation
- formally verified agent actions
- machine-to-machine economic authorization
- multi-agent governance and fail-safe controls

Maturity: D.

## 12 September training priority order

1. agent delegation and effective authority
2. PQ migration / cryptographic agility
3. MCP tool authorization and task provenance
4. ERC-8004 agent trust validation
5. Ethereum Frames / delegated account security
6. Solana runtime rollout and Token-2022 hooks/proofs
7. chain-agnostic identity / cross-chain context binding
8. TEE attestation
9. ZK statement/verifier correctness
10. Web4/Web6/Web7 concepts only after maturity labeling
11. Web8–Web10 discovery-only

## Dataset admission rule

A research signal may enter a reviewed curriculum queue only if it records:

1. canonical source and capture date/version
2. evidence maturity A/B/C/D
3. exact security primitive
4. threat/failure mode
5. evidence required for any conclusion
6. expected abstention condition
7. lineage/group ID for leakage-safe splitting
8. explicit note when a source is draft/project-specific/speculative

Research signals never override deterministic verdict authority and are never auto-promoted directly into production training data.
