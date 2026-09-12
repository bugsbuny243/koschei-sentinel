# Sentinel Daily Research — 2026-09-12 Addendum 2

Scope: defensive cybersecurity research for Koschei Sentinel. This addendum deepens Ethereum L2/rollup security, cross-chain intents/IBC v2, Solana Token-2022 privileged controls, and post-quantum implications. Driver remains out of scope.

## Evidence classes

- A: canonical standard/protocol/official documentation
- B: active implementation with reproducible code
- C: draft/experimental specification
- D: speculative/project-defined framing

Maturity must be tracked separately from deployment. A deployed implementation does not automatically make every associated specification final.

## 1. Ethereum L2 / rollup security

### Observed signals

Ethereum's current optimistic-rollup documentation emphasizes challenge periods, fraud proofs, L1 data availability, L1 settlement, forced transaction/withdrawal paths, and the dependence on at least one honest challenger. ZK-rollups instead rely on validity proofs verified by L1, but still retain data-availability, verifier-contract, circuit/specification, sequencer and proving-system assumptions.

The Ethereum Foundation's 2026 L1/L2 framing explicitly distinguishes Stage 2 rollups from systems that inherit only subsets of Ethereum security properties, such as validiums. Sentinel must therefore reason from the concrete security model rather than the marketing label “L2”.

### Sentinel primitives

- sequencer censorship and liveness boundaries
- forced-inclusion / escape-hatch availability
- challenge-window and watcher-liveness assumptions
- fraud-proof correctness and dispute deadlines
- validity-proof verifier correctness
- circuit/specification equivalence
- L1 data availability vs off-chain DA assumptions
- upgrade/admin-key and emergency-governance authority
- settlement/finality distinctions
- proving centralization and prover availability

### Eval cases

- Do not equate “rollup” with full Ethereum-equivalent security.
- Distinguish optimistic challenge assumptions from ZK validity-proof assumptions.
- Require an explicit escape/forced-exit path before claiming censorship resistance.
- Treat a valid proof as evidence only for the statement encoded by the verifier/circuit, not for broader application correctness.
- Track upgrade keys and emergency controls as first-class authority edges.

Maturity: A/B depending on source and implementation.

## 2. ERC-7683 cross-chain intents

### Observed signal

ERC-7683 defines a solver-facing cross-chain intent interface. Orders are resolved through protocol-specific resolver contracts into a common representation. Solvers can then evaluate and fulfill steps and variables, including dependency relations.

### Sentinel primitives

- resolver provenance and code identity
- solver authorization and incentive assumptions
- order/domain separation
- dependency-graph acyclicity
- target/selector/call-data validation
- price/oracle-resolution assumptions
- settlement and payment atomicity
- replay, stale-order and expiry handling
- destination-chain identity/address binding
- requested outcome vs actual post-state

### Eval cases

- A resolver being standards-compatible does not make it trustworthy.
- Solver success must be judged by fulfilled constraints/post-state, not transaction inclusion alone.
- Cross-chain address identity must include chain namespace.
- Sentinel should identify the minimum actor set or resolver compromise needed to violate the order.

Maturity: A for the ERC text; deployment maturity must be tracked per implementation.

## 3. IBC v2 / Eureka

### Observed signals

IBC v2/Eureka simplifies the protocol while retaining client-backed packet verification. The public IBC material says v2 launched in 2025 and expands beyond the Cosmos ecosystem, while the canonical IBC v2 specification in the repository is still marked EXPERIMENTAL and several ICS entries remain Draft.

The v2 security model places substantial responsibility in the configured IBC client. The specification explicitly permits security models ranging from fully verified light clients to multisignature/attestation clients; users must judge whether the instantiated client security model is acceptable. Implementations also note that light clients can freeze after conflicting headers or prolonged lack of updates.

### Sentinel primitives

- client type and concrete security model
- trusted initial consensus state
- counterparty/client-ID binding
- packet source/destination binding
- membership/non-membership proof verification
- timeout semantics
- relayer untrusted-input handling
- frozen/stale light-client recovery
- governance recovery authority
- attestation/multisig fallback vs verified-light-client distinction

### Eval cases

- Never classify “IBC” as a single uniform trust model without inspecting the client.
- Distinguish verified light client, ZK-assisted light client and multisig/attestation client.
- Detect wrong-counterparty/client-ID binding.
- Treat frozen-client governance recovery as an authority path, not a purely cryptographic event.
- Keep implementation deployment and specification maturity as separate fields.

Maturity: B for active implementations; C for the experimental/draft specification state.

## 4. Solana Token-2022 privileged controls

### Observed signals

Solana's Token Extensions documentation exposes privileged token behavior that ordinary SPL-token assumptions do not capture. `PermanentDelegate` is a mint-level authority that can authorize transfers and burns for any token account of the mint and cannot be revoked by individual token-account owners. Solana's own Kora security guidance highlights this as a material payment risk.

Token-2022 also includes Transfer Hook, CPI Guard, Confidential Transfer/Balance, pausability and other extensions that materially change the effective authorization or execution model.

### Sentinel primitives

- mint-extension inventory before risk conclusions
- permanent-delegate authority graph
- delegate rotation authority
- transfer-hook program provenance and mutability
- CPI-guard expectations
- confidential-balance proof and auditor-key assumptions
- pausability/freeze authority
- extension compatibility and initialization state

### Eval cases

- A token-account owner is not necessarily the highest transfer authority.
- PermanentDelegate must be flagged in payment/escrow risk reasoning.
- Transfer-hook existence changes execution assumptions and must be simulated/reviewed with hook accounts.
- Confidential balances hide amount/balance data, not necessarily participant identities.
- Auditor keys provide visibility, not transfer authority.

Maturity: A/B.

## 5. Post-quantum effects on wallets, bridges and rollups

### Observed signals

Ethereum's current security roadmap identifies four PQ migration surfaces: BLS validator signatures, KZG commitments, ECDSA accounts and some ZK proof systems. EIP-8141/native account abstraction is intended to provide signature agility for accounts. Ethereum documentation also notes that accounts that have already signed transactions expose their public key on-chain, unlike never-spent accounts where only the hash-derived address is visible.

### Sentinel primitives

- cryptographic algorithm inventory across wallet, bridge, validator and proof layers
- exposed-public-key status
- classical/PQC dual-stack fallback risk
- downgrade resistance
- threshold/multisig migration completeness
- bridge validator/attestor signer migration
- proof-system and commitment-scheme migration
- long-lived authorization/key exposure

### Eval cases

- Do not claim “PQ-ready” if a legacy master/recovery signer remains authoritative.
- Bridge signer migration must include every threshold member and fallback/recovery path.
- PQ account authentication does not imply PQ-safe bridge attestations, consensus signatures or proof systems.
- Track crypto agility separately for execution, consensus, DA and application layers.

Maturity: A for official Ethereum/NIST-backed primitives; concrete migration implementations vary.

## 6. New curriculum families

Add or extend:

- `rollup_security_and_escape_hatches`
- `cross_chain_intent_resolvers_and_solvers`
- `ibc_client_security_models`
- `token2022_privileged_extensions`
- `pq_bridge_and_wallet_migration`

Each example should require:

1. actor and authority graph
2. chain/domain identifiers
3. concrete trust assumptions
4. evidence and source maturity
5. fact vs inference separation
6. failure/compromise condition
7. bounded conclusion
8. remediation or next evidence request
9. confidence ceiling

## 7. Priority update for Sentinel

Priority order after this addendum:

1. delegated agent/tool authority and MCP/A2A boundaries
2. PQ crypto-agility and legacy fallback detection
3. rollup security-model extraction
4. cross-chain intents: resolver/solver/post-state verification
5. IBC client-type and packet-binding reasoning
6. ERC-8004 identity/reputation/validation trust
7. Solana runtime + Token-2022 privileged extension reasoning
8. TEE attestation / ZK statement-verifier reasoning
9. Web4/Web6/Web7 concepts only as research inputs
10. Web8-Web10 discovery-only namespaces

## 8. Admission rule

No training example should state that a protocol or numbered Web generation is “secure” based only on label, adoption or proof existence. Sentinel must identify the exact authority, evidence, trust boundary and failure mode that supports each conclusion.
