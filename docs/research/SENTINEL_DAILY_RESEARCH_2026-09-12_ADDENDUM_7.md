# Sentinel Daily Research — 2026-09-12 — Addendum 7

## Scope
This addendum extends the 12 September research set with ordering-right security, ePBS/inclusion-list maturity, prover/verifier supply-chain integrity, proof lifecycle governance, bridge/rollup operational authority, and AI-assisted security evidence quality.

## Evidence discipline
- Treat deployed behavior, scheduled fork content, active research, draft EIPs/ERCs, and project proposals as different maturity classes.
- A valid cryptographic proof does not prove the surrounding governance, verifier registry, public-input policy, or operational supply chain is safe.
- A protocol-level inclusion or commitment is not automatically equivalent to broad execution validation or economic finality.

## 1. Enshrined proposer-builder separation (ePBS)
Ethereum's Glamsterdam roadmap schedules ePBS / EIP-7732 for inclusion. The purpose is to move proposer-builder exchange from off-protocol middleware into protocol rules and reduce relay trust. The design also introduces payload timeliness handling and separates consensus-block timing from execution-payload handling.

Sentinel implications:
- Model proposer, builder, relay/middleware, payload, bid, signature, and timeliness committee as separate actors/evidence sources.
- Never equate `payload committed` with `payload broadly validated`.
- Track whether an observed network is pre-ePBS, transitional, or actually running the fork rules.
- Detect off-protocol relay dependencies that remain even after protocol support exists.

Training family:
`epbs_builder_proposer_and_payload_timeliness_security`

## 2. Inclusion lists and censorship evidence
Inclusion lists remain an active research/security mechanism. EIP-7547 defines forced-inclusion machinery but the broader roadmap still describes inclusion-list configuration as an active research area rather than settled deployed behavior.

Sentinel implications:
- Distinguish transaction visibility from transaction inclusion guarantees.
- Record who produced an inclusion list, its slot/domain, signature, execution eligibility, and whether the following payload actually satisfied it.
- Treat censorship-resistance evidence as temporal and slot-bound rather than as a permanent property of a builder or proposer.
- Do not infer inclusion-list deployment from PBS/ePBS deployment alone.

Training family:
`inclusion_list_and_censorship_evidence`

## 3. MEV and ordering-right trust boundaries
PBS changes who has practical transaction-ordering authority. MEV documentation continues to highlight builder concentration, private/permissioned mempools, and ordering incentives as central security/decentralization concerns.

Sentinel implications:
- Represent ordering authority separately from consensus authority.
- Detect private order-flow dependencies, builder concentration, and hidden middleware assumptions.
- Do not treat the highest-bid block as evidence of neutral ordering.
- Compare user intent, observed order, state transition, and economic outcome.

Training family:
`mev_ordering_authority_and_private_orderflow`

## 4. L1 zkEVM prover/verifier supply chain
Ethereum's L1 zkEVM direction relies on specialized provers producing proofs that inexpensive verifiers can check. EF funding in 2026 includes on-prem multi-GPU proving efforts intended to stress-test operational resilience and reduce cloud concentration.

Sentinel implications:
- Separate proof soundness from prover availability and supply-chain resilience.
- Track prover implementation/version, circuit artifact, verifier artifact, proving backend, hardware/cloud dependency, and reproducibility.
- Require verifier/circuit/version binding before trusting a proof result.
- Detect single-provider operational concentration even where cryptographic verification is sound.

Training family:
`prover_verifier_supply_chain_and_operational_resilience`

## 5. Verifier registry and proof-governance authority
ERC-8262 provides a useful research example of a verifier router with per-proof-type verifier registration, verifier versioning, administrative roles, attestation TTL, configuration history, and revocation semantics.

This is a draft/proposed ERC and must not be treated as a universal deployed compliance standard. It is valuable because it exposes an important class of security failure: a proof can be mathematically valid while the verifier address, public-input policy, provider configuration, or admin-controlled registry is wrong.

Sentinel implications:
- Verify exact verifier address/version and circuit/proof-type binding.
- Track ownership/admin roles on verifier registries.
- Require timestamp/TTL/freshness checks where the statement is time-dependent.
- Treat historical configuration revocation and version history as evidence.
- Flag silent verifier replacement or public-input-policy drift.

Training family:
`verifier_registry_versioning_and_proof_governance`

## 6. Proof statement vs surrounding metadata/privacy
Ethereum's 2026 privacy guidance emphasizes that ZK proofs protect the proof statement, not all surrounding metadata. RPC providers, session reuse, analytics, events, calldata, IP/session correlation, note handling, and public inputs can defeat intended privacy even when the proof is valid.

Sentinel implications:
- Separate cryptographic privacy from metadata privacy.
- Inspect public inputs/events/calldata/logging/session/RPC linkability.
- Treat nullifier/secret-note handling as key material.
- Never output `privacy preserved` solely because a ZK proof verifies.

Training family:
`zk_statement_privacy_vs_metadata_leakage`

## 7. AI security findings require evidence lineage
Ethereum Protocol Security's 2026 AI-agent work shows the useful unit is not merely a model finding but a triaged, reproducible security result grounded in affected code/version and reviewable evidence.

Sentinel implications:
- Require source location, affected version, reproduction evidence, security impact, and remediation/regression evidence for strong vulnerability conclusions.
- Separate model hypothesis from verified finding.
- Preserve provenance of agent outputs and supporting artifacts.
- Abstain or downgrade confidence when lineage is incomplete.

Training family:
`ai_finding_provenance_reproduction_and_triage`

## 8. Sentinel authority/evidence graph update
New graph segment:

`transaction intent -> order-flow source -> builder/solver authority -> inclusion/commitment -> execution payload -> broad validation -> settlement/finality`

and for ZK-backed decisions:

`statement -> public inputs -> circuit/version -> prover implementation -> proof -> verifier address/version -> registry/admin authority -> freshness/configuration -> resulting state transition`

A break in any edge lowers confidence. Cryptographic validity cannot overwrite missing authority, version, freshness, or governance evidence.

## 9. New evaluation cases
1. ePBS commitment exists but payload is not yet broadly validated: Sentinel must not call it final.
2. Inclusion list is present but target transaction was not executable: Sentinel must distinguish censorship evidence from execution eligibility.
3. ZK proof verifies against a stale verifier registry entry: Sentinel must reject the security conclusion.
4. Verifier is upgraded by an admin key after proof generation: Sentinel must bind the decision to the verifier version active at evaluation time.
5. Single cloud prover outage halts proof production while verifier correctness remains intact: Sentinel must classify availability failure separately from soundness failure.
6. Privacy proof is valid but RPC/session metadata links identities: Sentinel must report metadata privacy failure.
7. AI agent reports a vulnerability without affected-version or reproduction evidence: Sentinel must classify it as a hypothesis, not a verified finding.
8. Builder ordering produces an adverse execution despite protocol-valid block: Sentinel must distinguish consensus validity from user-intent/economic safety.

## Source snapshot
Primary/current sources reviewed for this addendum:
- Ethereum roadmap security page, updated July 2026.
- Ethereum proposer-builder separation roadmap page, updated June 2026.
- Ethereum Glamsterdam roadmap page, September 2026 snapshot.
- EIP-7732, Enshrined Proposer-Builder Separation.
- EIP-7547, Inclusion Lists.
- Ethereum MEV developer documentation.
- Ethereum L1 zkEVM roadmap page, updated June 2026.
- Ethereum Foundation Q2 2026 allocation update (on-prem proving resilience work).
- Ethereum 2026 privacy-app guidance.
- Ethereum Foundation Protocol Security post on AI-agent triage, July 2026.
- ERC-8262 proposal/reference implementation as a draft research example for verifier lifecycle/governance.

## Maturity labels
- ePBS / Glamsterdam: scheduled protocol work; do not mark as deployed until fork activation is verified.
- Inclusion lists: active research/proposal area; not equivalent to deployed ePBS.
- L1 zkEVM: active research/engineering direction; not current universal L1 validation architecture.
- ERC-8262: proposal/draft research input; not a universal standard.

## Sentinel conclusion
The model must reason over two independent dimensions:
1. **cryptographic correctness** — signatures, proofs, hashes, commitments;
2. **operational authority and lifecycle correctness** — who can replace, revoke, reorder, upgrade, withhold, censor, or reinterpret those artifacts.

A system is not secure merely because its cryptography verifies. Sentinel should only raise confidence when cryptographic evidence, authority lineage, version binding, freshness, deployment maturity, and observed state all agree.
