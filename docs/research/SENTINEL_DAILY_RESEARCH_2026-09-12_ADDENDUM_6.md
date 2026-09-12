# Sentinel Daily Research — 2026-09-12 — Addendum 6

Cutoff: 2026-09-12. Scope: defensive security research only.

## 1. Cross-chain signatures: replay can be intentional

ERC-7964 defines cross-chain EIP-712 signatures by omitting `chainId` from the top-level domain and encoding the authorized chain operations inside the signed message set. This is a deliberate design choice: one signature may be valid on multiple chains.

Sentinel implications:
- never classify signature reuse across chains as a replay bug without checking the signature scheme;
- require per-operation `chainId`, account/verifying-contract binding, nonce/deadline, and current chain account state;
- compare code/state at the same address across chains before assuming equivalent authority;
- treat partial cross-chain execution as a separate failure mode.

Curriculum family: `cross_chain_signature_replay_and_domain_binding`.

## 2. Recovery is a root-of-authority surface

ERC-7093 and draft ERC-7947 show recovery evolving from a single backup key into guardian/provider-based policy systems. Recovery providers can effectively become alternate roots of control.

Sentinel implications:
- enumerate all recovery providers/guardians and thresholds;
- verify add/remove-provider access control;
- distinguish recovery proof validation from post-recovery authority;
- test malicious-provider enrollment, stale guardians, threshold downgrade, replay and recovery race conditions;
- require revocation/rotation evidence after successful recovery.

Curriculum family: `wallet_recovery_provider_and_guardian_security`.

## 3. Agent wallets need sandboxed authority, not just an agent key

Draft ERC-8199 proposes a sandboxed smart-wallet pattern for agents and explicitly calls out replay protection, validity windows, whitelist-style policy enforcement and an owner escape path.

Sentinel implications:
- agent key != owner key;
- require time-bounded authority and replay protection;
- prefer positive allowlists over negative deny lists for high-impact actions;
- ensure owner emergency execution/revocation remains available;
- compare requested agent action against actual permitted target, selector, asset, value and time window.

Curriculum family: `sandboxed_agent_wallet_authority`.

## 4. Reputation is evidence, never authority

ERC-8004 explicitly acknowledges Sybil attacks against reputation. It also states that on-chain registration cannot prove advertised capabilities are functional or non-malicious.

Sentinel implications:
- reputation score is an input signal only;
- weigh reviewer identity/reputation, economic cost, feedback diversity, time distribution and corroborating validation;
- detect self-reinforcing reviewer clusters and burst/Sybil patterns;
- separate identity ownership from capability truth;
- require stronger validation as value-at-risk increases.

Curriculum family: `agent_reputation_sybil_and_signal_quality`.

## 5. Prover outsourcing adds availability and diversity assumptions

Ethereum's L1 zkEVM roadmap describes specialized provers generating proofs while validators verify them cheaply, with research into real-time proving, prover markets and multiple independent zkVM implementations. The design direction favors proof/client diversity rather than one proving implementation becoming a hidden single point of failure.

Sentinel implications:
- proof validity != prover availability;
- verify proof system/version/parameters and statement binding;
- detect single-prover or single-zkVM concentration;
- model proof-latency and missed-deadline failure modes;
- distinguish incorrect proof generation from inability to generate a proof in time;
- require provenance for prover software and circuit/VM version.

Curriculum family: `prover_market_diversity_and_availability`.

## 6. Preconfirmation is a promise layer, not settlement

2026 Ethereum research on based rollups and preconfirmations distinguishes L1-based ordering from off-chain sequencing and explores combining low-latency preconfirmations with synchronous composability.

Sentinel implications:
- preconfirmation must carry issuer, scope, expiry and commitment semantics;
- verify whether commitment covers inclusion, ordering, state result, or only transaction acceptance;
- do not equate a preconfirmation with L1 finality;
- include equivocation, missed commitment, builder/proposer change and fallback behavior in evaluation;
- bind user-visible guarantees to the exact preconfirmation protocol in use.

Curriculum family: `preconfirmation_commitment_and_finality_gap`.

## 7. Solver risk spans the whole settlement window

ERC-7683 security considerations now explicitly frame solver exposure from capital/approval commitment until payment becomes final and spendable. Resolver output does not guarantee protocol safety.

Sentinel implications:
- build a time-indexed trust graph from solver commitment to final payment;
- track token/oracle/permission/message-delivery changes during that window;
- require all solver obligations and asset-spend rights to be explicit;
- identify conditions under which a correctly behaving solver can still become insolvent or unpaid;
- distinguish resolver correctness from settlement-protocol safety.

Curriculum family: `solver_settlement_window_and_liquidity_risk`.

## 8. Proof freshness and cross-deployment binding

Draft ERC-8262 illustrates an important general pattern: a proof can be valid cryptographically while replayable across chain or contract deployments if chain/contract identity is not committed into public inputs. Signed variants bind chain id and oracle address to close this class of gap.

Sentinel implications:
- inspect proof public inputs for chain, contract, version, timestamp/epoch and subject binding;
- require freshness/replay policy even for cryptographically valid proofs;
- identify whether replay prevention is in-circuit, on-chain storage, signature domain, or external policy;
- downgrade confidence when proof validity is portable beyond its intended security domain.

Curriculum family: `proof_freshness_and_cross_deployment_binding`.

## Updated Sentinel reasoning chain

`identity -> delegated authority -> recovery authority -> scoped execution -> proof/attestation -> replay/freshness -> settlement window -> finality -> reputation corroboration`

A conclusion is bounded by the weakest unverified link in this chain.

## Source maturity notes

- ERC-8004: deployed on multiple networks, but still marked Draft in the ERC document; use implementation evidence and specification maturity separately.
- ERC-7964, ERC-7947, ERC-8199, ERC-8262: draft proposals; research inputs, not production-standard truth.
- Ethereum zkEVM L1 verification: active research, not yet integrated into production Ethereum clients.
- Ethereum Research preconfirmation discussions: research/design input, not protocol-final authority.

## Admission rule

Every training/eval item derived from these signals must include source URL/version/date, maturity class, trust boundary, replay/freshness assumptions, required evidence, abstention condition and defensive learning objective. No research label or reputation score may override deterministic evidence.
