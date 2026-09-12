# Sentinel Daily Research — 2026-09-12 — Addendum 3

Scope: defensive security research for Koschei Sentinel. This addendum separates deployed/official signals from drafts and project-specific framing. WebN labels are never treated as standards authority.

## 1. Ethereum interoperability and chain-context security

Current Ethereum protocol work treats interoperability as a first-class UX/security track. The Open Intents Framework is in production and current work builds toward trust-minimized cross-L2 interactions. ERC-7930 introduces chain-specific interoperable addresses; ERC-7828 adds human-readable interoperable names.

Sentinel primitives:
- chain context must be part of identity/address validation;
- never infer same authority from the same hexadecimal address on different chains;
- verify chain-id / namespace / address tuple before signing, routing or settlement;
- detect address-context stripping, chain-label spoofing and cross-chain replay ambiguity;
- preserve target-chain binding through intents, resolver output and execution receipts.

Eval ideas:
- identical address on L1 and L2 with different authority;
- malformed or downgraded interoperable address version;
- chain label resolves to unexpected chain metadata;
- solver executes valid intent on wrong destination chain;
- user-visible name and binary chain-specific address disagree.

Maturity:
- Open Intents Framework: production/deployed framework signal.
- ERC-7930 / ERC-7828: active peer-reviewed ERC proposals; treat specification status separately from deployed ecosystem use.

## 2. Intent-centric smart accounts expand execution trust boundaries

ERC-7806 describes an intent-centric EOA smart-account model where solvers/relayers may execute user intents. The proposal deliberately does not require one universal replay/re-entry rule; individual standards may define their own protections.

Sentinel primitives:
- intent domain separation;
- solver/relayer provenance;
- per-intent nonce/replay policy rather than assuming a global nonce exists;
- delegated-code storage and upgrade risk;
- paymaster / gas-sponsor authority;
- requested effects vs actual post-state;
- compensation-token and fee-routing verification.

Eval ideas:
- replayable intent because implementer assumed ERC-7806 supplies a global nonce;
- solver changes destination while preserving nominal amount;
- gas sponsor expands permissions beyond fee payment;
- delegated account code changes between signing and execution;
- intent succeeds but violates user-specified post-state invariant.

Maturity: ERC proposal / peer-review input. Do not teach as universal protocol truth.

## 3. Agent payments: x402 creates an authorization surface, not just a payment rail

Coinbase's x402 work extends HTTP-native payments to autonomous agents and, in 2026, expanded to broader ERC-20 support using Permit2 and gas-sponsorship extensions. AWS integration material shows agent-as-customer deployment moving into mainstream web infrastructure.

Sentinel primitives:
- payment authorization must be scoped independently from tool authorization;
- token approval amount, token identity, chain and recipient are all security-critical;
- gas sponsorship is a separate trust boundary;
- payment receipt must bind to the exact requested service/action;
- agent spending policy needs amount ceilings, rate limits and revocation;
- never infer that successful settlement proves service correctness.

Eval ideas:
- valid HTTP payment request with malicious recipient substitution;
- unlimited token approval for a low-value agent purchase;
- service response differs from the paid-for resource;
- cross-chain payment replay / wrong-chain settlement;
- compromised facilitator attempts to widen agent payment authority.

Maturity: deployed vendor/open-protocol ecosystem signal; not an IETF/W3C standard.

## 4. ERC-8004 + TEE/zkML/staked re-execution: validation method is evidence, not authority

ERC-8004's Validation Registry permits multiple verification models, including stake-secured re-execution, zkML and TEE oracles. These are explicitly pluggable trust models whose security level depends on the chosen validator and value at risk.

Sentinel primitives:
- validation-method provenance;
- request hash / payload commitment binding;
- validator identity and independence;
- proof/attestation freshness;
- statement definition for zkML;
- TEE measurement + verifier policy + TCB;
- stake amount / slashability / collusion assumptions for re-execution.

Eval rule: `validation exists` MUST NOT collapse to `result is safe`.

## 5. Restaking / AVS security must reason about slashable security, not brand labels

EigenLayer's security model emphasizes Operator Sets and Unique Stake to localize slashable security. Historical EigenLayer material also highlights operator collusion, unintended slashing and the gap between economic stake and potential harm.

Sentinel primitives:
- operator-set composition;
- unique slashable stake vs total delegated stake;
- quorum threshold;
- overlapping operators across services;
- maximum harm / profit-from-corruption estimate;
- slashing-condition correctness;
- governance / upgrade / veto assumptions where applicable;
- rate limits that cap damage during detection/slashing windows.

Eval ideas:
- high TVL but low uniquely slashable stake;
- same operators dominate multiple dependent AVSs;
- slashing bug punishes honest operators;
- bridge throughput exceeds economically attributable security;
- advertised pooled security is mistaken for directly slashable security.

Maturity: use current deployed-contract/docs evidence for production claims; whitepapers are architecture/risk inputs, not proof of current deployment state.

## 6. L2 scaling and interoperability: DA + settlement + censorship remain separate axes

Ethereum's scaling roadmap continues to emphasize blobs, DAS and progressive rollup decentralization. Cross-L2 UX work depends on shorter settlement and faster confirmations, but these improvements do not erase rollup-specific trust assumptions.

Sentinel must separately score:
1. data availability,
2. execution correctness,
3. settlement/finality,
4. censorship/escape path,
5. upgrade/admin authority,
6. interoperability resolver/solver assumptions.

A rollup or interop path is only as strong as its weakest required axis for the claim being made.

## 7. Web4–Web10 maturity check

A September 12 check still finds Web4 material such as `draft-reilly-web4-orion-00` as an individual active Internet-Draft with no IETF endorsement or formal standards standing.

Admission rule remains:
- standards-body Recommendation/RFC/final standard -> strongest authority;
- deployed protocol/reference implementation -> implementation evidence;
- active Internet-Draft / ERC draft -> research/spec input;
- project-specific Web6/Web7 labels -> discovery input;
- Web8/Web9/Web10 -> discovery namespaces unless concrete standards/implementations independently justify stronger classification.

No numbered Web generation may be promoted in Sentinel training merely because a document uses that label.

## 8. New curriculum families from Addendum 3

- `chain_context_and_interoperable_identity`
- `intent_smart_account_execution_security`
- `agent_payment_authorization_and_spend_policy`
- `pluggable_agent_validation_trust_models`
- `restaking_avs_cryptoeconomic_security`
- `cross_l2_interop_and_settlement_security`

Each supervised case should require:
- actor/principal graph,
- chain/protocol context,
- authority/delegation scope,
- concrete evidence IDs,
- trust assumptions,
- facts vs hypotheses,
- missing evidence,
- bounded conclusion,
- remediation or decisive verification step,
- confidence ceiling.

## 9. Priority change for Sentinel

Priority order after the third Sept 12 pass:
1. delegated authority + chain-context identity,
2. agent payment authorization,
3. intents/resolver/solver security,
4. PQ crypto-agility,
5. L2/interop trust decomposition,
6. ERC-8004 pluggable validation,
7. restaking/AVS cryptoeconomic reasoning,
8. TEE/ZK proof-assumption reasoning,
9. Web4/Web6/Web7 research-only signals,
10. Web8-Web10 discovery-only monitoring.

Driver remains outside this research change set.
