# Web4-Web6 Security Research — 2026-09-20

## Scope
Sentinel-relevant protocol work only. “Web4/Web5/Web6” is treated as a research grouping, not as a claim that Web6 is a standardized web generation.

## Findings

### Agent delegation
- IETF draft-hamr-oauth-agent-delegation-01 (2 Sep 2026) proposes attenuated authorization chains for automated agents across administrative domains. Each delegation link must narrow or preserve scope/conditions/expiry and verifiers check the complete chain.
- IETF draft-asor-wimse-agent-delegation-chain-01 (3 Sep 2026) explores verifiable attenuated delegation for AI-agent chains.
- draft-nelson-agent-delegation-receipts-10 remains an individual Active Internet-Draft, not an IETF standard. It proposes cryptographic delegation receipts for binding agent actions to authorization evidence.

Sentinel implication: model agent authorization as an evidence chain with explicit principal, delegate, scope, validity window, attenuation and action evidence. Treat these drafts as research inputs, not normative dependencies.

### Verifiable credentials / identity
- W3C Digital Credentials Working Draft, 4 Sep 2026, defines browser-mediated presentation and issuance with explicit user participation and separate permission-policy controls for get/create.
- W3C Verifiable Credentials Data Model v2.1 Working Draft, 5 Sep 2026, continues issuer-holder-verifier data-model work.
- W3C Recognized Entities v1.0 Working Draft, 6 Sep 2026, introduces cryptographically verifiable recognition of entities that issue or verify credentials.
- W3C Threat Model for Decentralized Credentials Group Note Draft, 8 Sep 2026, provides a living threat model spanning credential flows and security/privacy threats.

Sentinel implication: provenance and identity evidence should distinguish issuer identity, holder/delegate identity, verifier identity, authorization scope, user consent/presence and credential integrity. Threat-model ingestion is useful; browser credential APIs themselves should not be treated as agent authorization.

## Engineering direction
1. Keep authorization evidence separate from model output.
2. Represent delegation as an attenuating chain; fail closed on scope expansion, expired links, broken signatures, or ambiguous principal binding.
3. Preserve provenance sufficient to reconstruct why an agent/tool action was authorized.
4. Keep experimental drafts behind versioned adapters/policies rather than hard-coding them as standards.
5. Do not create a generic “Web6 protocol” dependency without a recognized standards basis.

## Sources
- https://datatracker.ietf.org/doc/draft-hamr-oauth-agent-delegation/01/
- https://datatracker.ietf.org/doc/html/draft-asor-wimse-agent-delegation-chain-01
- https://datatracker.ietf.org/doc/html/draft-nelson-agent-delegation-receipts-10
- https://www.w3.org/TR/2026/WD-digital-credentials-20260904/
- https://www.w3.org/TR/vc-data-model/
- https://www.w3.org/groups/wg/vc/publications/
- https://www.w3.org/TR/2026/DNOTE-threat-model-decentralized-credentials-20260908/
