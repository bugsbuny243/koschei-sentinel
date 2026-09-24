# Sentinel V1 — Model-Agnostic Web4/Web5/Web6 Security Architecture

Status: production direction for V1; 397B/35B-active remains a future scale target, not a V1 dependency.

## Product boundary

Sentinel is the security intelligence system, not a particular foundation model. Foundation models are replaceable reasoning engines behind a stable Sentinel control plane.

The stable product boundary owns:
- evidence/provenance and deterministic verification
- policy and authorization enforcement
- agent identity, controller and delegation-chain reasoning
- protocol adapters and normalization
- security tool mediation and audit
- conformance/adversarial evaluation
- ARVIS integration
- tenant isolation and deployment policy
- model routing, fallback and output validation

## V1 inference contract

Every reasoning provider must sit behind a provider-neutral contract. The contract must carry:
- request_id / tenant_id
- task class and risk class
- normalized evidence references, never implicit authority
- allowed tools/capabilities
- model/provider identity and version
- structured response schema
- citations/evidence bindings
- usage/latency metadata
- refusal/error state

Provider output is untrusted input until Sentinel validators accept it. A model may recommend an action; it does not grant itself authority to execute one.

## Web4/Web5/Web6 readiness

"Web4/Web5/Web6" are treated as a product horizon, not as three settled standards.

Current standards signals point toward an Agentic Web:
- W3C AI Agent Protocol CG is working on agent discovery, identification, intent/capability exchange, collaboration and secure interoperable agent communication.
- IETF individual drafts are exploring verifiable attenuated delegation chains where delegated authority cannot silently expand and can be checked by enforcement points.
- Identity, delegation, provenance, transaction security and protocol interoperability therefore belong in Sentinel's canonical security model rather than in provider-specific prompts.

Adapters may change as drafts evolve. Canonical Sentinel evidence semantics must not depend on a draft remaining unchanged.

## V1 routing

Initial routing classes:
1. fast triage — low-cost classification/extraction
2. deep reasoning — difficult code/security/agent cases
3. verifier — independent structured validation where required
4. local/private — customer-controlled deployment when external inference is forbidden

Routing policy is configuration, not business logic. No provider name is embedded into evidence semantics.

## Trust rules

1. Fail closed on malformed or unverifiable security-critical outputs.
2. Delegated authority can remain equal or narrow; never widen implicitly.
3. Bind every high-impact conclusion to evidence/provenance.
4. Record provider/model/version for reproducibility.
5. Separate identity from authority.
6. Separate reasoning from execution.
7. Treat tool results and model text as distinct evidence classes.
8. Preserve crypto agility and revocation/version metadata.
9. Never send customer data to an external provider unless tenant policy permits it.
10. Provider outage/model retirement must not corrupt Sentinel state.

## Commercial evolution

V1: replaceable reasoning engines + Sentinel-owned security/control plane.
V2: security-specialized fine-tuned/open-weight engines where economics and measured quality justify them.
Sentinel Max: Koschei-native large sparse-MoE target, including the existing 397B/~35B-active research architecture, only when revenue/compute and benchmark evidence justify the scale.

The API, evidence model and policy plane should remain stable across these transitions.

## Acceptance gates before customer production

- provider-neutral contract tests
- deterministic structured-output validation
- tenant data-egress policy tests
- prompt/tool-injection adversarial suite
- agent identity/delegation fixtures
- provider failure/fallback tests
- evidence provenance and audit replay
- security benchmark baseline across candidate engines
- private deployment path for sensitive tenants
- measured cost/latency/quality routing thresholds

## Research basis — 2026-09

W3C AI Agent Protocol Community Group scope and September meetings indicate active work on interoperable agent discovery, identity, collaboration, transaction security and protocol interoperability. This is Community Group work, not a W3C Recommendation.

IETF draft-asor-wimse-agent-delegation-chain-01 (2026-09-03) proposes verifiable attenuated delegation for AI-agent chains. It is an individual Internet-Draft with no formal IETF standards standing. Sentinel should track the security properties, not hard-code the draft.

IETF draft-seymour-wimse-connected-flight-00 (2026-09-18) is another individual Internet-Draft exploring chained agent trust in a zero-trust setting. It is research signal, not a standard dependency.


## Research delta — 2026-09-25

Fresh standards watch strengthens three V1 requirements without introducing any platform dependency:

- Agent identity must bind an agent to a controlling entity and an explicit authorization scope. W3C Agent Identity Registry Protocol work is incubating cryptographically verifiable cross-organization identity; this remains Community Group work, not a W3C Recommendation.
- Delegation must be capability-bound and attenuating. New September IETF individual drafts explore scoped context disclosure, verifiable delegation chains, and enforcement-point verification. These drafts have no formal IETF standards standing and are inputs to adapters/threat fixtures, not canonical dependencies.
- Authorization decisions need replayable audit evidence. September WIMSE document listings now include work on an AI-agent authorization audit-record format. Sentinel's canonical evidence envelope should therefore preserve principal/controller, delegate, authority scope, constraints, decision, evidence references, timestamps, protocol/version and cryptographic verification metadata.
- Credential ecosystems require explicit privacy threat handling. The 24 September 2026 VC Data Model Threat Model v2.1 is a W3C Group Note Draft and reinforces treating issuer/holder/verifier boundaries, correlation and disclosure as security inputs.

### V1 implementation consequence

The first provider-neutral runtime interface will expose model reasoning only behind Sentinel-owned identity, authority, evidence and policy envelopes. External protocol formats are normalized at adapters; no draft-specific object becomes the internal source of truth.
