# Koschei Sentinel — Web4 Security Intelligence Baseline

Date: 2026-09-06
Status: research baseline; not a training completion claim
Branch: `research/web4-security-intelligence`

## Executive conclusion

`Web4` is not one settled protocol or one globally agreed technical standard. In 2026 the label is being used for at least two overlapping directions:

1. physical/virtual convergence: AI + IoT/edge + XR/spatial computing + digital twins + high-performance networks + distributed systems + blockchain/identity;
2. agentic/verifiable/federated architectures: autonomous software/AI agents as first-class participants, machine-verifiable identity/artifacts, federated interconnection, provenance and conformance.

Sentinel must therefore learn **definition provenance** before it learns Web4 security. A claim about Web4 is incomplete unless the source family and revision status are known.

## Source-status rules

- European Commission Web 4.0 material is treated as authoritative policy/research framing, not as an Internet protocol specification.
- W3C Verifiable Credentials 2.0 is a W3C Recommendation and may be treated as a published Web standard for credential semantics.
- `draft-jacobs-web4-terminology-00` and `draft-reilly-web4-orion-00` are individual IETF Internet-Drafts. They are research inputs only and must never be labeled as IETF standards or Internet consensus.
- NIST AI-agent identity/security work is authoritative security research/guidance, while draft documents retain their draft status in all metadata.
- OWASP agentic-AI guidance is reviewed practitioner guidance; it is useful for threat taxonomies but must not replace primary incident evidence.
- ENISA threat-landscape material provides general threat priors and incident context; it must not be mislabeled as Web4-specific evidence.

## Web4 security model for Sentinel

The existing Sentinel representation remains the backbone:

`Entity -> Event -> Intent -> Capability -> Vulnerability -> Action -> Consequence -> Evidence`

Web4 extends the entity graph with:

- human
- AI/software agent
- avatar
- IoT device / sensor
- robot
- digital twin
- edge node
- identity issuer / credential holder / credential verifier
- wallet / smart contract
- service / tool
- data artifact

The important change is that authority can now move across human, agent, device, credential, wallet, and physical-world boundaries in one attack chain.

## Priority attack families

### Agent identity and authority

- agent identity spoofing
- authorization-scope drift
- delegation-chain confusion
- confused-deputy behavior
- excessive tool privilege
- non-repudiation failure
- cross-agent trust abuse

### Agentic AI

- direct and indirect prompt injection
- tool abuse and privilege escalation
- data exfiltration through tools or agent outputs
- memory/context poisoning
- persistent malicious state
- supply-chain poisoning of agent tools, skills, dependencies or external knowledge

### Verifiable identity and artifacts

- credential or attestation forgery
- stale/replayed credentials
- revocation-status bypass
- issuer/holder/verifier confusion
- provenance replay
- origin/integrity/time evidence mismatch
- cryptographic identity vs semantic authority confusion

### Physical/digital convergence

- digital-twin state poisoning
- sensor/reality-capture spoofing
- edge-state desynchronization
- virtual-to-physical authority confusion
- robotic action triggered from untrusted virtual state
- spatial/XR privacy exposure
- avatar impersonation linked to real-world permissions

### Web3/economic crossover

- wallet-agent delegation abuse
- autonomous transaction overreach
- signing-intent mismatch
- credential-to-wallet identity confusion
- agent-to-contract authorization errors
- cross-chain identity/authority drift

## Sentinel training objectives

Sentinel should learn to answer questions such as:

1. Which entity actually possessed authority at each step?
2. Was authority delegated, inferred, forged, replayed, or widened?
3. Which evidence proves identity, and which evidence merely increases confidence?
4. Did a digital event create a physical-world consequence?
5. Did an AI agent use a tool outside the user's intended scope?
6. Did a persistent memory/state mutation alter later agent behavior?
7. Is an artifact authentic, current and attributable, or only plausibly related?
8. Is a source describing a standard, a draft, a policy vision, a research proposal, or an observed incident?

## Evaluation families

Minimum Web4 benchmark families:

- Web4 definition disambiguation
- draft-vs-standard classification
- agent identity/authority reasoning
- delegation-chain reconstruction
- agentic attack-chain reconstruction
- digital-twin integrity reasoning
- physical/digital consequence reasoning
- credential/provenance validation
- prompt-injection/tool-abuse detection
- memory-poisoning detection
- wallet-agent authority reasoning
- false-positive resistance
- evidence grounding
- uncertainty calibration
- defensive recommendation

## Training admission policy

No source in `configs/corpus/web4-v1.sources.proposed.jsonl` is currently approved for model training.

Before a source can move into a training dataset it must have:

- immutable revision or snapshot identity
- SHA-256
- provenance tier
- license/right-to-use review
- timestamp
- category
- deduplication plan
- benchmark-overlap assessment
- training/eval split policy
- explicit `training_authorization=true`

Web content must never be ingested directly into Sentinel's self-improvement loop merely because it was retrieved or summarized by an AI model.

## Training sequence

1. Build and validate Web4 source registry.
2. Snapshot primary/authoritative sources with hashes.
3. Normalize into Sentinel's entity/event/intent/capability/vulnerability/action/consequence/evidence representation.
4. Produce reviewed training cases and an isolated unseen evaluation set.
5. Create security-specific Web4 regression benchmarks.
6. Run CPU/cheap pipeline tests only.
7. Add the approved Web4 mixture to the next legitimate model-training run.
8. Evaluate Web4 results separately before model promotion.

## Current state

VERIFIED

- Web4 research branch isolated from the active 397B/Gold PR chain.
- Web4 research curriculum exists.
- Initial proposed source registry exists.
- Training authority remains disabled for all Web4 sources.

NOT VERIFIED

- no Web4 corpus snapshot has been license-approved
- no Web4 dataset artifact has been materialized
- no Web4 benchmark has yet been human-reviewed
- no model training or fine-tuning has been executed for Web4
- no claim is made that either individual IETF Web4 draft will become an RFC or standard

NEXT

Build the first provenance-pinned Web4 research snapshot and transform it into reviewed Sentinel security examples without contaminating HOLDOUT evaluation data.
