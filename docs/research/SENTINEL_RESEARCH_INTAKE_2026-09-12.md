# Sentinel Research Intake — 2026-09-12

Status: active research intake
Scope: current Web3-to-Web10 signals, normalized into concrete security primitives rather than marketing labels.
Driver policy: Sentinel Driver remains unchanged by this intake. This file feeds research, curriculum and evaluation planning only.

## Priority A — adopt into research/eval planning

### 1. ERC-8004 trustless agents
Observed signal: Ethereum's agent stack now includes portable agent identity, reputation and validation registries. Ethereum.org reports ERC-8004 deployments across 16 networks.

Sentinel primitives to learn/evaluate:
- agent identity provenance;
- reputation-signal poisoning resistance;
- independent validation evidence;
- chain/network identity binding;
- trust-score uncertainty and abstention;
- validator disagreement handling.

Evaluation ideas:
- detect spoofed or mismatched agent identity;
- refuse unsupported reputation conclusions;
- distinguish registration, reputation and validation evidence;
- explain why a validator signal is insufficient for a deterministic verdict.

### 2. A2A v1.0 agent interoperability
Observed signal: A2A v1.0 is a production-ready open agent-to-agent standard with discovery, Agent Cards, multiple protocol bindings, authorization boundaries, TLS requirements and signed-card support.

Sentinel primitives to learn/evaluate:
- Agent Card validation;
- capability/skill trust boundaries;
- authentication vs authorization distinction;
- task/artifact access control;
- signed metadata and downgrade/tamper detection;
- cross-agent context minimization.

Evaluation ideas:
- malicious or over-privileged Agent Cards;
- auth scope mismatch;
- capability confusion;
- task-history leakage;
- unsigned/altered capability metadata.

### 3. MCP 2026 authorization hardening
Observed signal: the 2026 MCP release candidate moves toward a stateless core, first-class extensions, Tasks and authorization hardening.

Sentinel primitives to learn/evaluate:
- tool authorization boundaries;
- token/audience/resource scoping;
- extension trust boundaries;
- task provenance;
- confused-deputy and over-broad tool grants;
- credential isolation between transports and tools.

Evaluation ideas:
- tool call with excessive scope;
- mismatched resource audience;
- extension requesting undeclared authority;
- state/provenance loss between agent and tool boundaries.

### 4. IETF AI-agent delegation architecture
Observed signal: active 2026 Internet-Drafts explicitly call for separation of agent identity from principal identity, verifiable multi-hop delegation, constrained authority, revocation, provenance and auditability. These are drafts, not finalized IETF standards.

Sentinel primitives to learn/evaluate:
- principal/agent identity separation;
- multi-hop delegation chains;
- authority attenuation at each hop;
- time/value/context constraints;
- revocation and short-lived authority;
- provenance-preserving execution graphs.

Evaluation ideas:
- detect delegation escalation;
- detect a missing principal in a multi-hop chain;
- flag a re-delegation that violates parent constraints;
- reconstruct the effective authority for a request.

### 5. Post-quantum migration and crypto-agility
Observed signal: NIST states finalized PQC standards are ready to implement and that migration should begin now. 2026 PIV work uses a dual-stack transition model around ML-DSA and ML-KEM.

Sentinel primitives to learn/evaluate:
- algorithm inventory and crypto-agility;
- classical/PQC dual-stack transition;
- signature and KEM migration risk;
- downgrade resistance;
- long-lived key/data exposure analysis;
- evidence-based cryptographic recommendations.

Evaluation ideas:
- identify non-agile signature assumptions;
- detect unsafe downgrade paths;
- distinguish finalized PQC standards from candidates/drafts;
- require explicit evidence before declaring a system quantum-safe.

## Priority B — monitor, do not treat as authoritative standards

### Agent Identity Protocol drafts
Multiple active individual Internet-Drafts propose decentralized or verifiable agent identity and delegation. Useful concepts include DIDs, capability tokens, chained delegation and invocation-bound provenance. Treat them as research inputs, not standards truth.

### Web4/Web6/Web7 terminology
Keep these labels as discovery buckets. Promote only concrete protocols, interoperable specifications, implementations, security research or independently verifiable primitives into training data.

### Web8-Web10 terminology
Discovery-only until concrete protocol/code/evidence exists. Never train Sentinel to infer technical maturity from a generation label.

## Dataset admission rule

A research item may enter expert-reviewed training data only when all of the following are captured:
1. canonical source and publication/version date;
2. maturity class: standard / deployed protocol / draft / research / speculative;
3. concrete security primitive;
4. threat model or failure mode;
5. evidence required for Sentinel to make a claim;
6. expected abstention condition;
7. leakage-safe family/group identifier so related examples cannot cross train/validation/test splits.

## Research-to-model objective

Sentinel should not memorize that "WebN" is secure or advanced. It should learn to determine:
- who/what is acting;
- under whose authority;
- through which delegation and protocol boundary;
- against what verified evidence;
- with which cryptographic assumptions;
- what can and cannot be concluded;
- what remediation or additional evidence is required.

This preserves Sentinel's deterministic-verdict boundary while expanding the model's cybersecurity reasoning depth.
