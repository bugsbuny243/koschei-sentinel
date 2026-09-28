# Web4–Web8 Research Update — Koschei Sentinel — 2026-09-28

## Web4: strongest signal
SAMP draft-efstathiou-samp-agent-management-03 (2026-09-26) proposes an operational management plane for heterogeneous AI agents: discovery, state, event streams, autonomy classes, enrollment, trust state and policy-gated operations.
Sentinel implication: useful schema inspiration for observe/shadow telemetry; management telemetry must not become deterministic Web3 verdict authority.
Source: https://datatracker.ietf.org/doc/html/draft-efstathiou-samp-agent-management-03

Agent Action Capsules (2026-09-26) record agent action disposition, deterministic constraints, confirmed effects and human-in-the-loop state.
Sentinel implication: high-value evaluation/replay corpus format that separates observation from intent claims.
Source: https://datatracker.ietf.org/doc/html/draft-mih-scitt-agent-action-capsule-05

CLC (2026-09-27) defines deterministic capability evaluation including allow_unresolved. Sentinel should remain advisory/shadow while deterministic policy retains authority.
Source: https://datatracker.ietf.org/doc/draft-wei-capability-language-core/00/

JEP (2026-09-26) provides signed delegation/judgment/termination/verification events useful for provenance and delegation-chain evaluations.
Source: https://datatracker.ietf.org/doc/html/draft-wang-jep-judgment-event-protocol-07

## Web5
W3C Verifiable Credentials Data Model v2.1 Working Draft updated 2026-09-27. Sentinel may consume VC verification as bounded evidence; valid identity does not imply benign behavior or unlimited authority.
Source: https://www.w3.org/TR/2026/WD-vc-data-model-2.1-20260927/

## Web6
HPKE revision draft-ietf-hpke-hpke-05 (2026-09-26) and Conformance Continuity (2026-09-27) are relevant to crypto-suite provenance and evaluation metadata across changing baselines.
Sources:
https://datatracker.ietf.org/doc/draft-ietf-hpke-hpke/05/
https://datatracker.ietf.org/doc/draft-hillier-conformance-continuity/

## Web7 / Web8
No accepted IETF/W3C generation standards named Web7 or Web8 identified; keep these as research labels.

## Additional agent-security watch

### AAuth Protocol v11

AAuth introduces key-bound agent identity, mission-scoped governance and multiple agent-to-resource authorization modes.

Sentinel implication:
- model telemetry should distinguish authenticated agent, represented person/principal, mission scope and actual authorization;
- anomaly models must not treat a valid identity token as permission for the observed action.

Source:
https://datatracker.ietf.org/doc/html/draft-hardt-oauth-aauth-protocol-11

### Agent Execution Protocol (AEP)

AEP defines a governing enforcement boundary around agent execution and tamper-evident transition recording.

Sentinel implication:
- evaluate deviations between proposed intent, granted authority, attempted action and observed effect;
- keep Sentinel scoring advisory while deterministic enforcement remains outside the model.

Source:
https://datatracker.ietf.org/doc/draft-sato-soos-aep/

### A2A roadmap

A2A's current roadmap emphasizes v1.1 task semantics, bidirectional streaming, multi-turn/human-in-the-loop workflows and validation tooling.

Sentinel implication:
- streaming agent-to-agent sessions create a useful detection surface for delegation drift, scope changes and unexpected artifact/tool transitions;
- telemetry must be evidence-bound and caller-scoped.

Sources:
https://a2a-protocol.org/latest/roadmap/
https://a2a-protocol.org/dev/specification/

### MCP security proposal watch

Active proposals cover signed capability declarations, tamper-evident audit records, asynchronous approval and structured authorization denials.

Sentinel implication:
- useful future features for model evaluation datasets and security telemetry;
- proposal status means observe/research only, not production trust authority.

Sources:
https://github.com/modelcontextprotocol/modelcontextprotocol/pulls
https://plan.modelcontextprotocol.io/seps
