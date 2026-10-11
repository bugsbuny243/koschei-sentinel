# Sentinel Web3–Web8 research note — 2026-10-11

Scope: research and evaluation ideas for the Sentinel security reasoning model. These are not implemented features.

## Primary sources, published 10 October 2026
- Agent Identity Protocol revision 02: https://datatracker.ietf.org/doc/draft-prakash-aip/
- Evaluation State in Agent Protocol Decisions revision 01: https://datatracker.ietf.org/doc/draft-konda-agentproto-evaluation-state/
- Agentic AI Use Cases and Requirements revision 03: https://datatracker.ietf.org/doc/html/draft-agentic-ai-usecases-requirements-03

All three are individual Internet-Drafts, not approved RFCs. Exact release hours were not confirmed.

## Research opportunities
1. Delegation verification consistency. Evaluate whether an AI model distinguishes identity verification, delegation limits, status freshness, and permission to perform a requested action. The AIP author reports incomplete implementation of the latest status-checking requirement. Benchmark proposal: DELEGATION_CONSISTENCY.
2. Evidence state preservation. Distinguish confirmed-positive, confirmed-negative, unavailable, outdated and irrelevant inputs. Preserve the difference between an evaluation result and a real-world observed outcome. Benchmark proposal: EVIDENCE_STATE_PRESERVATION.
3. Dialog continuity and privacy. Distinguish conversation correlation metadata from private task content when an agent communicates across services. Benchmark proposal: DIALOG_PRIVACY_CONTINUITY.

## Product and market implications
Opportunity: evidence-aware agent security reasoning and benchmark datasets.
Threat: inaccurate conclusions from outdated evidence or incomplete delegation chains.
Competitor differentiation hypothesis: a model that explicitly reports uncertainty and the evidence behind decisions; market validation remains necessary.
Implementable experiment: small annotated benchmark set with expected outcomes, source timestamps, and explanations. No runtime or code changes in this note.

## Web4–Web8 classification
No new universal W3C/IETF internet-generation standard named Web4, Web5, Web6, Web7 or Web8 was identified. These draft documents do not establish such a standard.
