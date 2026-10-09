# Sentinel intelligence — 2026-10-09

Sentinel is a cybersecurity foundation-model project, NOT a scanner. These are training/evaluation proposals, not implemented capabilities.

## Candidate source signals
SAIP-12 signed agent identity individual Internet-Draft: https://datatracker.ietf.org/doc/draft-jovancevic-saip/12/ . Not an RFC or accepted standard; verify revision details.
Cosmos Ledger Security 2026.1 PQ announcement: independently confirm release artifacts, network deployment and signature verification semantics before claiming effective PQ protection.
OpenA2A AIP-04 dated October 6: context only; exclude from October 9 last-24h event count.

## New model evaluation suites
### SIGNED_IDENTITY_AUTHORITY_GAP
Input: valid agent identity proof, scoped delegation, tool request, external effect receipt.
Expected: distinguish cryptographic identity verification, current delegation, tool authority, policy limits and finalized outcome. Valid identity does not imply authorized effect.

### PQ_CONSENSUS_MIGRATION_CONTINUITY
Input: wallet ML-DSA support, validator signing policy, KMS version, network activation state, historical signature records.
Expected: distinguish implementation support from configured, activated and verified protection; flag mixed-scheme migration without asserting the whole chain is PQ-secure.

### AGENT_TRUST_EVIDENCE_CONTINUITY
Input: DID, hybrid signature, transparency log, capability claims and behavior signals.
Expected: provenance-aware confidence with no conversion of reputation or signature into permission.

## Threat / gap / implementation
Threat: authority amplification through multi-agent/tool composition and incomplete cryptographic migration.
Competitive hypothesis: benchmarks often evaluate signature correctness without end-to-end authorized effect; needs empirical validation.
Implement adversarial but non-operational evaluation fixtures, positive/negative cases, source timestamps, abstention rules and evidence-chain scoring.

## Classification
Individual IETF Internet-Draft != RFC; vendor architecture != W3C/IETF standard; Web4–Web8 names are not recognized universal internet generation standards.
