# Koschei Sentinel — Daily Research — 2026-10-08

Window: last 24 hours ending 2026-10-08 18:00 Europe/Istanbul.
Sentinel is a cybersecurity foundation model, not a scanner. External tools produce evidence; Sentinel reasons about it.

## W3C publishes five Verifiable Credential threat-model draft notes — 8 October 2026
New W3C VC Working Group draft notes address Recognized Entities, Data Integrity, credential lifecycle management (VCALM), barcodes and credential rendering. They are informative Group Note Drafts, not W3C Recommendations.

### Benchmark: RECOGNITION_TRUST_ANCHOR_CONTINUITY
Reason over publisher recognition, issuer, credential, holder, verifier trust policy and action. A valid signature does not independently establish recognition freshness or scope.
Opportunity: train evidence-grounded trust-chain reasoning.
Threat: an out-of-date recognition record can be mistaken for current permission.
Competitor gap (hypothesis): identity-only validation does not always account for downstream effects.
Action: build evaluation cases with changed recognition, distinct verifier policies and missing trust-anchor evidence.

### Benchmark: CREDENTIAL_STATUS_PRIVACY_FRESHNESS
Reason about status retrieval timing, caching, freshness, availability and metadata disclosure.
Opportunity: teach trade-offs among evidence freshness and privacy.
Threat: credential verification can reveal information about the verifier even if the signed content remains confidential.
Competitor gap: proof verification alone does not cover retrieval metadata.
Action: evaluation set comparing direct retrieval, cached status and holder-provided evidence.

### Benchmark: SIGNED_CLAIM_RENDERING_GAP
Distinguish cryptographically covered claims from fields displayed to a human or consumed by another agent.
Opportunity: improve evidence-to-decision reasoning.
Threat: presentation can be incomplete or misleading despite valid source proof.
Competitor gap: source validation and displayed interpretation are often separate checks.
Action: compare verified claims with rendered representations and identify any unsupported displayed claim.

### Benchmark: CREDENTIAL_LIFECYCLE_PROTOCOL_CONFUSION
Reason across credential issue, presentation, verification, status and permission scope.
Opportunity: model multi-stage consistency.
Threat: a correct isolated protocol step may not imply a correct end-to-end decision.
Action: request causal explanation, evidence timestamps and uncertainty.

## W3C SHACL 1.2 Core and SPARQL 1.2 RL Working Drafts — 8 October 2026
SHACL shapes validate RDF graph structure. SPARQL-RL proposes Datalog-style derivation and stratification. Both are drafts.

### Benchmark: SEMANTIC_INFERENCE_PROVENANCE
Keep observed evidence, signed claims, asserted facts and rule-derived conclusions distinct.
Opportunity: a graph-validation tool can support foundation-model reasoning.
Threat: inferred relationships can be misrepresented as directly observed facts.
Competitor gap (hypothesis): explainable provenance across inference and execution is valuable beyond a simple risk label.
Action: design evaluation cases that record source graph, ruleset version, derived fact and evidence limitations.

## Status
VC Data Model v2.0 and SHACL 2017 are W3C Recommendations. Today's new documents are drafts. No new universal Web4–Web8 internet-generation standard found.

## Primary sources — 2026-10-08
https://www.w3.org/news/2026/first-draft-notes-verifiable-credential-threat-models/
https://www.w3.org/TR/vc-recognized-entities-threat-model-1.0/
https://www.w3.org/TR/vc-data-integrity-threat-model-1.1/
https://www.w3.org/TR/vcalm-threat-model-1.0/
https://www.w3.org/TR/vc-barcodes-threat-model-1.0/
https://www.w3.org/TR/vc-render-method-threat-model-1.0/
https://www.w3.org/TR/shacl12-core/
https://www.w3.org/TR/sparql12-rl/

Documentation and benchmark design only; no model or runtime change.
