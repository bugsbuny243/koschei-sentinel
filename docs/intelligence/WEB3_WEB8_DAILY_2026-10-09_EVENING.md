# Koschei Sentinel — 2026-10-09 Evening Intelligence

**Window:** 2026-10-08 18:00 – 2026-10-09 18:00 Europe/Istanbul. Sentinel is a **cybersecurity foundation model**, not a scanner. All items are proposed reasoning/evaluation datasets, not deployed capabilities.

## Verified source 1: CCIP Vault Adapters (2026-10-08)
https://chain.link/blog/introducing-ccip-vault-adapters
https://docs.chain.link/solutions/cross-chain-vault-adapter

Technical docs explicitly state that CCIP Explorer **Success** may coexist with application-level `MessageFailed`; tokens remain held in the adapter until separately recovered. Failure processing, gas for return messages, cross-chain share delivery, and source/hub finality each carry independent state.

### Benchmark: TRANSPORT_SUCCESS_APPLICATION_FAILURE
Positive and negative case families:
- Transport delivered and vault shares actually minted/delivered.
- Transport delivered, vault execution reverted, assets retained by adapter.
- Return-leg gas insufficient; request recorded as failed, not refunded.
- Successful recovery event, but no proof of final share delivery.
Expected model behavior: infer neither vault completion nor asset loss from transport status alone. Cite distinct receipts/events, avoid conflating application and bridge finality, abstain on absent chain evidence.

**Threat:** incorrect settlement assurance and delayed recovery.
**Opportunity:** model evaluation for multi-layer causality and evidence reconciliation.
**Competitor-gap hypothesis:** event-only alerting can miss cross-layer outcomes; evaluate against concrete baselines.
**Product idea:** evidence-grounded cross-chain incident reasoning suite with outcome-state labels and confidence calibration.

## Verified source 2: Riskified Agent Identity Risk Intelligence (2026-10-08)
https://ir.riskified.com/news-releases/news-release-details/riskified-launches-agent-identity-risk-intelligence-know-who

Vendor announced agent-originated commerce risk evaluation; **limited beta planned December 2026**, GA expected 2027. It distinguishes agent authorization from underlying principal/transaction risk. These are vendor product claims, not an independent benchmark or ratified standard.

### Benchmark: AUTHORIZED_AGENT_UNTRUSTED_INTENT
Case families:
- Valid agent delegation + benign policy-compliant order.
- Valid agent delegation + transaction outside risk policy.
- Delegation by a compromised principal; identity/authorization proof alone is insufficient.
- Risk score high but supporting evidence absent; model must not treat score as ground truth.
Expected: separate `identity proof`, `delegated authority`, `behavioral risk`, `policy approval`, `external effect`. Respect privacy and uncertainty.

**Threat:** signed permission confused with safe intent.
**Opportunity:** stronger model reasoning over fraud, authorization and provenance without claiming certainty.
**Competitor-gap hypothesis:** authorization-only evaluation does not capture behavior and effect; validate with test suites.
**Product idea:** benchmark of signed-authority versus effect-policy continuity with explicit evidence requirements.

## Research-status watch (not new adoption)
CFRG call for Longfellow ZK adoption closes October 9; do not label as adopted or standardized without chair outcome.
https://mailarchive.ietf.org/arch/msg/cfrg/WIndEoOTVff7hjH-kXoi4XRqHnU/

## Training/evaluation specification
Store source URL, source timestamp, claim-vs-observation tag, expected answer, counterfactual, missing-evidence abstention criterion, and evaluator rationale. Avoid operational offensive instructions. Separate announced features from verified deployments.

## Web4–Web8
No accepted universal W3C/IETF internet-generation standards with these names verified. Vendor products and individual Internet-Drafts are not Recommendations/RFCs.
