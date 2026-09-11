# Koschei Sentinel Cybersecurity Curriculum v1

This curriculum upgrades Sentinel from contract imitation toward reviewed cybersecurity reasoning. It does not change the signed deterministic-verdict authority boundary.

## Objective

Train Sentinel to analyze bounded evidence, reason about vulnerabilities and attack paths, explain uncertainty, and propose verifiable remediation while preserving evidence citations, confidence ceilings, privacy, and abstention.

## Curriculum domains

1. Secure code review
   - authentication and authorization flaws
   - injection classes and unsafe deserialization
   - memory-safety and concurrency defects
   - secrets, configuration, and cryptographic misuse
   - dependency and supply-chain risk

2. Web3 and smart-contract security
   - access-control and privilege boundaries
   - reentrancy and callback hazards
   - oracle and price-manipulation reasoning
   - signature, replay, nonce, and domain-separation errors
   - upgradeability and proxy risks
   - cross-chain and bridge trust assumptions
   - token-accounting, precision, and invariant failures

3. Infrastructure and cloud security
   - IAM and least privilege
   - network exposure and segmentation
   - container and CI/CD hardening
   - secret-management and key-rotation failures
   - logging, detection, and recovery gaps

4. Vulnerability triage
   - exploitability prerequisites
   - affected surface and blast radius
   - evidence-backed severity
   - false-positive reduction
   - remediation prioritization

5. Incident and attack-path reasoning
   - reconstruct paths only from supplied evidence
   - distinguish observed facts from hypotheses
   - identify missing telemetry and decisive next evidence
   - produce containment and remediation options

6. Remediation engineering
   - minimal safe patch
   - defense-in-depth patch
   - regression tests
   - rollback and compatibility considerations

## Supervision ladder

Training examples should be promoted through the following ladder:

- L0 Contract: deterministic baseline targets teach output schema and hard authority boundaries.
- L1 Reviewed explanation: expert-reviewed commentary over existing evidence packets.
- L2 Comparative reasoning: multiple candidate explanations ranked by grounding, correctness, and uncertainty handling.
- L3 Remediation: reviewed patches, tests, and mitigation plans tied to the exact finding.
- L4 Adversarial robustness: misleading evidence, incomplete telemetry, contradictory signals, prompt injection, and poisoned context where the correct behavior is to reject unsupported conclusions.

L0 examples must never dominate mature releases. Production-quality releases should contain a majority of L1-L4 reviewed examples.

## Example quality requirements

Every supervised example must:

- preserve the `sentinel.case.v1` -> `sentinel.opinion.v1` boundary;
- cite only known `evidence_id` values for factual claims;
- separate fact, inference, limitation, and recommendation;
- keep claim confidence at or below the weakest supporting evidence;
- abstain when decisive evidence is missing;
- contain no raw secrets, credentials, personal identifiers, or private production payloads;
- keep related incidents and vulnerability families in a single split group to prevent leakage;
- include provenance and reviewer status outside the model-visible prompt where supported by the dataset contract.

## Defensive-use data policy

Sentinel training data should prioritize authorized labs, synthetic fixtures, public vulnerability disclosures, reviewed internal findings, secure-code examples, and remediation artifacts. Do not curate examples whose value depends on unauthorized access to live systems, credential theft, persistence, evasion, destructive actions, or operational targeting.

## Release mixture target

For a mature cybersecurity release, target the following mix by reviewed training examples:

- 25% secure code review
- 30% Web3 and smart-contract security
- 15% infrastructure/cloud
- 10% vulnerability triage
- 10% incident/attack-path reasoning
- 10% remediation engineering

Within every domain, include straightforward positives, hard negatives, ambiguous cases, and abstention cases.

## Evaluation gates

A trained candidate is not promotable unless it passes independent held-out evaluation for:

- evidence grounding
- authority preservation
- confidence calibration
- abstention quality
- privacy and secret non-disclosure
- vulnerability classification correctness
- attack-path consistency
- remediation correctness
- regression-test usefulness
- prompt-injection resistance

Track domain scores separately. A strong average must not hide a critical failure in authority, privacy, or grounding.

## 397B / 35B-active target

The current small-model QLoRA lane is a systems-validation and curriculum-learning lane, not the final architecture. Keep the 397B total / 35B active target independent from small-model experiments. Dataset contracts, curriculum labels, evaluation suites, and promotion gates should remain model-size agnostic so they can transfer to the eventual sparse model.
