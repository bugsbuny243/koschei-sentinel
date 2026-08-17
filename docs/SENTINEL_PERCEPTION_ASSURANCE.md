# Koschei Sentinel Perception Assurance

Sentinel's active-defense path must distinguish four different things:

1. what a sensor actually observed,
2. what Sentinel inferred or predicted from those observations,
3. which observations belong to the same attack component or temporal world-line, and
4. what defensive authority is justified by independently corroborated evidence for that component.

The production perception path is therefore fail-closed, provenance-bound, component-scoped, and temporal.

## Canonical production path

`sanitized telemetry -> perception adapter -> adapter receipt -> source admission -> admitted batch -> multi-sensor fusion -> assurance summary -> Cyber State Graph + graph receipt -> attack components -> world-lines -> assured multi-incident defense -> interception plan -> assured connector envelope`

Each boundary has a different job. Skipping a boundary is not equivalent to passing it.

## 1. Sanitized telemetry

Vendor collectors must remove secrets before events enter the Sentinel adapter layer. `SanitizedTelemetryEvent` rejects secret-bearing field names such as passwords, private keys, mnemonics, seed phrases and access tokens.

The adapter layer is intentionally offline-normalization-only. An adapter descriptor declares its immutable adapter identity, vendor/product identity, supported telemetry source types, deterministic operation, no network access, and reject-secret handling.

## 2. Perception adapters

Adapters normalize telemetry into `PerceptionObservation` records. They are not allowed to decide that an observation is an inference or prediction.

The compiled perception graph therefore creates only `OBSERVED` relations. `INFERRED`, `PREDICTED`, and `DISPROVED` states belong to later reasoning layers.

Adapter output is checked again by Sentinel. An adapter cannot silently change source type, source instance, evidence digest, observation namespace, or evidence namespace.

Reference adapters currently cover vendor-neutral examples for endpoint/process, cloud IAM, CI/CD, and signer/wallet telemetry. Their relation names are aligned with the attack-progression vocabulary so perception and reasoning use the same semantics.

## 3. Source enrollment and admission

A syntactically valid observation is not automatically trusted for production fusion.

`PerceptionSourceRegistry` enrolls an exact source principal of the form `SOURCE_TYPE:source_instance` and binds it to allowed adapter IDs, an independence domain, and enabled/disabled admission state.

`sentinel-perception-admit` re-verifies the entire adapter result:

`adapter result digest -> receipt -> observation IDs -> evidence SHA-256 -> source enrollment -> adapter permission`

The admitted envelope preserves the original adapter result so downstream fusion can re-run admission verification instead of trusting a detached receipt.

## 4. Multi-sensor fusion

Production fusion consumes admitted envelopes and the source registry. Raw `PerceptionBatch` fusion is development-only and requires the explicit `--allow-unadmitted-dev` flag.

Fusion rejects replayed or duplicate observation IDs, duplicate adapter batches, evidence IDs reused with different digests, conflicting entity types, conflicting entity labels, and forged or stale admission receipts.

Fusion does not raise confidence simply because the same source repeats an event many times.

## 5. Independence domains

Two evidence IDs are not necessarily two independent pieces of evidence. Two alerts emitted by the same EDR control plane may share the same failure or compromise domain. Source enrollment therefore assigns every source principal to an `independence_domain`.

Examples include `endpoint-control-plane`, `cloud-control-plane`, `cicd-control-plane`, and `signer-control-plane`.

The assurance summary records how many independent domains are represented, but global diversity alone never authorizes active defense.

## 6. Graph binding

`sentinel-cyber-perceive` emits both the Cyber State Graph and a `PerceptionGraphReceipt`.

The receipt binds:

`fused perception batch SHA-256 -> graph SHA-256 -> entity/relation counts -> receipt SHA-256`

This prevents an assurance summary from one telemetry batch being attached to a different graph.

## 7. Attack components

A Cyber State Graph may contain many unrelated events at the same time. Sentinel must not combine them merely because they arrived in the same telemetry window.

`analyze_attack_progression()` therefore partitions active `OBSERVED` and `INFERRED` relations into connected attack components before it computes stages, predictions, confidence, or defensive cut points.

Each `AttackComponentReport` has its own entities, relations, stage state, progression confidence, predictions, cut points, and risk score.

The legacy top-level `AttackProgressionReport` remains for compatibility, but its stage and cut-point fields now represent one selected primary component rather than an aggregate of unrelated graph regions. A caller may provide focus entities to select a particular component deterministically.

This establishes the rule:

**event != incident, and two disconnected incidents do not become one attack because they share a graph snapshot.**

## 8. Temporal attack world-lines

Component content can change as an attacker moves. A component identifier may therefore change between snapshots even when it represents the same continuing attack.

`sentinel-attack-world-lines` tracks component lineage across ordered graph snapshots using deterministic entity/relation overlap. The timeline distinguishes:

- `NEW`,
- `CONTINUED`,
- `SPLIT`,
- `MERGED`,
- `RECONFIGURED`, and
- `ENDED`.

A one-to-one continuation keeps the same world-line identity. Split and merge events create new line identities while preserving predecessor lineage. This allows Sentinel to recognize attacker rerouting and branching without rewriting history into one artificial straight line.

## 9. Multi-incident defense

A single primary component is not sufficient when several attacks are active at once.

`MultiIncidentDefensePlan` creates one independent `ActiveDefensePlan` per active component. Component boundaries are validated so authorized and withheld cut points cannot reference entities outside that component.

Critical-asset components receive a deterministic priority bonus, but non-critical active attacks remain visible and continue to receive their own plans.

`sentinel-assured-multi-defense` is the production multi-incident path. Each component is passed independently through the perception-assurance gate. Independent evidence from one component cannot raise authority for a disconnected component.

## 10. Assured active defense

The base planner may detect a high-confidence attack, but the assurance gate can only reduce authority; it never raises it.

Default policy per attack component:

- fewer than 2 independent domains -> at most `GUARD`,
- at least 2 independent domains -> at most `COMBAT`,
- at least 3 independent domains -> `SIEGE` may remain available if the base planner already justified it.

The key phrase is **per attack component**. Global source diversity is irrelevant to a component unless those admitted evidence sources support relations inside that component.

This means a three-domain signer attack may retain Siege while a simultaneous one-domain endpoint incident remains Guard. Authority never transfers between disconnected world-lines.

## 11. Production connector envelope

A production defensive connector must not execute a bare model instruction or a bare `DefenseConnectorCommand`.

`AssuredDefenseConnectorEnvelope` binds the protected scope, exact assured defense mode, exact interception plan and step, precondition evidence, active-defense assurance receipt and policy, and deterministic command idempotency.

The target/action pair must already exist as an authorized defensive cut point. Scope checks still apply independently.

This preserves Sentinel's human-protective rule:

**authority exists to stop a verified attacker inside defended scope; uncertainty does not create broader authority.**

## CLI flow

```text
sentinel-perception-adapt
  -> sentinel-perception-admit
  -> sentinel-perception-fuse
  -> sentinel-cyber-perceive
  -> sentinel-attack-world-lines
  -> sentinel-assured-multi-defense
```

For single-component compatibility or debugging, `sentinel-assured-defense-plan` remains available. Production orchestration should prefer the multi-incident path when the graph can contain concurrent attacks.

The actual vendor execution adapter should consume only an assured connector envelope after the interception execution state authorizes the next step.

## Current boundary

These modules establish contracts, validation, deterministic hashes, tests, CLI planning, attack-component isolation, and temporal lineage. They do not by themselves connect to a live EDR, cloud control plane, wallet, signer, or firewall. Real vendor adapters remain a separate deployment layer and must preserve the same protected-scope, component-isolation, and outcome-verification rules.
