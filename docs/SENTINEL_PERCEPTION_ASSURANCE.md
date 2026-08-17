# Koschei Sentinel Perception Assurance

Sentinel's active-defense path must distinguish three different things:

1. what a sensor actually observed,
2. what Sentinel inferred or predicted from those observations, and
3. what defensive authority is justified by independently corroborated evidence.

The production perception path is therefore fail-closed and provenance-bound.

## Canonical production path

`sanitized telemetry -> perception adapter -> adapter receipt -> source admission -> admitted batch -> multi-sensor fusion -> assurance summary -> Cyber State Graph + graph receipt -> assured active defense -> interception plan -> assured connector envelope`

Each boundary has a different job. Skipping a boundary is not equivalent to passing it.

## 1. Sanitized telemetry

Vendor collectors must remove secrets before events enter the Sentinel adapter layer. `SanitizedTelemetryEvent` rejects secret-bearing field names such as passwords, private keys, mnemonics, seed phrases and access tokens.

The adapter layer is intentionally offline-normalization-only. An adapter descriptor declares:

- its immutable adapter identity,
- vendor/product identity,
- supported telemetry source types,
- deterministic operation,
- no network access, and
- reject-secret handling.

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

The assurance summary records how many independent domains are represented, but global diversity alone does not authorize active defense.

## 6. Graph binding

`sentinel-cyber-perceive` emits both the Cyber State Graph and a `PerceptionGraphReceipt`.

The receipt binds:

`fused perception batch SHA-256 -> graph SHA-256 -> entity/relation counts -> receipt SHA-256`

This prevents an assurance summary from one telemetry batch being attached to a different graph.

## 7. Assured active defense

`sentinel-assured-defense-plan` is the production active-defense planner. It requires the Cyber State Graph, its graph receipt, the perception assurance summary, the exact source registry, and protected critical entity IDs when applicable.

The base planner may detect a high-confidence attack, but the assurance gate can only reduce authority; it never raises it.

Default policy:

- fewer than 2 independent domains in the relevant attack component -> at most `GUARD`,
- at least 2 independent domains -> at most `COMBAT`,
- at least 3 independent domains -> `SIEGE` may remain available if the base planner already justified it.

The key phrase is **relevant attack component**. Sentinel computes the connected active subgraph around declared critical assets, or around the highest-impact defensive cut point when no critical root is present. Telemetry from an unrelated component cannot be used to unlock Combat or Siege.

## 8. Production connector envelope

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
  -> sentinel-assured-defense-plan
```

The actual vendor execution adapter should consume only an assured connector envelope after the interception execution state authorizes the next step.

## Current boundary

These modules establish contracts, validation, deterministic hashes, tests, and CLI planning. They do not by themselves connect to a live EDR, cloud control plane, wallet, signer, or firewall. Real vendor adapters remain a separate deployment layer and must preserve the same protected-scope and outcome-verification rules.
