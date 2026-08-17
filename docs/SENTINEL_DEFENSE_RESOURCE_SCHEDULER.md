# Koschei Sentinel Defense Resource Scheduler

Koschei Sentinel is designed to defend multiple authorized protected systems at the same time. Detection, authority, resource allocation, execution, and post-action verification are separate decisions and must remain separate in code.

The scheduler does **not** create defensive authority. It allocates finite defensive resources only to cut points that have already passed component-scoped perception assurance and Guard / Combat / Siege policy.

## Canonical production cycle

```text
admitted telemetry
  -> assured perception fusion
  -> bound Cyber State Graph
  -> component-scoped attack progression
  -> assured multi-incident defense
  -> attack world-lines
  -> world-line scheduling context
  -> defense resource schedule
  -> defense wave execution
  -> scheduled assured connector envelope
  -> connector execution receipt
  -> verified outcome evidence
  -> fresh perception and materially changed evidence graph
  -> post-wave reassessment
  -> world-line carry
  -> next resource schedule
```

A completed wave cannot authorize another wave from the same stale perception state.

## 1. Authority remains upstream

`AssuredMultiIncidentDefensePlan` is the authority boundary for resource scheduling. Every disconnected attack component is independently evaluated. Independence-domain evidence from one incident cannot unlock Combat or Siege for another.

The scheduler consumes only an assured multi-incident plan. Priority, waiting time, or critical-resource pressure can never raise a component above its assured defense mode.

## 2. Exact-plan schedule binding

`DefenseResourceSchedule` v3 binds the schedule to `assured_multi_plan_sha256`.

This prevents a schedule created from one assured plan from being replayed against another plan that merely shares the same `graph_id`. Schedule verification checks both the schedule digest and the exact assured-plan digest before a production connector or defense wave can consume it.

## 3. Resource classes and capacity

The default resource classes are:

- evidence collection,
- network controls,
- identity controls,
- endpoint controls,
- cloud controls,
- CI/CD controls,
- signer controls,
- transaction controls, and
- emergency controls.

Each class has an independent capacity. Global parallel capacity is enforced separately. A signer-control bottleneck therefore does not silently consume endpoint-control capacity, and vice versa.

Critical protected assets receive a bounded reservation. Unused critical reservations are released to the global queue rather than wasted.

## 4. One component step per wave

A scheduling wave may contain several attack components in parallel, but each component contributes at most one currently authorized interception step.

This is intentional. Parallelism exists **across** independent incidents. Sequential verification remains mandatory **inside** each incident.

A component cannot execute its second cut point merely because its first cut point succeeded. Sentinel must observe the new state, rebuild the attack component, re-evaluate assurance, and schedule again.

## 5. World-line-aware fairness

Component IDs can change when an attacker reroutes, splits, or merges paths. Fairness cannot therefore be keyed only to an ephemeral component ID.

`DefenseSchedulingContext` maps the current component to a persistent scheduling subject, normally its attack world-line ID.

`SchedulerLineageCarryReceipt` carries waiting age across adjacent world-line transitions:

- continued or rerouted lineage keeps its age,
- split successors inherit predecessor age,
- merge/reconfigured successors inherit the maximum predecessor age, and
- a genuinely new world-line begins at zero.

An attacker cannot reset its scheduling history simply by restructuring the graph.

## 6. Bounded wait aging

Deferred scheduling subjects receive a bounded deterministic priority boost. The boost reduces starvation without overriding criticality, assurance mode, interception urgency, or resource capacity.

Scheduling a subject resets its wait age. A Guard component with no authorized containment cut point consumes no active-defense resource slot.

## 7. Defense Load Range

`sentinel-defense-load-range` stress-tests scheduling under constrained resources without touching live infrastructure.

The load gate measures:

- service coverage,
- critical first-service latency,
- maximum waiting cycles,
- resource-class utilization,
- starvation, and
- capacity violations.

A zero-capacity control plane cannot cause an infinite loop. The range stops deterministically and reports starvation.

Cyber defense promotion v2 requires three independent evaluation families to pass:

1. single-incident Cyber Range,
2. multi-incident / world-line Cyber Range, and
3. Defense Load Range.

A model that reasons correctly but starves critical incidents under resource pressure is not ready for production promotion.

## 8. Parallel Defense Wave Execution

`DefenseWaveExecution` coordinates scheduled components concurrently while retaining one isolated `InterceptionExecution` per component.

For every scheduled component the state transition remains:

```text
PENDING_AUTHORIZATION
  -> AUTHORIZED
  -> EXECUTED
  -> VERIFIED_SUCCEEDED | VERIFIED_FAILED
```

Precondition evidence is mandatory before authorization. Connector execution receipts are mandatory before execution is recorded. Outcome evidence is mandatory before the component becomes terminal.

The wave becomes complete only when every scheduled component has a verified terminal outcome.

A failed component does not block another independent component from completing its own verification.

## 9. Scheduled connector boundary

A production connector cannot consume a deferred component.

`ScheduledAssuredConnectorEnvelope` requires all of the following to agree:

- exact schedule and schedule SHA,
- exact assured multi-plan SHA,
- scheduled component and world-line subject,
- exact interception step,
- action and target,
- supporting relations,
- effective defense mode,
- protected scope, and
- precondition evidence.

The schedule is re-hashed before connector-envelope generation. Editing a schedule while keeping its old digest fails closed.

## 10. Fresh-state reassessment

After a defense wave completes, Sentinel must not immediately replay the same plan.

`sentinel-defense-reassess` requires:

- the completed, hash-verified wave,
- the exact previous schedule and assured plan,
- a newly admitted/fused perception state,
- a bound next graph and graph receipt,
- the matching perception assurance summary, and
- the exact source registry.

Both the fused perception batch SHA and the graph evidence SHA must differ from the previous plan's perception binding. Renaming a batch without changing graph evidence is rejected.

The reassessment gate builds the next `AssuredMultiIncidentDefensePlan` itself and emits a self-verifying receipt binding:

```text
previous wave SHA
  + previous schedule SHA
  + previous assured-plan SHA
  + previous source-batch SHA
  + previous graph SHA
  + next source-batch SHA
  + next graph SHA
  + next assured-plan SHA
```

This enforces the operational rule:

**act once on the current verified world state, verify the result, then see the world again before acting further.**

## Current boundary

The repository now contains deterministic planning, assurance, scheduling, wave coordination, connector-envelope contracts, simulation gates, lineage-aware fairness, and reassessment receipts.

It still does not by itself execute live vendor actions against EDR, cloud, CI/CD, signer, wallet, firewall, or identity systems. Live adapters must preserve protected scope, idempotency, execution receipts, and outcome verification.
