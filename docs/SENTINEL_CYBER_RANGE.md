# Koschei Sentinel Cyber Range

The Sentinel Cyber Range is a deterministic defense-behavior laboratory for the active-defense stack. It does not execute real endpoint, cloud, wallet, network or protocol actions. It evaluates the decisions Sentinel would make against simulated Cyber State Graph timelines.

## Purpose

The range measures whether Sentinel can:

1. identify credible hostile progression from evidence-backed graph relations,
2. select a useful defensive cut point,
3. remain inside Guard / Combat / Siege authority,
4. sequence containment rather than firing many actions at once,
5. require post-action verification before advancing,
6. detect when an attacker reroutes after containment,
7. avoid high-impact containment in benign scenarios, and
8. reach verified containment quickly enough to protect critical assets.

## Scenario model

A range scenario contains:

- a stable incident `graph_id`,
- one or more Cyber State Graph snapshots representing time ticks,
- ground-truth classification (`MALICIOUS` or `BENIGN`),
- protected critical entity IDs,
- optional expected reroute ticks, and
- optional simulated containment outcomes.

Every tick is independently planned from the current graph. Authorization is never inherited from a prior tick.

## Measured gates

The initial suite policy is intentionally strict:

- malicious containment rate: at least 95%,
- attacker-reroute detection rate: at least 90%,
- benign scenarios with high-impact false-positive containment: at most 1%,
- mean verified-containment tick: at most 2.0.

These are development gates, not claims of production performance. A candidate model or planner must be evaluated on sealed scenario suites before promotion.

## Safety boundary

The range simulates defensive effects only. It does not generate exploit payloads, compromise external systems, or perform hack-back. Simulated actions are limited to defensive controls already modeled by Sentinel, such as credential revocation, endpoint isolation, workload quarantine, pipeline pause, signer freeze, transaction hold and indicator blocking inside defended scope.

## Learning loop

Range failures are valuable training material. A failed scenario should be converted into a reviewed training trajectory containing:

`state -> evidence -> hypothesis -> defense mode -> chosen cut point -> simulated outcome -> reassessment -> corrected action`

The corrected trajectory can feed the future Adversarial Reasoning Corpus and Tool-Trajectory Corpus. Range results therefore become a measurable bridge between Sentinel's Cyber World Model and its defensive behavior training.

## CLI

Run one or more scenarios with:

`sentinel-cyber-range --scenario <scenario.json> [--scenario <scenario2.json>]`

An optional gate policy may be supplied with `--policy`. The command exits non-zero when the suite fails its gates.
