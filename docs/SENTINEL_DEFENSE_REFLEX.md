# Koschei Sentinel Defense Reflex Learning

Defense Reflex is the reviewed behavior-learning plane for Koschei Sentinel. It is deliberately separate from Cyber Corpus v3.

Cyber Corpus v3 teaches security knowledge and world structure. Defense Reflex teaches evidence-grounded defensive behavior under attack: when to remain in Guard, when Combat or Siege is justified, which protected cut point should be used first, how to verify containment, and how to react when the attacker changes path.

## Fail-closed learning loop

The canonical flow is:

`Cyber Range -> failure report -> Defense Reflex candidate -> reviewed correction -> verified outcome -> explicit training authorization -> Defense Reflex Corpus`

A range failure never becomes training data automatically.

`DefenseReflexCandidate` starts with:

- `review_status = REVIEW_REQUIRED`, and
- `training_authorization = false`.

A reviewed correction must preserve the source range-report SHA-256 and candidate identity. Approval requires a named reviewer, review evidence, a corrected interpretation, an ordered defensive sequence, supporting evidence for each step, and verified post-action outcome evidence.

Rejected or unverified corrections cannot be training-authorized.

## Reviewed correction trajectory

An approved trajectory captures:

1. the range failure that exposed the weakness,
2. why the prior interpretation or action was wrong,
3. the correct Guard / Combat / Siege mode,
4. the correct protected target and defensive action order,
5. evidence supporting each action,
6. post-action evidence proving the simulated defensive result, and
7. immutable source and correction digests.

This creates a supervised target for defensive reasoning without treating model-generated assertions as ground truth.

## Defense Reflex Corpus

`sentinel-defense-reflex-build` consumes reviewed correction JSONL and emits:

- `examples.jsonl`, and
- `manifest.json`.

The exporter fails the entire release if any correction is rejected, unverified, not explicitly training-authorized, duplicated by candidate, or duplicated by correction digest. It does not silently skip unsafe rows.

The manifest binds the release to an exact examples SHA-256 and the exact correction digests that produced it.

## Temporal Cyber World Model episodes

A Cyber Range timeline can also be converted into an observational `CyberWorldModelEpisode`. An episode preserves the sequence of graph snapshots and explicitly records OBSERVED, INFERRED, PREDICTED and DISPROVED relation counts, attack stage, defense mode, containment state and temporal transitions such as attacker reroute, containment and recovery.

A raw world-model episode always has `training_authorization = false`. It cannot promote itself into training data.

Each episode is bound to the exact source range report through `source_report_sha256`.

## Causal Defense Corpus

The Causal Defense plane teaches why a defensive action is correct in the context of an evolving attack, not merely which action should be selected.

A causal example can be created only when:

- the World Model Episode and reviewed correction belong to the same scenario,
- both reference the exact same source range-report SHA-256,
- the correction is APPROVED,
- its defensive outcome is verified, and
- explicit training authorization is present.

The resulting example binds temporal state history, graph changes, attacker reroutes, corrected interpretation and the verified defensive sequence into one provenance-preserving training item.

## Cyber training bundle v2

A Sentinel cyber training run is not considered ready merely because model weights and a dataset exist. The training bundle must bind four independent inputs:

- a ready Cyber Corpus collection-batch seal for the knowledge plane,
- a ready Defense Reflex Corpus manifest for immediate defensive behavior,
- a ready Causal Defense Corpus manifest for temporal/adversarial reasoning, and
- an independent evaluation-holdout SHA-256.

The holdout digest must differ from every training-plane digest.

The canonical training sequence is:

`KNOWLEDGE_CONTINUED_PRETRAINING -> DEFENSE_REFLEX_SFT -> ADVERSARIAL_REASONING -> CYBER_RANGE_REGRESSION`

`ADVERSARIAL_REASONING` is backed by the reviewed Causal Defense Corpus. `CYBER_RANGE_REGRESSION` remains a promotion gate, not a training-data source by default. Failures return to the reviewed Defense Reflex loop.

The bundle itself is immutable by digest and records the exact foundation model reference/revision plus the exact hashes of all three training planes.

## Design principle

Sentinel should learn from its mistakes, but it must not decide by itself that its own mistake is the correct lesson.

The learning authority therefore remains outside the model:

`model behavior -> deterministic measurement -> reviewed correction -> explicit authorization -> training`
