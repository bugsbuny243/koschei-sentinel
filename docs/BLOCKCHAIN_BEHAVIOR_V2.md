# Koschei Sentinel Blockchain Behavior V2

## Why V2 exists

Run `koschei-sentinel-blockchain-run-001` proved that a short QLoRA adaptation improved held-out blockchain language modeling while not yet improving overall security-analysis behavior.

Measured evidence:

- Base held-out loss: `1.260200`
- V1 adapter held-out loss: `1.188669`
- Relative held-out loss change: `-5.676%`
- Base behavior mean: `0.7000`
- V1 adapter behavior mean: `0.6833`
- Behavior record: `2 wins / 6 ties / 2 losses`

The important conclusion is narrow: V1 learned the blockchain/security distribution, but that gain did not yet translate into a stronger evidence-grounded security analyst.

## V2 objective

V2 is not a longer replay of V1. Its primary objective is to teach the model how to transform bounded facts into calibrated security analysis without erasing V1's domain-modeling gain.

The curriculum emphasizes:

1. fact versus inference separation;
2. abstention when evidence is missing;
3. explicit uncertainty and confidence boundaries;
4. safe, reversible containment actions;
5. false-positive and no-overclaim control;
6. signing payload integrity and independent reconstruction;
7. privileged-access and key compromise triage;
8. bridge/cross-chain domain separation;
9. RPC/state disagreement handling;
10. supply-chain and build provenance;
11. smart-contract and capability authorization controls.

## Data construction rules

The existing 1,099-document sealed corpus remains a knowledge source and retention replay pool, not the sole behavior-training source.

Behavior examples must be instruction-style cases with explicit supported findings, unsupported conclusions, missing evidence, uncertainty, and defensive actions. Training examples must not copy benchmark prompts or their reference answers. Variants must differ materially in facts, chain, actor role, evidence completeness, and safe next action.

Group-level train/eval splitting is mandatory. Duplicate and near-duplicate cases must not cross the split. Evaluation cases remain sealed and are never converted into training examples.

## Target scale

Initial V2 target:

- 2,400 new behavior examples;
- minimum acceptable build: 1,200 examples;
- approximately 400 retention-replay examples sampled from the technical corpus;
- 2,048-token sequence length;
- 1.5 epochs;
- learning rate `5e-5`;
- QLoRA rank 16 / alpha 32 / dropout 0.05.

This is deliberately a smaller learning rate than V1. V2 should adjust behavior while protecting the domain knowledge already gained.

## Promotion gates

V2 is not promoted merely because training loss falls.

It must satisfy all of the following:

- held-out blockchain loss no worse than `1.188669`;
- behavior mean at least `0.7500`, target `0.8000`;
- at most one behavior-case loss versus base;
- `clean-no-overclaim` score at least `0.75`;
- `move-capability-leak` score restored to `1.0`;
- Sentinel authority, grounding, abstention, and privacy hard gates all remain `1.0`;
- no automatic promotion.

## Stop conditions

A checkpoint is rejected if it improves domain loss while worsening clean/no-overclaim behavior, increases unsupported compromise claims, weakens abstention, or violates deterministic authority boundaries.

The model may observe, classify, explain, recommend restriction, request quarantine, or propose minimum-authority repair. It must never grant or widen authority, mutate evidence, override deterministic verdicts, or promote/deploy itself.
