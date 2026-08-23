# Koschei Sentinel Cyber SFT

Koschei Sentinel Cyber SFT now has one active model target:

`Qwen/Qwen3.5-397B-A17B@8472618112abcbd45acbcdc58436aff4233c23f7`

There is no separate 35B development model. Historical 0.8B/9B single-GPU experiments remain in
the repository for regression and provenance only; they are not part of the active training plan.

## Supervision contract

Defense Reflex v3 prompts contain only information available at decision time:

- protected critical entity IDs;
- Cyber State Graph snapshots;
- evidence IDs already present in those graphs; and
- the request to derive a defensive plan.

They exclude evaluator truth, failure labels, reviewer identity, post-action success, and future
outcome-evidence IDs. The answer contains an evidence-grounded interpretation and a reviewed
defense sequence whose steps cite existing evidence and require post-action verification.

## Executor

`sentinel-cyber-megatron-sft` renders this supervision into ms-swift messages JSONL and builds a
shell-free `megatron sft` argument vector. The configuration enforces:

- exact 397B model ID and weight revision;
- `ms-swift==4.5.2`;
- distributed TP/PP/CP/EP topology;
- language-model-only LoRA;
- frozen vision encoder and aligner;
- positive MoE router auxiliary loss;
- optimizer and RNG checkpoint retention; and
- explicit, run-ID-bound paid-cluster approval.

See `docs/CYBER_SFT_397B_MEGATRON.md` for the complete operator sequence.

## Active configurations

The deterministic seed/smoke configuration is:

`configs/training/cyber-sft.qwen3.5-397b-a17b.megatron.json`

The split-safe human-reviewed Gold example is:

`configs/training/cyber-sft.qwen3.5-397b-a17b.gold.example.json`

Both train the same pinned model. The difference is the corpus, not the model tier.

## Promotion boundary

The synthetic seed curriculum is always `promotion_eligible=false`. It can prove data rendering,
distributed launch, optimizer updates, checkpoint recovery, and artifact bindings, but it cannot
produce a production candidate.

Production promotion requires human-reviewed TRAIN and VALIDATION releases plus the single-incident,
multi-incident/world-line, resource-load, grounding, privacy, adversarial, and authority gates.

