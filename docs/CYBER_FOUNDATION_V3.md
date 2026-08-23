# Koschei Sentinel Cyber Foundation v3

## Decision

Koschei Sentinel has exactly one active model-training target:

`Qwen/Qwen3.5-397B-A17B`

`3.5` is the Qwen generation. It does not mean a separate 35B model. The active plan has no
35B development tier, no 9B scale-up target, and no small-model prerequisite.

## Immutable model identity

- Model: `Qwen/Qwen3.5-397B-A17B`
- Weight revision: `8472618112abcbd45acbcdc58436aff4233c23f7`
- Architecture: sparse MoE, 397B total parameters and 17B activated per token
- Language stack: 60 layers, 512 experts, 10 routed experts plus one shared expert
- Native context: 262,144 tokens
- License: Apache-2.0

The revision is passed to the training backend and is included in the materialized dataset
manifest and static launch plan. A different model identifier or revision is rejected during
configuration validation.

## Training backend

The active executor is Megatron-SWIFT, pinned to `ms-swift==4.5.2`. The checked-in contract uses
a candidate 32-GPU topology:

- 4 nodes × 8 GPUs
- TP 8
- PP 4
- CP 1
- EP 8
- sequence parallelism enabled

This topology is a starting contract, not a claim that an untested provider cluster will fit the
run. GPU memory, NCCL, shared storage, checkpoint throughput, wall time, and cost must be measured
before a paid launch.

## Specialization stack

The same 397B model proceeds through the following stages. No stage introduces another model.

### C0 — Security-domain continued pretraining

Shift part of the model distribution toward high-value defensive-security material while retaining
reasoning and code competence. Sources require provenance, licensing, freshness, content hashes,
semantic deduplication, malicious-instruction quarantine, and benchmark-answer exclusion.

### C1 — Evidence-grounded SFT

Train the model to separate observations, inference, uncertainty, unsupported conclusions, and
reversible defensive actions. Defense Reflex v3 is the first executable supervision contract.

### C2 — Adversarial preference training

Use hard negatives and preference pairs for false attribution, unsafe remediation, privilege
widening, stale or malicious telemetry, compromised signing/build surfaces, and attempts to bypass
deterministic authority.

### C3 — Tool and agent specialization

The model may inspect, correlate, explain, propose containment, and request evidence. It may not
grant authority, mutate evidence, self-promote, deploy itself, or replace deterministic policy.

### C4 — Hard-gate evaluation

Training loss alone never authorizes promotion. Required gates include grounding, abstention,
false-positive control, incident-response quality, signing integrity, privileged access, supply
chain, Web3/RPC/consensus reasoning, privacy, prompt resistance, and authority preservation.

## Active files

- Architecture decision: `configs/training/cyber-foundation-v3.plan.json`
- Seed/smoke SFT contract: `configs/training/cyber-sft.qwen3.5-397b-a17b.megatron.json`
- Gold SFT example: `configs/training/cyber-sft.qwen3.5-397b-a17b.gold.example.json`
- Operator runbook: `docs/CYBER_SFT_397B_MEGATRON.md`

Older dense single-GPU configs and notebooks remain only as historical regression artifacts. They
are not active training targets and are not prerequisites for the 397B run.

## Launch boundary

The repository can materialize data and produce a complete command without spending GPU budget.
Execution remains fail-closed until all of the following are true:

1. the source corpus and rendered messages JSONL hashes verify;
2. the exact `ms-swift` version and `megatron` executable are present;
3. `NNODES`, `NPROC_PER_NODE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, and visible GPUs match;
4. optimizer and RNG checkpoint state will be saved for recovery;
5. an operator explicitly approves the paid run by setting the run-ID-bound launch variable.

