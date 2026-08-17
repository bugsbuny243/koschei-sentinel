# Koschei Sentinel Cyber Foundation v3

## Decision

Koschei Sentinel no longer treats the 7B adapter line as the primary model-growth path. The 7B line remains a reproducible baseline, evaluator target, edge candidate, and future distillation target.

The primary architecture is now a large open-weight foundation model specialized into a security-center model through domain continued pretraining, evidence-grounded supervised fine-tuning, adversarial preference training, and strict promotion gates.

## Primary target

- Primary expert/teacher target: `Qwen/Qwen3.5-397B-A17B`
- Role: highest-capacity open-weight Sentinel expert and teacher candidate
- License: Apache-2.0
- Architecture: sparse MoE, 397B total parameters, ~17B activated per token
- Native context: 262,144 tokens
- The hosted Alibaba `qwen3.5-397b-a17b` endpoint is inference-only for our purposes; managed fine-tuning is not assumed.
- Any weight training of the 397B target is a self-hosted/distributed infrastructure project, not a Kaggle/T4 task.

## Practical development target

- Development specialization candidate: `Qwen/Qwen3.5-35B-A3B-Base`
- Role: trainable proving ground for the same corpus, objective, eval, and promotion contracts before expensive 397B runs
- License: Apache-2.0
- Open-weight footprint is approximately 72 GB before quantization/runtime overhead.
- This model is not a replacement for the primary 397B expert; it is the infrastructure-validation tier.

## Why this replaces the old direction

The v1 7B experiment demonstrated that a small adapter can learn the domain distribution while failing to improve security behavior. Held-out blockchain loss improved from 1.260200 to 1.188669 (-5.676%), while the 10-case behavior score moved from 0.7000 base to 0.6833 adapter.

This proves two things:

1. domain knowledge acquisition and security judgment are separate objectives;
2. repeatedly extending a capacity-limited 7B adapter is not the main strategy for a security-center model.

## Specialization stack

The large Sentinel is trained in layers. No stage may silently replace deterministic policy authority.

### C0 — Security corpus continued pretraining

Purpose: replace a portion of general-domain probability mass with high-value security distribution while retaining useful reasoning/code competence.

Data families include:

- endpoint and privileged-access compromise
- signing path and transaction-integrity failures
- key and wallet infrastructure
- software supply chain and build provenance
- cloud, IAM, identity, secrets, CI/CD, container and orchestration security
- network, RPC, consensus and validator infrastructure
- Web3 contracts, bridges and cross-chain messaging
- exploit root-cause reports and incident timelines
- malware behavior and defensive reverse engineering
- vulnerability classes, patches and secure implementation patterns
- cryptography, protocol security and post-quantum migration

### C1 — Evidence-grounded security SFT

Every answer must separate:

- observed facts
- inference
- unsupported conclusions
- uncertainty / missing evidence
- reversible defensive actions

The model must never turn suspicion into confirmed compromise or claim universal safety from a clean observation.

### C2 — Adversarial security alignment

Use hard negatives and preference pairs covering:

- false attribution
- premature exploit confirmation
- unsafe remediation
- privilege widening
- prompt-injected evidence
- malicious or stale telemetry
- conflicting RPC/provider state
- compromised build/signing UI
- red-team attempts to override deterministic verdicts

### C3 — Tool and agent specialization

Train the model to operate as an analyst behind deterministic boundaries. It may inspect, explain, correlate, propose containment, and request evidence. It may not grant authority, bypass compiler/runtime policy, mutate evidence, self-promote, or deploy itself.

### C4 — Teacher / distillation tier

The large Sentinel expert may later generate audited rationales, labels, preference pairs and synthetic cases for smaller deployable models. Distillation is allowed only from examples that pass the same evidence and leakage checks.

## Data scale

The existing 1,099 technical documents and 2,400 behavior examples are seed assets, not a sufficient final corpus for a 397B-class specialization.

The v3 corpus program should grow toward at least tens of thousands of curated source documents and tens of thousands of evidence-grounded behavior examples before committing expensive large-model training. Quantity alone is not a target; provenance, deduplication, eval separation, freshness and technical depth are mandatory.

## Promotion gates

A candidate is never promoted from training loss alone. Required gates include:

- unseen domain modeling / retention
- evidence grounding
- abstention and uncertainty calibration
- false-positive control
- incident-response quality
- signing/payload integrity
- privileged access and key compromise reasoning
- supply-chain provenance
- Web3 / bridge / RPC / consensus reasoning
- privacy and secret-exfiltration resistance
- adversarial prompt resistance
- authority-boundary preservation

The existing deterministic Sentinel benchmark remains authoritative. Model outputs remain commentary and recommendations.

## Infrastructure policy

Kaggle/T4 is retained for dataset construction, small-model experiments, scoring logic and smoke tests. It is not considered acceptable infrastructure for 397B training.

Before a 397B weight-training run, the project must produce:

1. a sealed corpus manifest;
2. a reproducible distributed-training plan;
3. exact accelerator/memory/storage requirements;
4. cost and checkpoint-retention budget;
5. a recovery/resume plan;
6. base-vs-candidate evaluation jobs that run before promotion.

Until those exist, the 397B model can be used only as an external teacher/inference reference where permitted, while the 35B-A3B-Base tier validates the training recipe.
