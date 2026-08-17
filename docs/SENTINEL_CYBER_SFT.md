# Koschei Sentinel Cyber SFT

This document defines the first real model-weight training path for the Cyber World Model / active-defense project.

## What this is

`sentinel-cyber-sft` is separate from the legacy `sentinel-train` path.

The legacy trainer consumes `SecurityCase` records and trains the older `sentinel.opinion.v1` behavior format. It must not be presented as training the Cyber State Graph, Defense Reflex or Causal Defense architecture.

The Cyber SFT path trains a pinned Qwen3.5 foundation checkpoint on evidence-grounded Sentinel supervision.

## Defense Reflex v3

Defense Reflex v3 is the current SFT input contract.

A training prompt contains only information that can exist at decision time:

- protected critical entity IDs,
- Cyber State Graph snapshots,
- evidence IDs already present in those graphs, and
- the task to derive a defensive plan.

It does not expose evaluator truth, failure labels, reviewer identity, review method, post-action success, or future outcome-evidence IDs to the model.

The supervised answer contains:

- an evidence-grounded interpretation,
- Guard / Combat / Siege mode for each reviewed step,
- defensive action,
- protected target,
- rationale,
- supporting evidence IDs that already exist in the input graph, and
- `outcome_verification_required = true`.

Future outcome evidence IDs remain review/audit metadata and are never prediction targets.

## Review classes

Defense Reflex v3 distinguishes review authority from training authorization.

`HUMAN` lessons may become promotion-eligible when all other review and evaluation requirements are satisfied.

`SYNTHETIC_POLICY` lessons may be training-authorized for deterministic smoke/curriculum testing, but are structurally forbidden from being promotion-eligible.

This prevents a synthetic seed run from being mistaken for production-grade Sentinel training.

## Seed curriculum

`sentinel-cyber-seed-curriculum` builds 32 deterministic, synthetic-policy-reviewed examples across eight balanced families:

1. credential-to-signer containment,
2. CI/CD supply-chain containment,
3. pending hostile transaction hold,
4. ambiguous single-source evidence that stays in Guard,
5. benign authentication that stays in Guard,
6. attacker reroute to the same protected signer,
7. endpoint execution containment, and
8. cloud workload / command-and-control containment.

The generated Defense Reflex v3 manifest is always `promotion_eligible = false`.

## First executable model

The first executable Cyber SFT smoke profile is:

- model: `Qwen/Qwen3.5-9B-Base`,
- exact model revision: `68c46c4b3498877f3ef123c856ecfde50c39f404`,
- execution profile: `DENSE_SINGLE_GPU_QLORA`,
- 4-bit NF4 QLoRA,
- LoRA only on the language backbone,
- vision modules excluded from LoRA targets,
- Qwen3.5 full-attention and Gated DeltaNet projection families included,
- no silent sequence truncation,
- prompt tokens masked from the training loss, and
- post-training adapter manifest bound to exact corpus/model hashes.

The current config is `configs/training/cyber-sft.qwen3.5-9b.smoke.json`.

## Qwen3.5 35B-A3B

`Qwen/Qwen3.5-35B-A3B-Base` remains the development specialization target, but its config uses `MOE_DISTRIBUTED_REQUIRED`.

The current single-GPU BitsAndBytes trainer intentionally refuses to execute that profile. A separate distributed / MoE-aware executor must be implemented and validated before the 35B Base plan can run.

This is fail-closed: a 35B plan may be inspected, but it cannot silently fall back to an unsafe or incomplete training recipe.

## Runtime readiness

Use:

```bash
sentinel-cyber-training-readiness \
  --config configs/training/cyber-sft.qwen3.5-9b.smoke.json \
  --check-runtime
```

Readiness verifies the corpus manifest and SHA-256, deterministic split plan, output collision, supported execution profile, required Python packages, CUDA visibility, current single-GPU memory and requested numeric precision.

Static planning without a runtime check never returns `ready_to_execute = true`.

## First real smoke run

The repository contains an end-to-end runner:

```bash
bash scripts/run_cyber_sft_qwen35_9b_smoke.sh
```

The script:

1. installs the training extras,
2. creates the seed curriculum if Defense Reflex v3 is absent,
3. runs the runtime/CUDA readiness gate,
4. writes the immutable SFT plan, and
5. starts the pinned QLoRA run.

A successful run produces an adapter directory plus `adapter-manifest.json` under the configured run output. The manifest records whether the source corpus was promotion-eligible.

For the synthetic seed curriculum this value must remain false.

## What a successful smoke run proves

A successful 9B seed run proves that the real training path can load the pinned foundation model, render leak-free Sentinel supervision, update LoRA weights, save the adapter and bind the artifact to exact corpus/model provenance.

It does not prove that the trained model is a production Sentinel.

Production promotion still requires human-reviewed/promotion-eligible training data plus the single-incident, multi-incident/world-line and defense-load evaluation gates.

## Next training stages

After the dense smoke recipe is validated:

1. replace synthetic-policy lessons with a large human-reviewed Defense Reflex v3 corpus,
2. train the Defense Reflex stage,
3. build a leak-free Causal Defense supervision revision,
4. train temporal/adversarial reasoning,
5. run all Cyber Range promotion gates,
6. build the distributed MoE executor,
7. repeat the validated recipe on the 35B-A3B Base development tier, and
8. use the larger distributed expert/teacher tier only after the lower tiers are reproducible.
