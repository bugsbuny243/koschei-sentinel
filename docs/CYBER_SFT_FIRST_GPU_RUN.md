# Koschei Sentinel — First Verified Cyber SFT GPU Run

> Historical 9B regression record. It is not an active training target or a prerequisite.
> The only active target is `Qwen/Qwen3.5-397B-A17B`; see
> `docs/CYBER_SFT_397B_MEGATRON.md`.

This document describes the first real model-weight update in the Cyber SFT line.

## Scope

The first run uses:

- base model: `Qwen/Qwen3.5-9B-Base`
- pinned revision: `68c46c4b3498877f3ef123c856ecfde50c39f404`
- stage: `DEFENSE_REFLEX`
- execution profile: `DENSE_SINGLE_GPU_QLORA`
- quantization: 4-bit NF4 with double quantization
- compute dtype: float16
- sequence budget: 2048
- LoRA rank: 16
- optimizer: `paged_adamw_8bit`
- curriculum: Defense Reflex v3 synthetic-policy smoke curriculum

This is a real QLoRA update, but the seed corpus is deliberately marked
`promotion_eligible=false`. A successful smoke run proves the training path, not production
model quality.

## Recommended Colab path

Open `notebooks/koschei_sentinel_cyber_sft_9b_colab.ipynb` in Colab, select a GPU runtime,
and run the notebook from top to bottom.

The notebook stores the repository and training outputs under:

`/content/drive/MyDrive/Koschei-Sentinel/runtime/koschei-sentinel`

The default Hugging Face cache is persistent under:

`/content/drive/MyDrive/Koschei-Sentinel/hf-cache`

Set `KOSCHEI_HF_HOME` before launching if a different cache location is desired.

## Fail-closed launch sequence

The launcher executes the following chain:

1. install `.[training]`
2. materialize the deterministic 32-scenario Defense Reflex v3 seed curriculum if absent
3. build and validate the immutable Cyber SFT plan
4. check training runtime packages and the current CUDA device
5. load only the pinned processor/tokenizer and verify every supervision fits the 2048-token budget
6. load the pinned Qwen3.5 9B checkpoint in 4-bit mode
7. resolve LoRA targets only inside the language-model backbone
8. perform the real optimizer steps
9. save the LoRA adapter and processor
10. emit `adapter-manifest.json`
11. emit `training-receipt.json`
12. run `sentinel-cyber-sft-verify` over the completed run

The shell entry point is:

```bash
bash scripts/run_cyber_sft_qwen35_9b_smoke.sh
```

The Colab wrapper is:

```bash
bash scripts/run_cyber_sft_qwen35_9b_colab.sh
```

## Readiness evidence

Before model weights are loaded, the readiness report must show all of the following:

- `static_plan_ready=true`
- `runtime_checked=true`
- `runtime_dependencies_ready=true`
- `tokenization_checked=true`
- `tokenization_ready=true`
- `cuda_available=true`
- `ready_to_execute=true`
- `use_class=SMOKE_ONLY`
- zero blockers

`max_observed_sequence_tokens` records the largest actual supervision after applying the pinned
Qwen chat template. If an example exceeds 2048 tokens, the run stops before the full model is
loaded.

## Training receipt

A successful run writes `training-receipt.json` next to the adapter manifest. The receipt binds:

- exact model revision
- corpus examples SHA-256
- corpus manifest SHA-256
- adapter SHA-256
- optimizer
- global optimizer step count
- train metrics
- eval metrics
- CUDA device and total memory
- peak CUDA allocated/reserved memory
- installed Torch, Transformers, PEFT, BitsAndBytes, Accelerate and Datasets versions

The receipt has its own deterministic `receipt_sha256`.

## Final artifact verification

Run:

```bash
sentinel-cyber-sft-verify \
  --run-dir build/cyber-training/runs/qwen35-9b-smoke-001
```

A completed smoke run is accepted only when the report returns:

- `valid=true`
- `adapter_digest_verified=true`
- `receipt_digest_verified=true`
- `receipt_bindings_verified=true`
- `global_step > 0`
- `smoke_only=true`

Changing an adapter file, changing a receipt field without recomputing its digest, breaking a
manifest/receipt binding, or producing zero optimizer steps makes verification fail.

## What this run does not prove

The smoke adapter is not a production Sentinel candidate. It cannot become promotion-eligible
merely because training loss decreases or artifact verification passes.

Production promotion still requires human-reviewed training material plus the versioned
single-incident, multi-incident/world-line, resource-load and authority/grounding safety gates.

The 35B-A3B MoE development tier remains `MOE_DISTRIBUTED_REQUIRED` and is intentionally blocked
from this single-GPU trainer.
