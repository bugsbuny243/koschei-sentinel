# Koschei Sentinel Cyber SFT — Kaggle Micro-First Progression

The free-GPU path is intentionally staged so that a large model is never the first proof of the training stack.

## Stage 1 — Qwen3.5-0.8B micro-smoke

Run:

`notebooks/koschei_sentinel_cyber_sft_08b_micro_kaggle.ipynb`

Pinned model:

- `Qwen/Qwen3.5-0.8B-Base`
- revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`

The micro run uses the same Defense Reflex v3 corpus, text-only `Qwen3_5ForCausalLM` executor, QLoRA path, receipt, attestation, and portable export verifier as the 9B run.

A valid micro run must produce:

- `global_step > 0`
- valid adapter digest
- valid training receipt
- text-only runtime proof
- valid run attestation
- valid portable export verification
- `selected_profile=micro`
- `promotion_eligible=false`

The portable archive is:

`/kaggle/working/koschei-sentinel-qwen35-08b-micro.zip`

## Stage 2 — scale gate

The micro launcher automatically runs:

`sentinel-cyber-sft-scale-gate`

The gate consumes the fully verified micro export and the pinned 9B smoke config. It emits `scale-gate.json` with one of three dispositions:

- `PROCEED_9B`
- `PROCEED_9B_CAUTION`
- `BLOCK_9B`

The gate blocks 9B when the micro export is invalid, when the micro run did not perform a real optimizer step, when the micro evidence does not bind the pinned Qwen3.5-0.8B-Base checkpoint, when the target is not the pinned Qwen3.5-9B-Base config, or when the observed GPU memory is below the 9B config minimum.

Missing Qwen3.5 DeltaNet acceleration packages are recorded as a caution rather than a false success or unconditional block. The model can fall back to PyTorch kernels, but the fallback can be slower and more memory intensive.

## Stage 3 — Qwen3.5-9B smoke

Only run:

`notebooks/koschei_sentinel_cyber_sft_9b_kaggle.ipynb`

when `scale-gate.json` reports:

`allowed_to_attempt_9b=true`

The 9B launcher still performs its own model-access preflight, CUDA/runtime readiness, tokenizer-length check, real QLoRA execution, resumable checkpoints, artifact verification, attestation, and offline portable export verification. The micro gate does not bypass any 9B check.

## Canonical portable export layout

Both micro and 9B archives use the same model-independent names:

- `training-config.json`
- `training-plan.json`
- `model-preflight.json`
- `verification.json`
- `run-attestation.json`
- `export-verification.json`
- `selected-profile.txt`
- `repository-commit.txt`
- `corpus-examples.jsonl`
- `corpus-manifest.json`
- `run/adapter-manifest.json`
- `run/training-receipt.json`
- `run/model-runtime.json`
- `run/resume-runtime.json`

The purpose of the micro stage is execution proof, not model quality. Neither the 0.8B micro adapter nor the 9B synthetic-policy smoke adapter is promotion-eligible.
