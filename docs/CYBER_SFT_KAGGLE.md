# Koschei Sentinel Cyber SFT — Kaggle GPU Run

The preferred free-GPU path is now **micro-first**. Do not make Qwen3.5-9B the first proof of the training stack.

Start with:

`notebooks/koschei_sentinel_cyber_sft_micro_first_kaggle.ipynb`

The notebook first performs a real QLoRA update on pinned `Qwen/Qwen3.5-0.8B-Base`, verifies and attests the resulting adapter, evaluates the micro-to-9B scale gate, and leaves the 9B phase disabled by default. The 9B phase should be enabled only when `scale-gate.json` reports `allowed_to_attempt_9b=true`.

## Runtime model path

Cyber SFT is text-only. The executor uses `AutoTokenizer` and `AutoModelForCausalLM` and refuses to train unless the loaded model class is exactly `Qwen3_5ForCausalLM` and no module name contains `visual` or `vision`.

The current real training progression is:

- micro proof: `Qwen/Qwen3.5-0.8B-Base` at revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`
- 9B smoke: `Qwen/Qwen3.5-9B-Base` at revision `68c46c4b3498877f3ef123c856ecfde50c39f404`

Both use Defense Reflex v3 synthetic-policy supervision and remain `promotion_eligible=false`.

## Transformers and dtype integrity

Cyber SFT requires a final `transformers>=5.12,<6` release. Prerelease, RC, dev, and nightly builds are rejected by readiness even when their numeric prefix is 5.12 or newer.

The text trainer explicitly passes the configured dtype to `AutoModelForCausalLM.from_pretrained`. For the Kaggle micro and 9B profiles this is `float16`.

Before PEFT k-bit preparation, the trainer inspects the loaded floating parameters and writes the result to `model-runtime.json`:

- `requested_compute_dtype`
- `observed_floating_dtypes_before_kbit_prepare`

The requested dtype must actually be observed. A competing low-precision dtype is forbidden: a float16 run cannot contain bfloat16 leakage and a bfloat16 run cannot contain float16 leakage. This check is independently enforced by the run verifier, run attestation, and portable export verifier.

## Exact model-access gate

Before a profile trains, `sentinel-cyber-model-preflight` requires:

- exact 40-character model revision resolution,
- public and ungated model access,
- safetensors weights,
- `AutoConfig.model_type=qwen3_5`, and
- `AutoModelForCausalLM -> Qwen3_5ForCausalLM` mapping.

A profile is never trained against an unverified model pin.

## GPU readiness

The Kaggle readiness gate checks the actual current CUDA device rather than summing multiple GPUs. It records GPU name, VRAM, and compute capability.

Current quantization requirements are fail-closed:

- 4-bit NF4/FP4 requires NVIDIA compute capability 6.0 or newer.
- the 8-bit loading path requires compute capability 7.5 or newer.

The micro profile requires at least 6 GiB visible VRAM. The normal 9B profile requires 14 GiB; the low-memory 9B profile requires 12 GiB.

Qwen3.5 DeltaNet can use optional acceleration packages. If they are absent, the model can fall back to slower, more memory-intensive PyTorch paths. Their absence is therefore recorded as a scale-up caution rather than being hidden. P100-class execution must not depend on optional kernels that target newer SM architectures.

## Phase 1 — micro-smoke

Preferred notebook:

`notebooks/koschei_sentinel_cyber_sft_micro_first_kaggle.ipynb`

Standalone micro notebook:

`notebooks/koschei_sentinel_cyber_sft_08b_micro_kaggle.ipynb`

Standalone launcher:

`bash scripts/run_cyber_sft_qwen35_08b_micro_kaggle.sh .`

The micro profile uses 4-bit NF4, float16, 2048-token context, paged AdamW 8-bit, gradient checkpointing, LoRA rank 8, and the same LoRA target classes required by the normal 9B recipe.

A valid micro run must perform real optimizer steps and pass adapter verification, run attestation, and portable export verification. It then executes `sentinel-cyber-sft-scale-gate`.

## Micro-to-9B scale gate

`sentinel-cyber-sft-scale-gate` emits one of:

- `PROCEED_9B`
- `PROCEED_9B_CAUTION`
- `BLOCK_9B`

The gate validates the complete micro export and requires the micro run to bind the pinned 0.8B model, perform `global_step > 0`, run on sufficient VRAM for the selected 9B target, and prove recipe compatibility.

Recipe checks include:

- quantization bit parity,
- compute dtype parity,
- LoRA target-class coverage,
- training stage and execution-profile parity, and
- context-length coverage.

Missing required LoRA target classes or a different quantization/dtype recipe block scale-up. A shorter micro context or missing optional fast kernels produces caution rather than silent approval.

## Phase 2 — 9B smoke

Standalone notebook:

`notebooks/koschei_sentinel_cyber_sft_9b_kaggle.ipynb`

Standalone launcher:

`bash scripts/run_cyber_sft_qwen35_9b_kaggle.sh .`

The launcher itself requires a verified micro export before doing any 9B model work, so bypassing the notebook does not bypass the micro gate.

In `auto` mode the launcher evaluates the normal 9B scale gate first. If the micro evidence cannot satisfy normal-profile requirements, it separately evaluates the low-memory profile. If neither profile is allowed, 9B is not started.

If normal 9B passes its scale gate but later encounters a real CUDA allocation failure, automatic fallback is allowed only for CUDA-memory failure signatures. Non-CUDA failures are never hidden by a weaker profile.

## Resumable checkpoints

The text trainer saves a checkpoint every two optimizer steps and retains the latest two. Resume is bound to:

- run ID,
- exact base model and revision,
- corpus examples SHA-256,
- corpus manifest SHA-256, and
- canonical full training-config SHA-256.

A checkpoint from a different model, corpus, or config is rejected.

## Verification and attestation

A completed run must pass `sentinel-cyber-sft-verify`. The verifier recomputes the adapter digest, training receipt digest and bindings, validates text-only runtime identity and dtype evidence, and requires `global_step > 0`.

`sentinel-cyber-sft-attest` then performs a fresh verification and binds the verified run to the selected profile, repository commit, exact model revision, training config, execution plan, runtime evidence, resume evidence, receipt, adapter, and corpus hashes.

## Canonical portable export

Micro and 9B use the same model-independent export layout:

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

`sentinel-cyber-sft-export-verify --export-dir <path>` requires no GPU or network access. It recomputes hashes and independently checks semantic bindings, including config-to-runtime dtype integrity. Archive creation is blocked unless portable verification reports `valid=true`.

## Kaggle persistence

The preferred micro-first notebook keeps micro evidence and optional 9B execution in the same Kaggle session. The 9B phase is disabled by default to protect free GPU quota.

Micro archive:

`/kaggle/working/koschei-sentinel-qwen35-08b-micro.zip`

9B archive, when explicitly run:

`/kaggle/working/koschei-sentinel-qwen35-9b-smoke.zip`

Save a Kaggle notebook version with outputs or preserve/download the generated archive before the working session ends. Neither smoke adapter is production-ready merely because the training pipeline executed successfully.
