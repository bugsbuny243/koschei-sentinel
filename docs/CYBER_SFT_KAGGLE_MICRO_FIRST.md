# Koschei Sentinel Cyber SFT — Kaggle Micro-First Progression

> Historical regression workflow. The active training target is now only
> `Qwen/Qwen3.5-397B-A17B`; see `docs/CYBER_SFT_397B_MEGATRON.md`.

The free-GPU path is intentionally staged so that a large model is never the first proof of the training stack.

## Stage 1 — Qwen3.5-0.8B micro-smoke

Run:

`notebooks/koschei_sentinel_cyber_sft_08b_micro_kaggle.ipynb`

Pinned model:

- `Qwen/Qwen3.5-0.8B-Base`
- revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`

The micro run uses the same Defense Reflex v3 corpus, text-only `Qwen3_5ForCausalLM` executor, QLoRA path, training-source provenance, receipt, attestation, and portable export verifier as the 9B run.

Before the first optimizer step, the launcher seals a deterministic `training-source` binding over:

- exact repository commit
- selected training config SHA-256
- execution plan SHA-256
- pinned base model and revision
- corpus examples SHA-256
- corpus manifest SHA-256

If a completed run already exists, the launcher does not retrain automatically. It first verifies the completed run and then requires the stored training-source binding to match the current repository/config/plan identity. A valid match permits completed-run reuse so a later packaging or attestation failure does not burn free GPU quota a second time. A mismatch blocks reuse and blocks automatic retraining.

A valid micro run must produce:

- `global_step > 0`
- valid adapter digest
- valid training receipt
- text-only runtime proof
- valid training-source provenance
- valid run attestation
- valid portable export verification
- `selected_profile=micro`
- `promotion_eligible=false`

The text-only runtime explicitly requests the configured low-precision dtype. Runtime enforcement is based on the floating parameter dtypes actually loaded into the model: the requested dtype must be observed and the competing low-precision dtype must not be present. Checkpoint/config dtype metadata is not used as a substitute for loaded-weight evidence.

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

The micro launcher evaluates the normal 9B profile first. If the normal profile is blocked only by its stronger runtime requirements, the low-memory 9B profile is evaluated separately. `recommended-9b-profile.txt` records `normal`, `lowmem`, or `blocked`.

Missing Qwen3.5 DeltaNet acceleration packages are recorded as a caution rather than a false success or unconditional block. The model can fall back to PyTorch kernels, but the fallback can be slower and more memory intensive.

## Stage 3 — Qwen3.5-9B smoke

Only run:

`notebooks/koschei_sentinel_cyber_sft_9b_kaggle.ipynb`

when the verified micro scale gate reports:

`allowed_to_attempt_9b=true`

The 9B launcher independently re-verifies the micro export before any 9B work. Normal and low-memory profiles have separate execution plans, separate training-source bindings, and separate run directories. The 9B launcher performs its own model-access preflight, CUDA/runtime readiness, tokenizer-length check, real QLoRA execution, resumable checkpoints, artifact verification, attestation, and offline portable export verification. The micro gate does not bypass any 9B check.

If a normal or low-memory 9B run already completed, the same fail-closed completed-run reuse policy applies. Valid same-source evidence skips GPU retraining; missing or mismatched provenance refuses silent reuse and refuses automatic overwrite/retraining.

## Canonical portable export layout

Both micro and 9B archives use the same model-independent names:

- `training-config.json`
- `training-plan.json`
- `training-source.json`
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

The portable verifier recomputes file hashes and semantic bindings for the config, plan, training source, model preflight, runtime evidence, receipt, adapter, corpus, profile, and repository commit. A portable archive is only valid when all of those bindings agree with the run attestation.

The purpose of the micro stage is execution proof, not model quality. Neither the 0.8B micro adapter nor the 9B synthetic-policy smoke adapter is promotion-eligible.
