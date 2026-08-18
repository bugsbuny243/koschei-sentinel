# Koschei Sentinel Cyber SFT — Kaggle GPU Run

This is the Kaggle execution path for the first real Qwen3.5-9B Cyber SFT smoke adapter when Google Colab GPU quota is unavailable.

## Runtime model path

Cyber SFT is text-only. The executor uses `AutoTokenizer` and `AutoModelForCausalLM` against the pinned `Qwen/Qwen3.5-9B-Base` checkpoint. The executor refuses to train unless the loaded class is exactly `Qwen3_5ForCausalLM` and no module name contains `visual` or `vision`.

The completed run writes `model-runtime.json`, and `sentinel-cyber-sft-verify` requires that file to confirm `AutoModelForCausalLM`, `Qwen3_5ForCausalLM`, and `text_only=true` before a run is considered valid.

## Exact model-access gate

Before each profile trains, `sentinel-cyber-model-preflight` checks that profile's configured Hugging Face model and exact 40-character revision. The gate requires:

- the pinned revision to resolve to the exact configured commit SHA,
- the repository to be public and ungated for anonymous Kaggle execution,
- safetensors weights to be present,
- `AutoConfig.model_type` to be `qwen3_5`, and
- `AutoModelForCausalLM` to resolve the config to `Qwen3_5ForCausalLM`.

Profile-specific reports are kept as `model-preflight-normal.json` and, when needed, `model-preflight-lowmem.json`. The report for the profile that actually completed is copied to `model-preflight.json`. A low-memory fallback is therefore never trained against an unverified model pin.

## Platform assumptions

- Kaggle Notebook accelerator is set to GPU.
- Notebook Internet is enabled so the repository and pinned Hugging Face checkpoint can be fetched.
- The normal smoke config is `DENSE_SINGLE_GPU_QLORA`, 4-bit NF4, float16, 2048 max sequence length, paged AdamW 8-bit, gradient checkpointing, and LoRA rank 16.
- The low-memory fallback is 4-bit NF4, float16, 1024 max sequence length, LoRA rank 8, and excludes MLP LoRA targets to reduce training memory.
- The training corpus is Defense Reflex v3 synthetic-policy seed curriculum and is intentionally `promotion_eligible=false`.

## Notebook

Use:

`notebooks/koschei_sentinel_cyber_sft_9b_kaggle.ipynb`

The notebook clones or updates the repository, checks the GPU, runs the Kaggle launcher, verifies the completed run, verifies the portable export, and leaves the final archive in `/kaggle/working`.

## Launcher

`bash scripts/run_cyber_sft_qwen35_9b_kaggle.sh .`

The default profile mode is `auto`. It first attempts the normal smoke profile. Automatic fallback is permitted **only** when the training log contains a CUDA-memory allocation failure. A non-memory failure stops the run instead of being hidden by a weaker profile.

To force a profile:

```bash
KOSCHEI_KAGGLE_PROFILE=normal bash scripts/run_cyber_sft_qwen35_9b_kaggle.sh .
KOSCHEI_KAGGLE_PROFILE=lowmem bash scripts/run_cyber_sft_qwen35_9b_kaggle.sh .
```

The launcher performs:

1. Text-only training dependency installation.
2. Defense Reflex v3 seed generation when missing.
3. GPU visibility preflight.
4. Exact Hugging Face revision/access/CausalLM mapping preflight for the profile that is about to train.
5. Runtime/CUDA readiness and tokenizer context-length checks.
6. Text-only `AutoModelForCausalLM` loading and strict Qwen3.5 runtime validation.
7. Real QLoRA execution on the pinned Qwen3.5-9B-Base revision.
8. CUDA-memory-only retry with a separately preflighted low-memory profile when required.
9. Adapter, training receipt, and text-runtime verification.
10. Fail-closed run attestation binding config, execution plan, repository commit, exact resolved model revision, verification report, model runtime, resume runtime, receipt, adapter digest, and corpus hashes.
11. Export of the selected config, selected profile, run, plan, corpus manifest, corpus examples, repository commit, selected model preflight, readiness reports, training logs, verification report, and run attestation.
12. Offline re-verification of the copied export with `sentinel-cyber-sft-export-verify`.
13. Creation of `/kaggle/working/koschei-sentinel-qwen35-9b-smoke.zip` only after export verification succeeds.

## Resumable checkpoints

The text-only trainer stores a deterministic resume area beside the final run directory and saves a checkpoint every two optimizer steps, retaining the latest two checkpoints.

Resume is fail-closed. `resume-binding.json` binds the checkpoint lineage to:

- `run_id`,
- exact base model and revision,
- corpus examples SHA-256,
- corpus manifest SHA-256, and
- SHA-256 of the complete Cyber SFT config.

If any of those values change, the old checkpoint is refused instead of silently resumed. A completed run writes `resume-runtime.json` recording whether the run resumed and from which checkpoint. After a successful final artifact commit, the temporary resume directory is removed. If execution is interrupted before completion, the bound checkpoints remain available for the next compatible invocation as long as the Kaggle working state itself is still available.

## Run attestation

After `sentinel-cyber-sft-verify` returns a valid report, `sentinel-cyber-sft-attest` performs a fresh verification and refuses to emit an attestation unless all bindings still match. `run-attestation.json` records and hashes:

- selected profile and repository commit,
- exact configured and resolved model revision,
- canonical training config SHA-256,
- raw execution-plan SHA-256,
- raw model-preflight and verification SHA-256 values,
- model-runtime and resume-runtime SHA-256 values,
- training receipt SHA-256,
- adapter digest,
- corpus examples and manifest SHA-256 values,
- completed optimizer step count, and
- resume status/checkpoint.

The attestation contains its own deterministic `attestation_sha256`.

## Portable export verification

The final export contains both `defense-reflex-v3.manifest.json` and `defense-reflex-v3.examples.jsonl`, so the corpus hashes can be recomputed without access to the original repository build directory.

`sentinel-cyber-sft-export-verify --export-dir <path>` requires no GPU or network access. It independently checks:

- the attestation self-hash,
- canonical selected-config hash,
- raw execution-plan hash,
- selected model-preflight hash and semantic model binding,
- verification-report hash plus a fresh verification of the copied `run/` directory,
- model-runtime and resume-runtime hashes and semantics,
- receipt and adapter bindings,
- corpus examples and manifest hashes,
- selected-profile binding, and
- repository-commit binding.

The launcher writes the result to `export-verification.json`. If it does not report `valid=true`, archive creation is stopped.

## Success conditions

The run is accepted as a real smoke training only if all of the following are true:

- `training-receipt.json` exists and reports `global_step > 0`.
- `model-runtime.json` verifies the text-only Qwen3.5 runtime.
- `resume-runtime.json` is bound to the same config/model/corpus lineage.
- Adapter digest recomputation matches the manifest.
- Training receipt self-digest is valid.
- Receipt model/corpus/adapter bindings match the run artifacts.
- `verification.json` reports `valid=true` and `model_runtime_verified=true`.
- `run-attestation.json` verifies the config/plan/model/runtime/receipt/corpus lineage.
- `export-verification.json` reports `valid=true` after checking the copied portable bundle.
- `adapter-manifest.json` reports `corpus_promotion_eligible=false`.
- `selected-profile.txt` records the profile that actually completed.

A successful smoke run proves the tokenizer, text-only quantized model load, LoRA target resolution, optimizer, backward pass, resumable checkpoint path, receipt generation, artifact verification, run attestation, and portable offline-verification path. It does **not** make the adapter production-ready.

## Kaggle output persistence

The final ZIP is written under `/kaggle/working`. Save a notebook version with outputs after the run or download the ZIP before ending the session. The Hugging Face cache is not included in the export archive.
