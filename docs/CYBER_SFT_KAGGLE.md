# Koschei Sentinel Cyber SFT — Kaggle GPU Run

This is the Kaggle execution path for the first real Qwen3.5-9B Cyber SFT smoke adapter when Google Colab GPU quota is unavailable.

## Runtime model path

Cyber SFT is text-only. The executor uses `AutoTokenizer` and `AutoModelForCausalLM` against the pinned `Qwen/Qwen3.5-9B-Base` checkpoint. Transformers maps the Qwen3.5 VLM-compatible config to `Qwen3_5ForCausalLM`; that class uses the Qwen3.5 text model and ignores `model.visual.*` checkpoint keys. The executor therefore refuses to train unless:

- the loaded class is exactly `Qwen3_5ForCausalLM`, and
- no module name contains `visual` or `vision`.

The completed run writes `model-runtime.json`, and `sentinel-cyber-sft-verify` requires that file to confirm `AutoModelForCausalLM`, `Qwen3_5ForCausalLM`, and `text_only=true` before a run is considered valid.

## Exact model-access gate

Before CUDA training starts, `sentinel-cyber-model-preflight` checks the configured Hugging Face model and exact 40-character revision. The gate requires:

- the pinned revision to resolve to the exact configured commit SHA,
- the repository to be public and ungated for anonymous Kaggle execution,
- safetensors weights to be present,
- `AutoConfig.model_type` to be `qwen3_5`, and
- `AutoModelForCausalLM` to resolve the config to `Qwen3_5ForCausalLM`.

The report is exported as `model-preflight.json`. Any mismatch stops the launcher before training.

## Platform assumptions

- Kaggle Notebook accelerator is set to GPU.
- Notebook Internet is enabled so the repository and pinned Hugging Face checkpoint can be fetched.
- The normal smoke config is `DENSE_SINGLE_GPU_QLORA`, 4-bit NF4, float16, 2048 max sequence length, paged AdamW 8-bit, gradient checkpointing, and LoRA rank 16.
- The low-memory fallback is 4-bit NF4, float16, 1024 max sequence length, LoRA rank 8, and excludes MLP LoRA targets to reduce training memory.
- The training corpus is Defense Reflex v3 synthetic-policy seed curriculum and is intentionally `promotion_eligible=false`.

## Notebook

Use:

`notebooks/koschei_sentinel_cyber_sft_9b_kaggle.ipynb`

The notebook clones or updates the repository, checks the GPU, runs the Kaggle launcher, verifies the completed run, and leaves the export archive in `/kaggle/working`.

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
3. Exact Hugging Face revision/access/CausalLM mapping preflight.
4. GPU preflight via `nvidia-smi` and saves the output.
5. Runtime/CUDA readiness check.
6. `AutoTokenizer` context-length preflight before model weights are loaded.
7. Text-only `AutoModelForCausalLM` loading and strict Qwen3.5 runtime validation.
8. Real QLoRA execution on the pinned Qwen3.5-9B-Base revision.
9. CUDA-memory-only retry with the low-memory profile when required.
10. Adapter, training receipt, and text-runtime verification.
11. Fail-closed run attestation that binds the selected config, repository commit, exact resolved model revision, verification report, model runtime, resume runtime, receipt, adapter digest, and corpus hashes.
12. Export of the selected config, selected profile, run, plan, corpus manifest, repository commit, model preflight, readiness reports, training logs, verification report, and run attestation.
13. Creation of `/kaggle/working/koschei-sentinel-qwen35-9b-smoke.zip`.

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
- raw model-preflight and verification SHA-256 values,
- model-runtime and resume-runtime SHA-256 values,
- training receipt SHA-256,
- adapter digest,
- corpus examples and manifest SHA-256 values,
- completed optimizer step count, and
- resume status/checkpoint.

The attestation contains its own deterministic `attestation_sha256`. The ZIP is not produced if attestation creation fails.

## Success conditions

The run is accepted as a real smoke training only if all of the following are true:

- `training-receipt.json` exists.
- `model-runtime.json` exists and verifies the text-only Qwen3.5 runtime.
- `resume-runtime.json` is bound to the same config/model/corpus lineage.
- `global_step > 0`.
- Adapter digest recomputation matches the manifest.
- Training receipt self-digest is valid.
- Receipt model/corpus/adapter bindings match the run artifacts.
- `verification.json` reports `valid=true` and `model_runtime_verified=true`.
- `run-attestation.json` is produced only after a fresh valid verification and exact model/config/runtime binding checks.
- `adapter-manifest.json` reports `corpus_promotion_eligible=false`.
- `selected-profile.txt` records the profile that actually completed.

A successful smoke run proves the tokenizer, text-only quantized model load, LoRA target resolution, optimizer, backward pass, resumable checkpoint path, checkpoint save, receipt generation, artifact-verification path, and cross-artifact run attestation path. It does **not** make the adapter production-ready.

## Kaggle output persistence

The final ZIP is written under `/kaggle/working`. Save a notebook version with outputs after the run or download the ZIP before ending the session. The Hugging Face cache is not included in the export archive.
