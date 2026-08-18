# Koschei Sentinel Cyber SFT — Kaggle GPU Run

This is the Kaggle execution path for the first real Qwen3.5-9B Cyber SFT smoke adapter when Google Colab GPU quota is unavailable.

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

1. GPU preflight via `nvidia-smi` and saves the output.
2. Training dependency installation.
3. Defense Reflex v3 seed generation when missing.
4. Runtime/CUDA readiness check.
5. Tokenizer-only context-length preflight before model weights are loaded.
6. Real QLoRA execution on the pinned Qwen3.5-9B-Base revision.
7. CUDA-memory-only retry with the low-memory profile when required.
8. Adapter and training-receipt verification.
9. Export of the selected config, selected profile, run, plan, corpus manifest, repository commit, readiness reports, training logs, and verification report.
10. Creation of `/kaggle/working/koschei-sentinel-qwen35-9b-smoke.zip`.

## Success conditions

The run is accepted as a real smoke training only if all of the following are true:

- `training-receipt.json` exists.
- `global_step > 0`.
- Adapter digest recomputation matches the manifest.
- Training receipt self-digest is valid.
- Receipt model/corpus/adapter bindings match the run artifacts.
- `verification.json` reports `valid=true`.
- `adapter-manifest.json` reports `corpus_promotion_eligible=false`.
- `selected-profile.txt` records the profile that actually completed.

A successful smoke run proves the tokenizer, quantized model load, LoRA target resolution, optimizer, backward pass, checkpoint save, receipt generation, and artifact-verification path. It does **not** make the adapter production-ready.

## Kaggle output persistence

The final ZIP is written under `/kaggle/working`. Save a notebook version with outputs after the run or download the ZIP before ending the session. The Hugging Face cache is not included in the export archive.
