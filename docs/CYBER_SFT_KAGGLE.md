# Koschei Sentinel Cyber SFT — Kaggle GPU Run

This is the Kaggle execution path for the first real Qwen3.5-9B Cyber SFT smoke adapter when Google Colab GPU quota is unavailable.

## Runtime model path

Cyber SFT is text-only. The executor uses `AutoTokenizer` and `AutoModelForCausalLM` against the pinned `Qwen/Qwen3.5-9B-Base` checkpoint. Transformers maps the Qwen3.5 VLM-compatible config to `Qwen3_5ForCausalLM`; that class uses the Qwen3.5 text model and ignores `model.visual.*` checkpoint keys. The executor therefore refuses to train unless:

- the loaded class is exactly `Qwen3_5ForCausalLM`, and
- no module name contains `visual` or `vision`.

The completed run writes `model-runtime.json`, and `sentinel-cyber-sft-verify` requires that file to confirm `AutoModelForCausalLM`, `Qwen3_5ForCausalLM`, and `text_only=true` before a run is considered valid.

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
2. Text-only training dependency installation.
3. Defense Reflex v3 seed generation when missing.
4. Runtime/CUDA readiness check.
5. `AutoTokenizer` context-length preflight before model weights are loaded.
6. Text-only `AutoModelForCausalLM` loading and strict Qwen3.5 runtime validation.
7. Real QLoRA execution on the pinned Qwen3.5-9B-Base revision.
8. CUDA-memory-only retry with the low-memory profile when required.
9. Adapter, training receipt, and text-runtime verification.
10. Export of the selected config, selected profile, run, plan, corpus manifest, repository commit, readiness reports, training logs, and verification report.
11. Creation of `/kaggle/working/koschei-sentinel-qwen35-9b-smoke.zip`.

## Success conditions

The run is accepted as a real smoke training only if all of the following are true:

- `training-receipt.json` exists.
- `model-runtime.json` exists and verifies the text-only Qwen3.5 runtime.
- `global_step > 0`.
- Adapter digest recomputation matches the manifest.
- Training receipt self-digest is valid.
- Receipt model/corpus/adapter bindings match the run artifacts.
- `verification.json` reports `valid=true` and `model_runtime_verified=true`.
- `adapter-manifest.json` reports `corpus_promotion_eligible=false`.
- `selected-profile.txt` records the profile that actually completed.

A successful smoke run proves the tokenizer, text-only quantized model load, LoRA target resolution, optimizer, backward pass, checkpoint save, receipt generation, and artifact-verification path. It does **not** make the adapter production-ready.

## Kaggle output persistence

The final ZIP is written under `/kaggle/working`. Save a notebook version with outputs after the run or download the ZIP before ending the session. The Hugging Face cache is not included in the export archive.
