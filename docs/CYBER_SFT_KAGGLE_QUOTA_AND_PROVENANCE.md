# Koschei Sentinel Cyber SFT — Kaggle Quota and Provenance Controls

> Historical regression workflow. The active training target is now only
> `Qwen/Qwen3.5-397B-A17B`; see `docs/CYBER_SFT_397B_MEGATRON.md`.

This document defines the fail-closed free-GPU workflow for the first real Sentinel Cyber SFT runs.

## 1. Spend zero GPU quota first

Run:

`notebooks/koschei_sentinel_cyber_sft_cpu_preflight_kaggle.ipynb`

with Kaggle accelerator set to **None** and Internet enabled.

The CPU preflight does not load model weights and does not train. It verifies:

- final `transformers>=5.12,<6` runtime,
- exact Qwen3.5-0.8B and Qwen3.5-9B model revisions,
- public/ungated Hugging Face access,
- `qwen3_5 -> Qwen3_5ForCausalLM` mapping,
- Defense Reflex v3 static training plans,
- Qwen3.5 tokenizer/chat-template compatibility, and
- sequence-length fit for micro, normal 9B, and low-memory 9B configs.

Only continue when `cpu-preflight-summary.json` reports:

`ready_for_gpu_micro_attempt=true`

## 2. Prove the real training engine on 0.8B

Run the preferred micro-first GPU notebook:

`notebooks/koschei_sentinel_cyber_sft_micro_first_kaggle.ipynb`

The micro run uses the pinned Qwen3.5-0.8B-Base checkpoint and performs real QLoRA optimizer steps. It remains synthetic-policy smoke data and is not promotion-eligible.

## 3. Seal source identity before the first optimizer step

Before training, the launcher creates:

`training-plan.json` / repository build plan

and a deterministic pre-training source binding containing:

- exact repository commit,
- canonical training-config SHA-256,
- training-plan SHA-256,
- base model and exact revision,
- corpus examples SHA-256, and
- corpus manifest SHA-256.

The exported canonical file is:

`training-source.json`

The binding is created before training. This means a later adapter can be attributed to the exact code/config/plan/corpus identity that authorized the optimizer run rather than to whatever repository state exists when the ZIP is eventually packaged.

## 4. Never silently retrain a completed run

A valid final run directory is reusable only when `sentinel-cyber-sft-reuse-check` proves all of the following:

- adapter/receipt/runtime verification is valid,
- the original training-source binding is valid,
- the repository commit is unchanged,
- training config and plan are unchanged,
- base model/revision are unchanged, and
- corpus examples/manifest identity is unchanged.

If these conditions pass, the launcher skips GPU training and continues verification, attestation, and export.

If a completed run exists but provenance is missing, invalid, or stale, the launcher stops. It does **not** delete the run and does **not** automatically retrain over ambiguity.

This specifically protects Kaggle quota when training succeeded but a later packaging or attestation step failed.

## 5. Interrupted training is separate from completed-run reuse

During an incomplete run, checkpoints are retained under the deterministic resume area. Resume is bound to model/revision, corpus hashes, run ID, and canonical training-config SHA-256.

The pre-training source binding adds the repository-commit constraint before execution, so a run interrupted under one code revision cannot be silently resumed after the repository changes.

## 6. Runtime integrity before scale-up

The real GPU readiness gate also requires:

- NVIDIA compute capability compatible with the selected quantization path,
- enough VRAM for the selected profile,
- final Transformers 5.12+ rather than prerelease/nightly builds,
- configured Qwen3.5 dtype requested explicitly at load time, and
- pre-kbit floating parameter dtype evidence with no competing low-precision dtype leakage.

For the current Kaggle recipes:

- micro and 9B use 4-bit NF4 + float16,
- NF4 requires compute capability 6.0+,
- normal 9B requires at least 14 GiB visible VRAM,
- low-memory 9B requires at least 12 GiB.

Optional Qwen3.5 DeltaNet fast-kernel packages are not treated as mandatory on older GPU architectures. Missing acceleration is surfaced as a scale-up caution because fallback kernels can be slower and more memory intensive.

## 7. Scale only after micro evidence

The micro launcher evaluates normal 9B first and low-memory 9B second when needed. It writes:

- `scale-gate.json`
- `recommended-9b-profile.txt`

The 9B phase remains disabled by default in the preferred notebook. The user must explicitly opt into spending more Kaggle GPU quota after reviewing the micro evidence.

## 8. Portable evidence chain

The final portable bundle includes:

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

Run attestation hashes the pre-training source file. Portable export verification independently re-validates source semantics against the exported repository commit, config, plan, model identity, and corpus identity.

A successful smoke proves the execution pipeline and provenance chain. It does not by itself make the resulting adapter production-ready.
