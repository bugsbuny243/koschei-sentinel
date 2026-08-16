# Blockchain Base Candidates v1

Status: preflight admission only  
Model download authority: none  
Training authority: none  
Production authority: none

## Why this layer exists

The Stage 2 trainer and the security-first tournament should not silently chase whatever a model registry calls `main` on the day a GPU job starts. Base-model selection is therefore a separate, immutable admission step.

A candidate here is **not approved for training**. It is only admitted for an offline compatibility, license and hardware preflight. Actual training still requires a later preflight receipt and an explicit immutable training plan.

## Initial three lanes

The first registry deliberately spans three different model shapes rather than pretending one parameter count answers every deployment question.

### FEASIBILITY — DeepSeek-Coder-V2-Lite-Base

- model: `deepseek-ai/DeepSeek-Coder-V2-Lite-Base`
- exact revision: `5a3cf151e6eb71197e34b87d42546c97b2adb139`
- official model-card declaration: 16B total / 2.4B active, 128K context
- model license: `deepseek-license`; the official card declares commercial use support, but Koschei treats this custom license as requiring manual review
- official Transformers usage requires `trust_remote_code=True`

Koschei's current blockchain trainer is fail-closed around `trust_remote_code=False`, so this candidate is intentionally blocked behind custom-code review/runtime integration. It must not be enabled merely because it is the smallest candidate.

### EFFICIENCY — Qwen3-Coder-Next-Base

- model: `Qwen/Qwen3-Coder-Next-Base`
- exact revision: `1b6df59d5f75ab51edb9ad8cb3ea69c5d0aedd57`
- official model-card declaration: 80B total / 3B active, native 262,144-token context
- license: Apache-2.0
- official Transformers usage does not require remote model code

This is the sparse efficiency lane, not an automatic winner. It still needs a local tokenizer/config/load/LoRA preflight and a real hardware plan before any training authorization review.

### POWER — GLM-4.5-Air-Base

- model: `zai-org/GLM-4.5-Air-Base`
- exact revision: `b52435f955c9dc2c38e75319e3d2f0bf511e15ff`
- official model-card declaration: 106B total / 12B active
- license: MIT; the official model card states commercial use and secondary development are permitted
- official Transformers usage does not require remote model code

The registry intentionally leaves context length unset rather than copying an unverified number into an immutable training contract.

## Fail-closed admission

Every candidate must use a 40-character immutable revision pin. `main`, moving tags and unpinned model IDs are not training lineage.

The registry also distinguishes:

- open-license policy admission from a custom license requiring manual review;
- native Transformers integration from remote custom code requiring review;
- model selection from runtime preflight;
- runtime preflight from hardware approval;
- hardware approval from actual training authorization.

All initial candidates have `training_authorized: false`. The registry audit may report `ready_for_preflight: true` while correctly reporting zero candidates ready for training authorization review.

## Command

```bash
sentinel-blockchain-base-candidates \
  --registry configs/models/blockchain-base-candidates.v1.json \
  --policy configs/models/blockchain-base-candidate-policy.v1.json \
  --output build/model-preflight/base-candidates.audit.json
```

This command performs no network request, model download, GPU allocation or training.

## Next gate

The next artifact must be a preflight receipt for each candidate that binds the exact model revision, downloaded config/tokenizer/model-code facts, runtime/library versions, quantization/LoRA compatibility, measured memory requirements and approved hardware class. Only a passing receipt may remove `runtime_preflight_not_run` and `hardware_plan_not_approved`.
