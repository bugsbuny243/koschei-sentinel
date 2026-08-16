# Blockchain Runtime Preflight v1

This stage prevents large model downloads and GPU training from starting before a pinned base-model candidate has a credible hardware fit and a real quantized LoRA forward/backward probe.

## Boundary

Preflight is not training authorization and it is not a security-quality benchmark. Every artifact keeps `training_authorized: false`; a later explicit authorization stage must consume a passing audit.

The v1 executor is intentionally fail-closed:

- exact 40-character model revisions only,
- no `trust_remote_code=True`,
- only allowlisted-open-license candidates,
- only candidates marked for native Transformers integration,
- no raw generations or model outputs are persisted,
- one-device fit is required before a model download/probe is authorized,
- model output is never generated; the dynamic check is a short causal-LM loss forward/backward pass,
- the probe does not start continued pretraining.

Candidate registry blockers `runtime_preflight_not_run` and `hardware_plan_not_approved` are transitional and are resolved by this stage. License-review, remote-code-review and other substantive blockers remain fatal.

## Resource estimates

The offline plan estimates three lower-bound resource classes from the candidate's declared total parameter count:

1. checkpoint disk footprint, assuming the configured checkpoint bytes per parameter plus a safety margin,
2. quantized weight footprint,
3. conservative VRAM and host-RAM floors for the configured quantization mode.

These estimates are admission filters, not promises. Actual runtime compatibility is established only by the dynamic probe.

Using total rather than active MoE parameters is deliberate: inactive experts still occupy model-weight storage and normally must be represented in the loaded checkpoint.

## CLI

Inspect the actual machine first:

```bash
sentinel-blockchain-preflight inventory \
  --inventory-id colab-runtime \
  --output build/preflight/hardware.json
```

Build an offline plan. This command performs no model download:

```bash
sentinel-blockchain-preflight plan \
  --registry configs/models/blockchain-base-candidates.v1.json \
  --candidate-id qwen3-coder-next-base \
  --hardware build/preflight/hardware.json \
  --policy configs/models/blockchain-runtime-preflight-policy.v1.json \
  --output build/preflight/qwen3-plan.json
```

Only when `runtime_probe_authorized` is true may the explicit probe be run:

```bash
sentinel-blockchain-preflight execute \
  --registry configs/models/blockchain-base-candidates.v1.json \
  --candidate-id qwen3-coder-next-base \
  --hardware build/preflight/hardware.json \
  --policy configs/models/blockchain-runtime-preflight-policy.v1.json \
  --plan build/preflight/qwen3-plan.json \
  --output build/preflight/qwen3-receipt.json
```

Verify the receipt and produce the training-authorization-review audit:

```bash
sentinel-blockchain-preflight audit \
  --registry configs/models/blockchain-base-candidates.v1.json \
  --candidate-id qwen3-coder-next-base \
  --hardware build/preflight/hardware.json \
  --policy configs/models/blockchain-runtime-preflight-policy.v1.json \
  --plan build/preflight/qwen3-plan.json \
  --receipt build/preflight/qwen3-receipt.json \
  --output build/preflight/qwen3-audit.json
```

A passing audit means only that the exact pinned candidate fits the measured environment and survived tokenizer/config/quantized-load/LoRA/forward/backward checks without remote model code. It does not mean the candidate is blockchain-security capable or production-ready.

## Current shortlist implication

The repository policy is deliberately strict enough that a T4-class 16 GB GPU should reject the large 80B-total and 106B-total candidates before download based on single-device 4-bit footprint. The DeepSeek feasibility candidate has a smaller estimated footprint, but v1 still blocks it because its current registry record requires custom remote-code review and manual license review.

That outcome is useful: preflight is supposed to expose hardware and trust-boundary mismatches before money is spent.
