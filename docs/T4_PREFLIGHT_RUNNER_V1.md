# T4 Runtime Preflight Runner v1

This runner executes the complete fail-closed runtime preflight for a pinned base-model candidate without starting training.

Default candidate: `qwen2.5-coder-7b-base`.

The runner persists only aggregate hardware, planning, probe and audit artifacts. It does not persist raw model outputs and it never grants training or production authority.

Example:

```bash
python scripts/run_blockchain_t4_preflight.py \
  --output-dir build/preflight/qwen2.5-coder-7b-base
```

A successful run requires:

- measured CUDA hardware,
- static disk/RAM/VRAM fit,
- exact pinned revision,
- `trust_remote_code=False`,
- tokenizer/config/model load,
- 4-bit quantized model load,
- LoRA attachment,
- forward loss,
- backward pass,
- a passing final preflight audit.

Passing this runner means only that the exact pinned candidate is technically loadable/trainable under the current runtime policy. It does not authorize continued pretraining, promotion or production use.
