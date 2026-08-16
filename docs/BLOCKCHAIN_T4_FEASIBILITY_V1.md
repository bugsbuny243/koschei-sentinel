# T4 Feasibility Candidate v1

The first three serious Stage 2 candidates exposed an operational gap: the two native high-capability candidates require far more single-device 4-bit weight memory than a T4-class GPU, while the smaller DeepSeek lane remains behind custom-code and manual-license review.

To keep the cheap experimentation lane usable without weakening the trust boundary, the registry adds a fourth candidate:

- candidate: `qwen2.5-coder-7b-base`
- model: `Qwen/Qwen2.5-Coder-7B`
- exact revision: `0396a76181e127dfc13e5c5ec48a8cee09938b02`
- training stage: base pretraining
- declared parameters: 7.61B total / 7.61B active
- declared context capability: 131,072 tokens
- license: Apache-2.0
- integration: native Transformers expected
- `trust_remote_code_required: false`
- training remains unauthorized until the actual hardware/runtime preflight passes.

This candidate is not declared the future Sentinel winner. It exists to give the Stage 2 data and training pipeline a realistic low-cost feasibility target that can be compared later against the larger efficiency and power lanes using the same held-out security tournament.

The repository runtime-preflight policy estimates the 7.61B model's 4-bit weight footprint before any download. A T4-class inventory is expected to admit this candidate to the dynamic probe while still rejecting the 80B-total and 106B-total candidates on single-device capacity.

No model quality claim is made by this admission. The candidate must still pass the actual quantized LoRA forward/backward probe, Stage 2 training, the 500+ case blockchain-security gate, and the security-first tournament.
