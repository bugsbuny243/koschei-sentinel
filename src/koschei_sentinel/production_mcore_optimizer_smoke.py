from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import Field

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_mcore_distributed_runtime import MCoreDistributedRuntime
from koschei_sentinel.production_mcore_smoke import run_mcore_forward_backward_smoke
from koschei_sentinel.production_megatron_model_spec import ProductionMegatronModelSpec


class MCoreOptimizerSmokeResult(StrictModel):
    schema_version: Literal["sentinel.mcore-optimizer-smoke.v1"] = (
        "sentinel.mcore-optimizer-smoke.v1"
    )
    status: Literal["optimizer_step_verified"] = "optimizer_step_verified"
    global_rank: int = Field(ge=0)
    world_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    clip_grad: float = Field(gt=0.0)
    grad_norm: float = Field(ge=0.0)
    parameter_name: str = Field(min_length=1)
    before_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    after_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    parameter_changed: Literal[True] = True
    forward_verified: Literal[True] = True
    backward_verified: Literal[True] = True
    optimizer_step_verified: Literal[True] = True
    execution_authorized: Literal[False] = False


def _tensor_digest(tensor: Any) -> str:
    # Smoke-only digest of a deterministic sample, avoiding host copies of full 397B shards.
    detached = tensor.detach().reshape(-1)
    if detached.numel() == 0:
        raise RuntimeError("cannot digest an empty parameter")
    sample = detached[: min(4096, detached.numel())].float().cpu().contiguous()
    return hashlib.sha256(sample.numpy().tobytes()).hexdigest()


def _first_trainable_parameter(model: Any) -> tuple[str, Any]:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.numel() > 0:
            return name, parameter
    raise RuntimeError("model has no trainable parameters on this rank")


def build_mcore_distributed_optimizer(
    runtime: MCoreDistributedRuntime,
    *,
    learning_rate: float = 1.0e-5,
    weight_decay: float = 0.1,
    clip_grad: float = 1.0,
) -> Any:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    if clip_grad <= 0:
        raise ValueError("clip_grad must be positive")

    try:
        import torch
        from megatron.core.optimizer import get_megatron_optimizer
        from megatron.core.optimizer.optimizer_config import OptimizerConfig
    except ImportError as exc:  # pragma: no cover - production dependency path
        raise RuntimeError("Megatron-Core optimizer runtime is required") from exc

    config = OptimizerConfig(
        optimizer="adam",
        lr=learning_rate,
        min_lr=learning_rate,
        weight_decay=weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1.0e-8,
        decoupled_weight_decay=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=True,
        clip_grad=clip_grad,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
    )
    return get_megatron_optimizer(
        config=config,
        model_chunks=[runtime.model],
        use_gloo_process_groups=True,
    )


def run_mcore_optimizer_smoke(
    runtime: MCoreDistributedRuntime,
    spec: ProductionMegatronModelSpec,
    *,
    seq_length: int = 64,
    global_seed: int = 39735,
    learning_rate: float = 1.0e-5,
    weight_decay: float = 0.1,
    clip_grad: float = 1.0,
) -> MCoreOptimizerSmokeResult:
    """Run one real MCore BF16 distributed-AdamW update after the FWD/BWD smoke.

    The function requires the full torchrun/MCore process topology. It verifies that a
    trainable parameter changes after optimizer.step(). It does not authorize a training
    campaign or claim convergence.
    """
    model = runtime.model
    optimizer = build_mcore_distributed_optimizer(
        runtime,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        clip_grad=clip_grad,
    )
    optimizer.zero_grad(set_to_none=True)

    smoke = run_mcore_forward_backward_smoke(
        runtime,
        spec,
        seq_length=seq_length,
        global_seed=global_seed,
    )
    if not smoke.forward_verified or not smoke.backward_verified:
        raise RuntimeError("forward/backward smoke did not verify before optimizer step")

    parameter_name, parameter = _first_trainable_parameter(model)
    before_digest = _tensor_digest(parameter)

    step_result = optimizer.step()
    if isinstance(step_result, tuple):
        update_successful = bool(step_result[0])
        grad_norm_value = step_result[1] if len(step_result) > 1 else 0.0
    else:
        update_successful = bool(step_result) if step_result is not None else True
        grad_norm_value = 0.0
    if not update_successful:
        raise RuntimeError("Megatron optimizer reported an unsuccessful step")

    after_digest = _tensor_digest(parameter)
    if before_digest == after_digest:
        raise RuntimeError(
            f"optimizer step completed but sampled parameter did not change: {parameter_name}"
        )

    try:
        grad_norm = float(grad_norm_value.item())
    except AttributeError:
        grad_norm = float(grad_norm_value or 0.0)

    return MCoreOptimizerSmokeResult(
        global_rank=runtime.global_rank,
        world_size=runtime.world_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        clip_grad=clip_grad,
        grad_norm=max(0.0, grad_norm),
        parameter_name=parameter_name,
        before_digest=before_digest,
        after_digest=after_digest,
    )
