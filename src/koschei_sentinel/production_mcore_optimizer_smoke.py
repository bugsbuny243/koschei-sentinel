from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_mcore_checkpoint_roundtrip import verify_mcore_checkpoint_roundtrip
from koschei_sentinel.production_mcore_ddp import wrap_runtime_with_megatron_ddp
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
    checkpoint_roundtrip_verified: bool = False
    checkpoint_dir: str | None = None
    restored_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    forward_verified: Literal[True] = True
    backward_verified: Literal[True] = True
    optimizer_step_verified: Literal[True] = True
    execution_authorized: Literal[False] = False

    @model_validator(mode="after")
    def roundtrip_fields_are_coherent(self) -> "MCoreOptimizerSmokeResult":
        if self.before_digest == self.after_digest:
            raise ValueError("optimizer smoke requires a changed parameter digest")
        if self.checkpoint_roundtrip_verified:
            if not self.checkpoint_dir or not self.restored_digest:
                raise ValueError("verified checkpoint roundtrip requires path and restored digest")
            if self.restored_digest != self.after_digest:
                raise ValueError("restored digest must equal post-step digest")
        elif self.checkpoint_dir is not None or self.restored_digest is not None:
            raise ValueError("unverified checkpoint roundtrip cannot carry checkpoint evidence")
        return self


def _tensor_digest(tensor: Any) -> str:
    detached = tensor.detach().reshape(-1)
    if detached.numel() == 0:
        raise RuntimeError("cannot digest an empty parameter")
    sample = detached[: min(4096, detached.numel())].float().cpu().contiguous()
    return hashlib.sha256(sample.numpy().tobytes()).hexdigest()


def _gradient_for(parameter: Any) -> Any | None:
    main_grad = getattr(parameter, "main_grad", None)
    return main_grad if main_grad is not None else parameter.grad


def _first_parameter_with_gradient(model: Any) -> tuple[str, Any]:
    fallback: tuple[str, Any] | None = None
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.numel() == 0:
            continue
        if fallback is None:
            fallback = (name, parameter)
        gradient = _gradient_for(parameter)
        if gradient is not None:
            sample = gradient.detach().reshape(-1)[: min(4096, gradient.numel())]
            if sample.numel() and bool(sample.abs().max().item() > 0):
                return name, parameter
    if fallback is not None:
        raise RuntimeError("no trainable parameter with a nonzero gradient was found on this rank")
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
        config_overrides={},
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
    checkpoint_dir: str | Path | None = None,
) -> MCoreOptimizerSmokeResult:
    """Run one BF16 distributed-AdamW update and optional checkpoint roundtrip.

    The raw GPT graph is wrapped in Megatron DDP so contiguous FP32 main-grad/parameter
    buffers exist for reduce-scatter and distributed AdamW. This is a one-step systems
    smoke only and never authorizes a training campaign.
    """
    ddp_runtime = wrap_runtime_with_megatron_ddp(runtime)
    model = ddp_runtime.model
    optimizer = build_mcore_distributed_optimizer(
        ddp_runtime,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        clip_grad=clip_grad,
    )
    optimizer.zero_grad(set_to_none=True)
    model.zero_grad_buffer()

    smoke = run_mcore_forward_backward_smoke(
        ddp_runtime,
        spec,
        seq_length=seq_length,
        global_seed=global_seed,
    )
    if not smoke.forward_verified or not smoke.backward_verified:
        raise RuntimeError("forward/backward smoke did not verify before optimizer step")

    model.finish_grad_sync()
    parameter_name, parameter = _first_parameter_with_gradient(model)
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

    roundtrip_verified = False
    restored_digest: str | None = None
    checkpoint_path: str | None = None
    if checkpoint_dir is not None:
        roundtrip = verify_mcore_checkpoint_roundtrip(
            model,
            checkpoint_dir=checkpoint_dir,
            parameter_name=parameter_name,
        )
        roundtrip_verified = roundtrip.digest_restored
        restored_digest = roundtrip.restored_digest
        checkpoint_path = roundtrip.checkpoint_dir
        if restored_digest != after_digest:
            raise RuntimeError("checkpoint roundtrip restored digest differs from post-step parameter")

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
        checkpoint_roundtrip_verified=roundtrip_verified,
        checkpoint_dir=checkpoint_path,
        restored_digest=restored_digest,
    )
