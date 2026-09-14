from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_mcore_ddp import wrap_runtime_with_megatron_ddp
from koschei_sentinel.production_mcore_distributed_runtime import MCoreDistributedRuntime
from koschei_sentinel.production_mcore_native_checkpoint import initialize_mcore_native_parameters
from koschei_sentinel.production_mcore_optimizer_smoke import build_mcore_distributed_optimizer
from koschei_sentinel.production_mcore_smoke import run_mcore_forward_backward_smoke
from koschei_sentinel.production_mcore_training_checkpoint import (
    MCoreTrainingState,
    load_mcore_training_checkpoint,
    save_mcore_training_checkpoint,
)
from koschei_sentinel.production_megatron_model_spec import ProductionMegatronModelSpec


class MCoreTrainingLoopConfig(StrictModel):
    schema_version: Literal["sentinel.mcore-training-loop-config.v1"] = (
        "sentinel.mcore-training-loop-config.v1"
    )
    max_steps: int = Field(gt=0)
    microbatches_per_step: int = Field(gt=0)
    seq_length: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    min_learning_rate: float = Field(gt=0.0)
    warmup_steps: int = Field(ge=0)
    weight_decay: float = Field(ge=0.0)
    clip_grad: float = Field(gt=0.0)
    global_seed: int = Field(ge=0, lt=2**63)
    checkpoint_every_steps: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_schedule(self) -> "MCoreTrainingLoopConfig":
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate")
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be smaller than max_steps")
        return self


class MCoreTrainingStepResult(StrictModel):
    global_step: int = Field(gt=0)
    consumed_microbatches: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    grad_norm: float = Field(ge=0.0)
    optimizer_step_verified: Literal[True] = True


class MCoreTrainingLoopResult(StrictModel):
    schema_version: Literal["sentinel.mcore-training-loop-result.v1"] = (
        "sentinel.mcore-training-loop-result.v1"
    )
    status: Literal["training_loop_completed"] = "training_loop_completed"
    start_step: int = Field(ge=0)
    final_step: int = Field(gt=0)
    resumed_from_checkpoint: bool
    step_results: list[MCoreTrainingStepResult] = Field(min_length=1)
    final_checkpoint_dir: str | None = None
    execution_authorized: Literal[False] = False


def _learning_rate_for_step(config: MCoreTrainingLoopConfig, step: int) -> float:
    if step < 0:
        raise ValueError("step cannot be negative")
    if config.warmup_steps and step < config.warmup_steps:
        progress = (step + 1) / config.warmup_steps
        return max(config.min_learning_rate, config.learning_rate * progress)
    decay_steps = max(1, config.max_steps - config.warmup_steps)
    decay_index = max(0, step - config.warmup_steps)
    progress = min(1.0, decay_index / decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * cosine


def _set_optimizer_lr(optimizer: Any, learning_rate: float) -> None:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    param_groups = getattr(optimizer, "param_groups", None)
    if param_groups is None:
        raise RuntimeError("Megatron optimizer does not expose param_groups")
    changed = False
    for group in param_groups:
        if isinstance(group, dict):
            group["lr"] = learning_rate
            changed = True
    if not changed:
        raise RuntimeError("Megatron optimizer exposed no mutable parameter groups")


def _step_result(optimizer: Any) -> tuple[bool, float]:
    value = optimizer.step()
    if isinstance(value, tuple):
        successful = bool(value[0])
        grad_norm_raw = value[1] if len(value) > 1 else 0.0
    else:
        successful = bool(value) if value is not None else True
        grad_norm_raw = 0.0
    if not successful:
        return False, 0.0
    try:
        grad_norm = float(grad_norm_raw.item())
    except AttributeError:
        grad_norm = float(grad_norm_raw or 0.0)
    return True, max(0.0, grad_norm)


def run_mcore_resumable_training_loop(
    runtime: MCoreDistributedRuntime,
    spec: ProductionMegatronModelSpec,
    config: MCoreTrainingLoopConfig,
    *,
    checkpoint_root: str | Path,
    resume_checkpoint: str | Path | None = None,
) -> MCoreTrainingLoopResult:
    """Execute a bounded, resumable MCore systems-training loop.

    The loop accumulates synthetic microbatches, performs distributed AdamW updates,
    checkpoints model+optimizer+training state, and can resume from that checkpoint.
    It is a systems-validation loop only: no cybersecurity corpus is consumed here.
    """
    if config.seq_length > spec.max_position_embeddings:
        raise ValueError("training seq_length exceeds model context window")
    if config.seq_length % (2 * spec.topology.context_parallel_size):
        raise ValueError("training seq_length must be divisible by 2*context_parallel_size")

    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)

    ddp_runtime = wrap_runtime_with_megatron_ddp(runtime)
    model = ddp_runtime.model
    optimizer = build_mcore_distributed_optimizer(
        ddp_runtime,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        clip_grad=config.clip_grad,
    )

    resumed = resume_checkpoint is not None
    if resume_checkpoint is None:
        initialize_mcore_native_parameters(model, global_seed=config.global_seed)
        training_state = MCoreTrainingState(
            global_step=0,
            consumed_microbatches=0,
            learning_rate=_learning_rate_for_step(config, 0),
            global_seed=config.global_seed,
        )
    else:
        training_state, _ = load_mcore_training_checkpoint(
            model,
            optimizer,
            checkpoint_dir=resume_checkpoint,
            expected_global_seed=config.global_seed,
        )
        if training_state.global_step >= config.max_steps:
            raise ValueError("resume checkpoint is already at or beyond max_steps")

    start_step = training_state.global_step
    results: list[MCoreTrainingStepResult] = []
    final_checkpoint: str | None = None

    for zero_based_step in range(start_step, config.max_steps):
        lr = _learning_rate_for_step(config, zero_based_step)
        _set_optimizer_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad_buffer()

        run_mcore_forward_backward_smoke(
            ddp_runtime,
            spec,
            seq_length=config.seq_length,
            global_seed=config.global_seed + zero_based_step * config.microbatches_per_step,
            num_microbatches=config.microbatches_per_step,
            initialize_parameters=False,
        )
        model.finish_grad_sync()
        successful, grad_norm = _step_result(optimizer)
        if not successful:
            raise RuntimeError(f"Megatron optimizer failed at global step {zero_based_step + 1}")

        training_state = MCoreTrainingState(
            global_step=zero_based_step + 1,
            consumed_microbatches=training_state.consumed_microbatches + config.microbatches_per_step,
            learning_rate=lr,
            global_seed=config.global_seed,
        )
        results.append(
            MCoreTrainingStepResult(
                global_step=training_state.global_step,
                consumed_microbatches=training_state.consumed_microbatches,
                learning_rate=lr,
                grad_norm=grad_norm,
            )
        )

        should_checkpoint = (
            training_state.global_step % config.checkpoint_every_steps == 0
            or training_state.global_step == config.max_steps
        )
        if should_checkpoint:
            step_dir = root / f"step-{training_state.global_step:08d}"
            save_mcore_training_checkpoint(
                model,
                optimizer,
                training_state,
                checkpoint_dir=step_dir,
            )
            final_checkpoint = step_dir.as_posix()

    if not results:
        raise RuntimeError("training loop executed zero steps")

    return MCoreTrainingLoopResult(
        start_step=start_step,
        final_step=training_state.global_step,
        resumed_from_checkpoint=resumed,
        step_results=results,
        final_checkpoint_dir=final_checkpoint,
    )
