from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_foundation_pretokenized import (
    PretokenizedCursor,
    PretokenizedFoundationReader,
    take_distributed_pretokenized_sequences,
)
from koschei_sentinel.production_mcore_ddp import wrap_runtime_with_megatron_ddp
from koschei_sentinel.production_mcore_distributed_runtime import MCoreDistributedRuntime
from koschei_sentinel.production_mcore_foundation_training import _run_real_batches
from koschei_sentinel.production_mcore_native_checkpoint import initialize_mcore_native_parameters
from koschei_sentinel.production_mcore_optimizer_smoke import build_mcore_distributed_optimizer
from koschei_sentinel.production_mcore_training_checkpoint import (
    MCoreTrainingState,
    load_mcore_training_checkpoint,
    save_mcore_training_checkpoint,
)
from koschei_sentinel.production_megatron_model_spec import ProductionMegatronModelSpec


class PretokenizedTrainingConfig(StrictModel):
    schema_version: Literal["sentinel.pretokenized-training-config.v1"] = (
        "sentinel.pretokenized-training-config.v1"
    )
    max_steps: int = Field(gt=0)
    microbatches_per_step: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    clip_grad: float = Field(gt=0.0)
    global_seed: int = Field(ge=0, lt=2**63)
    checkpoint_every_steps: int = Field(gt=0)


class PretokenizedStepMetrics(StrictModel):
    global_step: int = Field(gt=0)
    local_tokens: int = Field(gt=0)
    global_tokens: int = Field(gt=0)
    elapsed_seconds: float = Field(gt=0.0)
    local_tokens_per_second: float = Field(gt=0.0)
    global_tokens_per_second: float = Field(gt=0.0)


class PretokenizedTrainingResult(StrictModel):
    schema_version: Literal["sentinel.pretokenized-training-result.v1"] = (
        "sentinel.pretokenized-training-result.v1"
    )
    status: Literal["pretokenized_training_completed"] = "pretokenized_training_completed"
    start_step: int = Field(ge=0)
    final_step: int = Field(gt=0)
    resumed_from_checkpoint: bool
    step_metrics: list[PretokenizedStepMetrics] = Field(min_length=1)
    final_checkpoint_dir: str | None = None
    mmap_data_path_used: Literal[True] = True
    online_tokenization_used: Literal[False] = False
    execution_authorized: Literal[False] = False


def run_pretokenized_training(
    runtime: MCoreDistributedRuntime,
    spec: ProductionMegatronModelSpec,
    train_config: PretokenizedTrainingConfig,
    *,
    manifest_path: str | Path,
    checkpoint_root: str | Path,
    resume_checkpoint: str | Path | None = None,
) -> PretokenizedTrainingResult:
    try:
        import torch
        import torch.distributed as dist
        from megatron.core import parallel_state
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch distributed and Megatron-Core are required") from exc

    ddp_runtime = wrap_runtime_with_megatron_ddp(runtime)
    model = ddp_runtime.model
    optimizer = build_mcore_distributed_optimizer(
        ddp_runtime,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        clip_grad=train_config.clip_grad,
    )

    resumed = resume_checkpoint is not None
    cursor: PretokenizedCursor | None = None
    if resumed:
        state, _ = load_mcore_training_checkpoint(
            model,
            optimizer,
            checkpoint_dir=resume_checkpoint,
            expected_global_seed=train_config.global_seed,
        )
        raw_cursor = state.data_state
        if not isinstance(raw_cursor, dict) or raw_cursor.get("schema_version") != "sentinel.foundation-pretokenized-cursor.v1":
            raise ValueError("resume checkpoint lacks a compatible pretokenized cursor")
        cursor = PretokenizedCursor.model_validate(raw_cursor)
    else:
        initialize_mcore_native_parameters(model, global_seed=train_config.global_seed)
        state = MCoreTrainingState(
            global_step=0,
            consumed_microbatches=0,
            learning_rate=train_config.learning_rate,
            global_seed=train_config.global_seed,
            data_state=None,
        )

    start_step = state.global_step
    if start_step >= train_config.max_steps:
        raise ValueError("resume checkpoint is already at or beyond max_steps")

    dp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=False)
    dp_size = parallel_state.get_data_parallel_world_size(with_context_parallel=False)
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    final_checkpoint: str | None = None
    metrics: list[PretokenizedStepMetrics] = []

    with PretokenizedFoundationReader(manifest_path) as reader:
        if reader.manifest.sequence_length > spec.max_position_embeddings:
            raise ValueError("pretokenized sequence length exceeds model context window")
        if reader.manifest.sequence_length % (2 * spec.topology.context_parallel_size):
            raise ValueError("pretokenized sequence length must be divisible by 2*context_parallel_size")
        if cursor is not None and cursor.manifest_sha256 != reader.manifest_sha256:
            raise ValueError("resume checkpoint pretokenized manifest mismatch")

        for zero_based_step in range(start_step, train_config.max_steps):
            sequences, cursor = take_distributed_pretokenized_sequences(
                reader,
                state=cursor,
                data_parallel_rank=dp_rank,
                data_parallel_size=dp_size,
                count=train_config.microbatches_per_step,
            )
            optimizer.zero_grad(set_to_none=True)
            model.zero_grad_buffer()
            dist.barrier()
            torch.cuda.synchronize(runtime.local_rank)
            started = time.perf_counter()
            _run_real_batches(ddp_runtime, spec, sequences)
            model.finish_grad_sync()
            step_result = optimizer.step()
            successful = bool(step_result[0]) if isinstance(step_result, tuple) else bool(
                step_result if step_result is not None else True
            )
            if not successful:
                raise RuntimeError(f"pretokenized optimizer failed at step {zero_based_step + 1}")
            torch.cuda.synchronize(runtime.local_rank)
            dist.barrier()
            elapsed = time.perf_counter() - started
            if elapsed <= 0:
                raise RuntimeError("invalid non-positive training step duration")

            local_tokens = reader.manifest.sequence_length * len(sequences)
            global_tokens = local_tokens * dp_size
            metrics.append(
                PretokenizedStepMetrics(
                    global_step=zero_based_step + 1,
                    local_tokens=local_tokens,
                    global_tokens=global_tokens,
                    elapsed_seconds=elapsed,
                    local_tokens_per_second=local_tokens / elapsed,
                    global_tokens_per_second=global_tokens / elapsed,
                )
            )
            state = MCoreTrainingState(
                global_step=zero_based_step + 1,
                consumed_microbatches=state.consumed_microbatches + train_config.microbatches_per_step,
                learning_rate=train_config.learning_rate,
                global_seed=train_config.global_seed,
                data_state=cursor.model_dump(mode="json"),
            )
            should_checkpoint = (
                state.global_step % train_config.checkpoint_every_steps == 0
                or state.global_step == train_config.max_steps
            )
            if should_checkpoint:
                step_dir = root / f"step-{state.global_step:08d}"
                save_mcore_training_checkpoint(model, optimizer, state, checkpoint_dir=step_dir)
                final_checkpoint = step_dir.as_posix()

    return PretokenizedTrainingResult(
        start_step=start_step,
        final_step=train_config.max_steps,
        resumed_from_checkpoint=resumed,
        step_metrics=metrics,
        final_checkpoint_dir=final_checkpoint,
    )
