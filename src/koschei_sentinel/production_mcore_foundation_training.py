from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_foundation_data import (
    FoundationDataConfig,
    FoundationSamplerState,
    PackedFoundationSequence,
    take_distributed_foundation_sequences,
)
from koschei_sentinel.production_mcore_ddp import wrap_runtime_with_megatron_ddp
from koschei_sentinel.production_mcore_distributed_runtime import MCoreDistributedRuntime
from koschei_sentinel.production_mcore_native_checkpoint import initialize_mcore_native_parameters
from koschei_sentinel.production_mcore_optimizer_smoke import build_mcore_distributed_optimizer
from koschei_sentinel.production_mcore_training_checkpoint import (
    MCoreTrainingState,
    load_mcore_training_checkpoint,
    save_mcore_training_checkpoint,
)
from koschei_sentinel.production_megatron_model_spec import ProductionMegatronModelSpec


class FoundationTrainingConfig(StrictModel):
    schema_version: Literal["sentinel.foundation-training-config.v1"] = "sentinel.foundation-training-config.v1"
    max_steps: int = Field(gt=0)
    microbatches_per_step: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    clip_grad: float = Field(gt=0.0)
    global_seed: int = Field(ge=0, lt=2**63)
    checkpoint_every_steps: int = Field(gt=0)


class FoundationTrainingResult(StrictModel):
    schema_version: Literal["sentinel.foundation-training-result.v1"] = "sentinel.foundation-training-result.v1"
    status: Literal["foundation_batches_executed"] = "foundation_batches_executed"
    start_step: int = Field(ge=0)
    final_step: int = Field(gt=0)
    consumed_global_sequences: int = Field(gt=0)
    resumed_from_checkpoint: bool
    final_checkpoint_dir: str | None = None
    foundation_data_consumed: Literal[True] = True
    synthetic_data_used: Literal[False] = False
    execution_authorized: Literal[False] = False


def _to_device_batch(sequence: PackedFoundationSequence, *, device):
    import torch
    return {
        "tokens": torch.tensor([sequence.input_ids], device=device, dtype=torch.long),
        "labels": torch.tensor([sequence.labels], device=device, dtype=torch.long),
        "loss_mask": torch.tensor([sequence.loss_mask], device=device, dtype=torch.float32),
        "position_ids": torch.tensor([sequence.position_ids], device=device, dtype=torch.long),
        "attention_mask": None,
        "cu_seqlens": None,
    }


def _run_real_batches(runtime, spec, sequences: list[PackedFoundationSequence]) -> None:
    try:
        import torch
        from megatron.core import parallel_state
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
        from megatron.core.utils import get_batch_on_this_cp_rank
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch and Megatron-Core are required for foundation training") from exc

    device = torch.device("cuda", runtime.local_rank)
    cp_size = spec.topology.context_parallel_size
    batches = []
    for sequence in sequences:
        batch = _to_device_batch(sequence, device=device)
        if cp_size > 1:
            batch = get_batch_on_this_cp_rank(
                batch,
                is_hybrid_cp=False,
                cp_group=parallel_state.get_context_parallel_group(),
            )
        batches.append(batch)

    def loss_func(mask, output_tensor):
        losses = output_tensor.float().view(-1)
        local_mask = mask.float().view(-1)
        if losses.numel() != local_mask.numel():
            raise RuntimeError("foundation loss and mask shape mismatch")
        denom = local_mask.sum()
        if denom.item() <= 0:
            raise RuntimeError("foundation loss mask is empty")
        loss = (losses * local_mask).sum() / denom
        return loss, {"foundation_lm_loss": loss.detach()}

    def forward_step(data_iterator, stage_model, checkpoint_activations_microbatch=None):
        del checkpoint_activations_microbatch
        local = next(data_iterator)
        output = stage_model(
            input_ids=local["tokens"],
            position_ids=local["position_ids"],
            attention_mask=local.get("attention_mask"),
            labels=local["labels"],
            loss_mask=local["loss_mask"],
        )
        return output, partial(loss_func, local["loss_mask"])

    get_forward_backward_func()(
        forward_step_func=forward_step,
        data_iterator=iter(batches),
        model=runtime.model,
        num_microbatches=len(batches),
        seq_length=spec.transformer.max_position_embeddings if hasattr(spec.transformer, "max_position_embeddings") else len(sequences[0].input_ids),
        micro_batch_size=1,
        forward_only=False,
        collect_non_loss_data=False,
    )


def run_foundation_training(
    runtime: MCoreDistributedRuntime,
    spec: ProductionMegatronModelSpec,
    data_config: FoundationDataConfig,
    train_config: FoundationTrainingConfig,
    *,
    checkpoint_root: str | Path,
    resume_checkpoint: str | Path | None = None,
    sampler_state: FoundationSamplerState | None = None,
) -> tuple[FoundationTrainingResult, FoundationSamplerState]:
    try:
        from megatron.core import parallel_state
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Megatron-Core is required for foundation training") from exc

    if data_config.seq_length > spec.max_position_embeddings:
        raise ValueError("foundation seq_length exceeds model context window")
    if data_config.seq_length % (2 * spec.topology.context_parallel_size):
        raise ValueError("foundation seq_length must be divisible by 2*context_parallel_size")

    ddp_runtime = wrap_runtime_with_megatron_ddp(runtime)
    model = ddp_runtime.model
    optimizer = build_mcore_distributed_optimizer(
        ddp_runtime,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        clip_grad=train_config.clip_grad,
    )
    resumed = resume_checkpoint is not None
    if resumed:
        training_state, metadata = load_mcore_training_checkpoint(
            model,
            optimizer,
            checkpoint_dir=resume_checkpoint,
            expected_global_seed=train_config.global_seed,
        )
        if sampler_state is None:
            payload = metadata.get("foundation_sampler_state") if isinstance(metadata, dict) else None
            if payload is None:
                raise ValueError("resume checkpoint lacks foundation sampler state")
            sampler_state = FoundationSamplerState.model_validate(payload)
    else:
        initialize_mcore_native_parameters(model, global_seed=train_config.global_seed)
        training_state = MCoreTrainingState(
            global_step=0,
            consumed_microbatches=0,
            learning_rate=train_config.learning_rate,
            global_seed=train_config.global_seed,
        )

    start_step = training_state.global_step
    dp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=False)
    dp_size = parallel_state.get_data_parallel_world_size(with_context_parallel=False)
    final_checkpoint: str | None = None
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)

    for step in range(start_step, train_config.max_steps):
        sequences, sampler_state = take_distributed_foundation_sequences(
            data_config,
            state=sampler_state,
            data_parallel_rank=dp_rank,
            data_parallel_size=dp_size,
            count=train_config.microbatches_per_step,
        )
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad_buffer()
        _run_real_batches(ddp_runtime, spec, sequences)
        model.finish_grad_sync()
        step_result = optimizer.step()
        successful = bool(step_result[0]) if isinstance(step_result, tuple) else bool(step_result if step_result is not None else True)
        if not successful:
            raise RuntimeError(f"foundation optimizer failed at step {step + 1}")
        training_state = MCoreTrainingState(
            global_step=step + 1,
            consumed_microbatches=training_state.consumed_microbatches + train_config.microbatches_per_step,
            learning_rate=train_config.learning_rate,
            global_seed=train_config.global_seed,
        )
        if training_state.global_step % train_config.checkpoint_every_steps == 0 or training_state.global_step == train_config.max_steps:
            step_dir = root / f"step-{training_state.global_step:08d}"
            save_mcore_training_checkpoint(
                model,
                optimizer,
                training_state,
                checkpoint_dir=step_dir,
                extra_metadata={"foundation_sampler_state": sampler_state.model_dump(mode="json")},
            )
            final_checkpoint = step_dir.as_posix()

    if sampler_state is None:
        raise RuntimeError("foundation sampler state was not produced")
    return FoundationTrainingResult(
        start_step=start_step,
        final_step=train_config.max_steps,
        consumed_global_sequences=sampler_state.global_sequence_offset,
        resumed_from_checkpoint=resumed,
        final_checkpoint_dir=final_checkpoint,
    ), sampler_state
