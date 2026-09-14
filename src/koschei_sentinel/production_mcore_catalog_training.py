from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_foundation_catalog import FoundationCatalogCursor, load_foundation_catalog
from koschei_sentinel.production_foundation_prefetch import FoundationDoubleBuffer
from koschei_sentinel.production_foundation_worker_prefetch import FoundationWorkerPrefetch
from koschei_sentinel.production_mcore_ddp import wrap_runtime_with_megatron_ddp
from koschei_sentinel.production_mcore_distributed_runtime import MCoreDistributedRuntime
from koschei_sentinel.production_mcore_native_checkpoint import initialize_mcore_native_parameters
from koschei_sentinel.production_mcore_optimizer_smoke import build_mcore_distributed_optimizer
from koschei_sentinel.production_mcore_router_probe import MCoreRouterProbe
from koschei_sentinel.production_mcore_stability_telemetry import (
    MCoreStabilityStepTelemetry,
    StabilityThresholds,
    collect_mcore_stability_telemetry,
)
from koschei_sentinel.production_mcore_training_checkpoint import (
    MCoreTrainingState,
    load_mcore_training_checkpoint,
    save_mcore_training_checkpoint,
)
from koschei_sentinel.production_megatron_model_spec import ProductionMegatronModelSpec


class CatalogTrainingConfig(StrictModel):
    schema_version: Literal["sentinel.catalog-training-config.v1"] = "sentinel.catalog-training-config.v1"
    max_steps: int = Field(gt=0)
    microbatches_per_step: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    clip_grad: float = Field(gt=0.0)
    global_seed: int = Field(ge=0, lt=2**63)
    checkpoint_every_steps: int = Field(gt=0)
    prefetch_workers: int = Field(gt=0)
    prefetch_depth: int = Field(ge=2)
    stability_thresholds: StabilityThresholds = Field(
        default_factory=lambda: StabilityThresholds(
            max_grad_norm=100.0,
            max_cuda_allocated_fraction=0.95,
            min_expert_utilization_fraction=0.50,
            max_expert_load_cv=2.0,
        )
    )


class CatalogTrainingStepMetrics(StrictModel):
    global_step: int = Field(gt=0)
    local_tokens: int = Field(gt=0)
    global_tokens: int = Field(gt=0)
    elapsed_seconds: float = Field(gt=0.0)
    local_tokens_per_second: float = Field(gt=0.0)
    global_tokens_per_second: float = Field(gt=0.0)
    stability: MCoreStabilityStepTelemetry
    worker_prefetch: Literal[True] = True
    pinned_memory: Literal[True] = True
    async_h2d: Literal[True] = True
    double_buffered: Literal[True] = True


class CatalogTrainingResult(StrictModel):
    schema_version: Literal["sentinel.catalog-training-result.v1"] = "sentinel.catalog-training-result.v1"
    status: Literal["catalog_training_completed"] = "catalog_training_completed"
    start_step: int = Field(ge=0)
    final_step: int = Field(gt=0)
    resumed_from_checkpoint: bool
    catalog_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    step_metrics: list[CatalogTrainingStepMetrics] = Field(min_length=1)
    final_checkpoint_dir: str | None = None
    all_steps_stable: Literal[True] = True
    execution_authorized: Literal[False] = False


def _run_prefetched_batches(runtime, spec, batches) -> None:
    try:
        from megatron.core import parallel_state
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
        from megatron.core.utils import get_batch_on_this_cp_rank
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Megatron-Core is required for catalog training") from exc

    cp_size = spec.topology.context_parallel_size
    prepared = []
    for batch in batches:
        local = {
            "tokens": batch.tokens,
            "labels": batch.labels,
            "loss_mask": batch.loss_mask,
            "position_ids": batch.position_ids,
            "attention_mask": None,
            "cu_seqlens": None,
        }
        if cp_size > 1:
            local = get_batch_on_this_cp_rank(
                local,
                is_hybrid_cp=False,
                cp_group=parallel_state.get_context_parallel_group(),
            )
        prepared.append(local)

    if not prepared:
        raise RuntimeError("catalog prefetch produced no batches")

    def loss_func(mask, output_tensor):
        losses = output_tensor.float().view(-1)
        local_mask = mask.float().view(-1)
        if losses.numel() != local_mask.numel():
            raise RuntimeError("catalog loss and mask shape mismatch")
        denom = local_mask.sum()
        if denom.item() <= 0:
            raise RuntimeError("catalog loss mask is empty")
        loss = (losses * local_mask).sum() / denom
        return loss, {"catalog_lm_loss": loss.detach()}

    def forward_step(data_iterator, stage_model, checkpoint_activations_microbatch=None):
        del checkpoint_activations_microbatch
        local = next(data_iterator)
        output = stage_model(
            input_ids=local["tokens"],
            position_ids=local["position_ids"],
            attention_mask=local["attention_mask"],
            labels=local["labels"],
            loss_mask=local["loss_mask"],
        )
        return output, partial(loss_func, local["loss_mask"])

    get_forward_backward_func()(
        forward_step_func=forward_step,
        data_iterator=iter(prepared),
        model=runtime.model,
        num_microbatches=len(prepared),
        seq_length=prepared[0]["tokens"].shape[1] * cp_size,
        micro_batch_size=1,
        forward_only=False,
        collect_non_loss_data=False,
    )


def _optimizer_step_result(value) -> tuple[bool, float]:
    if isinstance(value, tuple):
        successful = bool(value[0])
        raw_grad_norm = value[1] if len(value) > 1 else 0.0
    else:
        successful = bool(value) if value is not None else True
        raw_grad_norm = 0.0
    try:
        grad_norm = float(raw_grad_norm.item())
    except AttributeError:
        grad_norm = float(raw_grad_norm or 0.0)
    return successful, grad_norm


def run_catalog_training(
    runtime: MCoreDistributedRuntime,
    spec: ProductionMegatronModelSpec,
    config: CatalogTrainingConfig,
    *,
    catalog_path: str | Path,
    checkpoint_root: str | Path,
    resume_checkpoint: str | Path | None = None,
) -> CatalogTrainingResult:
    try:
        import torch
        import torch.distributed as dist
        from megatron.core import parallel_state
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch distributed and Megatron-Core are required") from exc

    catalog = load_foundation_catalog(catalog_path)
    if catalog.sequence_length > spec.max_position_embeddings:
        raise ValueError("catalog sequence length exceeds model context window")
    if catalog.sequence_length % (2 * spec.topology.context_parallel_size):
        raise ValueError("catalog sequence length must be divisible by 2*context_parallel_size")
    if config.prefetch_depth < config.prefetch_workers:
        raise ValueError("prefetch_depth must be >= prefetch_workers")

    ddp_runtime = wrap_runtime_with_megatron_ddp(runtime)
    model = ddp_runtime.model
    optimizer = build_mcore_distributed_optimizer(
        ddp_runtime,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        clip_grad=config.clip_grad,
    )
    router_probe = MCoreRouterProbe()
    router_probe.attach(model)

    resumed = resume_checkpoint is not None
    cursor: FoundationCatalogCursor | None = None
    if resumed:
        state, _ = load_mcore_training_checkpoint(
            model,
            optimizer,
            checkpoint_dir=resume_checkpoint,
            expected_global_seed=config.global_seed,
        )
        raw = state.data_state
        if not isinstance(raw, dict) or raw.get("schema_version") != "sentinel.foundation-catalog-cursor.v1":
            raise ValueError("resume checkpoint lacks compatible catalog cursor")
        cursor = FoundationCatalogCursor.model_validate(raw)
        if cursor.catalog_sha256 != catalog.catalog_sha256:
            raise ValueError("resume checkpoint catalog digest mismatch")
    else:
        initialize_mcore_native_parameters(model, global_seed=config.global_seed)
        state = MCoreTrainingState(
            global_step=0,
            consumed_microbatches=0,
            learning_rate=config.learning_rate,
            global_seed=config.global_seed,
            data_state=None,
        )

    start_step = state.global_step
    if start_step >= config.max_steps:
        raise ValueError("resume checkpoint is already at or beyond max_steps")

    dp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=False)
    dp_size = parallel_state.get_data_parallel_world_size(with_context_parallel=False)
    if cursor is not None and cursor.data_parallel_size != dp_size:
        raise ValueError("resume checkpoint data parallel size mismatch")

    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    metrics: list[CatalogTrainingStepMetrics] = []
    final_checkpoint: str | None = None
    device = torch.device("cuda", runtime.local_rank)
    global_offset = cursor.global_sequence_offset if cursor is not None else 0

    try:
        for zero_based_step in range(start_step, config.max_steps):
            window = config.microbatches_per_step * dp_size
            indices = [
                ordinal
                for ordinal in range(global_offset, global_offset + window)
                if (ordinal - global_offset) % dp_size == dp_rank
            ]
            worker_prefetch = FoundationWorkerPrefetch(
                catalog_path,
                indices,
                workers=config.prefetch_workers,
                prefetch_depth=config.prefetch_depth,
            )
            sequences = [item.sequence for item in worker_prefetch]
            if len(sequences) != config.microbatches_per_step:
                raise RuntimeError("catalog worker prefetch returned wrong local batch count")

            optimizer.zero_grad(set_to_none=True)
            model.zero_grad_buffer()
            router_probe.reset()
            torch.cuda.reset_peak_memory_stats(runtime.local_rank)
            double_buffer = FoundationDoubleBuffer(sequences, device=device, depth=2)

            dist.barrier()
            torch.cuda.synchronize(runtime.local_rank)
            started = time.perf_counter()
            _run_prefetched_batches(ddp_runtime, spec, double_buffer)
            model.finish_grad_sync()
            step_result = optimizer.step()
            successful, grad_norm = _optimizer_step_result(step_result)
            if not successful:
                raise RuntimeError(f"catalog optimizer failed at step {zero_based_step + 1}")
            torch.cuda.synchronize(runtime.local_rank)
            dist.barrier()
            elapsed = time.perf_counter() - started
            if elapsed <= 0:
                raise RuntimeError("catalog training step duration must be positive")

            global_offset += window
            cursor = FoundationCatalogCursor(
                global_sequence_offset=global_offset,
                catalog_sha256=catalog.catalog_sha256,
                data_parallel_size=dp_size,
            )
            local_tokens = catalog.sequence_length * config.microbatches_per_step
            global_tokens = local_tokens * dp_size
            router_snapshot = router_probe.snapshot()
            stability = collect_mcore_stability_telemetry(
                model=model,
                global_step=zero_based_step + 1,
                local_tokens=local_tokens,
                global_tokens=global_tokens,
                elapsed_seconds=elapsed,
                grad_norm=grad_norm,
                thresholds=config.stability_thresholds,
                local_rank=runtime.local_rank,
                aux_loss=router_snapshot.aux_loss,
                z_loss=router_snapshot.z_loss,
                tokens_per_expert=router_snapshot.tokens_per_expert,
            )
            if not stability.stability_passed:
                raise RuntimeError(
                    f"training stability gate failed at step {zero_based_step + 1}: {stability.blockers}"
                )

            metrics.append(
                CatalogTrainingStepMetrics(
                    global_step=zero_based_step + 1,
                    local_tokens=local_tokens,
                    global_tokens=global_tokens,
                    elapsed_seconds=elapsed,
                    local_tokens_per_second=local_tokens / elapsed,
                    global_tokens_per_second=global_tokens / elapsed,
                    stability=stability,
                )
            )
            state = MCoreTrainingState(
                global_step=zero_based_step + 1,
                consumed_microbatches=state.consumed_microbatches + config.microbatches_per_step,
                learning_rate=config.learning_rate,
                global_seed=config.global_seed,
                data_state=cursor.model_dump(mode="json"),
            )
            if state.global_step % config.checkpoint_every_steps == 0 or state.global_step == config.max_steps:
                step_dir = root / f"step-{state.global_step:08d}"
                save_mcore_training_checkpoint(model, optimizer, state, checkpoint_dir=step_dir)
                final_checkpoint = step_dir.as_posix()
    finally:
        router_probe.detach()

    return CatalogTrainingResult(
        start_step=start_step,
        final_step=config.max_steps,
        resumed_from_checkpoint=resumed,
        catalog_sha256=catalog.catalog_sha256,
        step_metrics=metrics,
        final_checkpoint_dir=final_checkpoint,
    )
