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
from koschei_sentinel.production_mcore_activation_probe import MCoreActivationProbe
from koschei_sentinel.production_mcore_ddp import wrap_runtime_with_megatron_ddp
from koschei_sentinel.production_mcore_distributed_runtime import MCoreDistributedRuntime
from koschei_sentinel.production_mcore_moe_metrics_probe import clear_native_moe_metrics, read_native_moe_losses
from koschei_sentinel.production_mcore_native_checkpoint import initialize_mcore_native_parameters
from koschei_sentinel.production_mcore_optimizer_smoke import build_mcore_distributed_optimizer
from koschei_sentinel.production_mcore_recovery import (
    RecoveryDecision,
    RecoveryLedger,
    RecoveryPolicy,
    build_recovery_decision,
    distributed_any_unstable,
    fingerprint_batch,
)
from koschei_sentinel.production_mcore_resume_bundle import (
    attach_catalog_resume_state,
    restore_catalog_resume_bundle,
)
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
from koschei_sentinel.production_mcore_training_trends import (
    TrainingTrendSnapshot,
    TrainingTrendThresholds,
    TrainingTrendTracker,
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
    trend_thresholds: TrainingTrendThresholds = Field(
        default_factory=lambda: TrainingTrendThresholds(
            history_window=32,
            min_history_for_spike=3,
            max_loss_ratio_to_median=2.5,
            max_grad_norm_ratio_to_median=4.0,
            max_activation_rms_ratio_to_median=4.0,
            max_activation_abs=1.0e4,
            max_router_utilization_drop=0.35,
        )
    )
    recovery_policy: RecoveryPolicy = Field(
        default_factory=lambda: RecoveryPolicy(
            max_retries_per_batch=2,
            learning_rate_backoff=0.5,
            min_learning_rate=1.0e-7,
            quarantine_after_failures=3,
            skip_quarantined_batch=True,
        )
    )


class CatalogTrainingStepMetrics(StrictModel):
    global_step: int = Field(gt=0)
    local_tokens: int = Field(gt=0)
    global_tokens: int = Field(gt=0)
    elapsed_seconds: float = Field(gt=0.0)
    local_tokens_per_second: float = Field(gt=0.0)
    global_tokens_per_second: float = Field(gt=0.0)
    lm_loss: float | None = Field(default=None, ge=0.0)
    stability: MCoreStabilityStepTelemetry
    trend: TrainingTrendSnapshot
    recovery_attempts_before_success: int = Field(ge=0)
    effective_learning_rate: float = Field(gt=0.0)
    worker_prefetch: Literal[True] = True
    pinned_memory: Literal[True] = True
    async_h2d: Literal[True] = True
    double_buffered: Literal[True] = True


class CatalogQuarantineEvent(StrictModel):
    batch_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    global_sequence_offset: int = Field(ge=0)
    global_window: int = Field(gt=0)
    failure_count: int = Field(gt=0)
    skipped: bool
    reason_codes: list[str] = Field(default_factory=list)


class CatalogTrainingResult(StrictModel):
    schema_version: Literal["sentinel.catalog-training-result.v1"] = "sentinel.catalog-training-result.v1"
    status: Literal["catalog_training_completed"] = "catalog_training_completed"
    start_step: int = Field(ge=0)
    final_step: int = Field(gt=0)
    resumed_from_checkpoint: bool
    catalog_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    step_metrics: list[CatalogTrainingStepMetrics] = Field(default_factory=list)
    quarantine_events: list[CatalogQuarantineEvent] = Field(default_factory=list)
    total_retries: int = Field(ge=0)
    final_learning_rate: float = Field(gt=0.0)
    final_checkpoint_dir: str | None = None
    all_steps_stable: Literal[True] = True
    distributed_recovery_consensus: Literal[True] = True
    execution_authorized: Literal[False] = False


def _run_prefetched_batches(runtime, spec, batches) -> float | None:
    try:
        import torch
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

    local_losses: list[float] = []

    def loss_func(mask, output_tensor):
        losses = output_tensor.float().view(-1)
        local_mask = mask.float().view(-1)
        if losses.numel() != local_mask.numel():
            raise RuntimeError("catalog loss and mask shape mismatch")
        denom = local_mask.sum()
        if denom.item() <= 0:
            raise RuntimeError("catalog loss mask is empty")
        loss = (losses * local_mask).sum() / denom
        local_losses.append(float(loss.detach().item()))
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
    local_sum = float(sum(local_losses))
    local_count = float(len(local_losses))
    payload = torch.tensor(
        [local_sum, local_count],
        device=prepared[0]["tokens"].device,
        dtype=torch.float64,
    )
    torch.distributed.all_reduce(payload, op=torch.distributed.ReduceOp.SUM)
    total_sum, total_count = float(payload[0].item()), float(payload[1].item())
    return total_sum / total_count if total_count > 0 else None


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


def _restore_last_good(model, optimizer, *, checkpoint_dir: str | None, global_seed: int) -> MCoreTrainingState | None:
    if checkpoint_dir is None:
        return None
    restored_state, _ = load_mcore_training_checkpoint(
        model,
        optimizer,
        checkpoint_dir=checkpoint_dir,
        expected_global_seed=global_seed,
    )
    return restored_state


def _set_optimizer_lr(optimizer, learning_rate: float) -> None:
    groups = getattr(optimizer, "param_groups", None)
    if groups is None:
        raise RuntimeError("Megatron optimizer does not expose param_groups for recovery LR backoff")
    changed = False
    for group in groups:
        if isinstance(group, dict):
            group["lr"] = learning_rate
            changed = True
    if not changed:
        raise RuntimeError("Megatron optimizer exposed no mutable parameter groups")


def _attach_runtime_state(
    state: MCoreTrainingState,
    *,
    catalog_sha256: str,
    current_learning_rate: float,
    ledger: RecoveryLedger,
    trend_tracker: TrainingTrendTracker,
    total_retries: int,
    quarantine_events: list[CatalogQuarantineEvent],
) -> MCoreTrainingState:
    return attach_catalog_resume_state(
        state.model_copy(update={"learning_rate": current_learning_rate}),
        catalog_sha256=catalog_sha256,
        current_learning_rate=current_learning_rate,
        ledger=ledger,
        trend_tracker=trend_tracker,
        total_retries=total_retries,
        quarantine_events=[item.model_dump(mode="json") for item in quarantine_events],
    )


def _save_runtime_checkpoint(
    model,
    optimizer,
    state: MCoreTrainingState,
    *,
    checkpoint_dir: Path,
    catalog_sha256: str,
    current_learning_rate: float,
    ledger: RecoveryLedger,
    trend_tracker: TrainingTrendTracker,
    total_retries: int,
    quarantine_events: list[CatalogQuarantineEvent],
) -> tuple[MCoreTrainingState, str]:
    attached = _attach_runtime_state(
        state,
        catalog_sha256=catalog_sha256,
        current_learning_rate=current_learning_rate,
        ledger=ledger,
        trend_tracker=trend_tracker,
        total_retries=total_retries,
        quarantine_events=quarantine_events,
    )
    save_mcore_training_checkpoint(model, optimizer, attached, checkpoint_dir=checkpoint_dir)
    return attached, checkpoint_dir.as_posix()


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

    if resume_checkpoint is None:
        initialize_mcore_native_parameters(runtime.model, global_seed=config.global_seed)
    ddp_runtime = wrap_runtime_with_megatron_ddp(runtime)
    model = ddp_runtime.model
    optimizer = build_mcore_distributed_optimizer(
        ddp_runtime,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        clip_grad=config.clip_grad,
    )

    router_probe = MCoreRouterProbe()
    activation_probe = MCoreActivationProbe()
    router_probe.attach(model)
    activation_probe.attach(model)

    resumed = resume_checkpoint is not None
    cursor: FoundationCatalogCursor | None = None
    recovery_ledger = RecoveryLedger.empty()
    trend_tracker = TrainingTrendTracker(config.trend_thresholds)
    quarantines: list[CatalogQuarantineEvent] = []
    total_retries = 0

    if resumed:
        state, _ = load_mcore_training_checkpoint(
            model,
            optimizer,
            checkpoint_dir=resume_checkpoint,
            expected_global_seed=config.global_seed,
        )
        bundle = restore_catalog_resume_bundle(
            state,
            expected_catalog_sha256=catalog.catalog_sha256,
            trend_thresholds=config.trend_thresholds,
        )
        cursor = bundle.cursor
        recovery_ledger = bundle.ledger
        trend_tracker = bundle.trend_tracker
        current_lr = bundle.current_learning_rate
        total_retries = bundle.total_retries
        quarantines = [CatalogQuarantineEvent.model_validate(item) for item in bundle.quarantine_events]
        _set_optimizer_lr(optimizer, current_lr)
    else:
        current_lr = config.learning_rate
        state = MCoreTrainingState(
            global_step=0,
            consumed_microbatches=0,
            learning_rate=current_lr,
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
    final_checkpoint: str | None = str(resume_checkpoint) if resume_checkpoint is not None else None

    if final_checkpoint is None:
        state, final_checkpoint = _save_runtime_checkpoint(
            model,
            optimizer,
            state,
            checkpoint_dir=root / "recovery-base-step-00000000",
            catalog_sha256=catalog.catalog_sha256,
            current_learning_rate=current_lr,
            ledger=recovery_ledger,
            trend_tracker=trend_tracker,
            total_retries=total_retries,
            quarantine_events=quarantines,
        )

    device = torch.device("cuda", runtime.local_rank)
    global_offset = cursor.global_sequence_offset if cursor is not None else 0
    completed_step = start_step

    try:
        while completed_step < config.max_steps:
            window = config.microbatches_per_step * dp_size
            batch_fingerprint = fingerprint_batch(
                catalog_sha256=catalog.catalog_sha256,
                global_sequence_offset=global_offset,
                global_window=window,
            )
            if batch_fingerprint in recovery_ledger.quarantined:
                global_offset += window
                cursor = FoundationCatalogCursor(
                    global_sequence_offset=global_offset,
                    catalog_sha256=catalog.catalog_sha256,
                    data_parallel_size=dp_size,
                )
                state = state.model_copy(update={"data_state": cursor.model_dump(mode="json")})
                state, final_checkpoint = _save_runtime_checkpoint(
                    model,
                    optimizer,
                    state,
                    checkpoint_dir=root / f"recovery-skip-{global_offset:016d}",
                    catalog_sha256=catalog.catalog_sha256,
                    current_learning_rate=current_lr,
                    ledger=recovery_ledger,
                    trend_tracker=trend_tracker,
                    total_retries=total_retries,
                    quarantine_events=quarantines,
                )
                continue

            retry_attempts = 0
            step_committed = False
            while not step_committed:
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
                activation_probe.reset()
                clear_native_moe_metrics()
                torch.cuda.reset_peak_memory_stats(runtime.local_rank)
                double_buffer = FoundationDoubleBuffer(sequences, device=device, depth=2)

                dist.barrier()
                torch.cuda.synchronize(runtime.local_rank)
                started = time.perf_counter()
                local_exception: Exception | None = None
                lm_loss: float | None = None
                grad_norm = 0.0
                stability = None
                trend = None
                local_blockers: list[str] = []
                try:
                    lm_loss = _run_prefetched_batches(ddp_runtime, spec, double_buffer)
                    model.finish_grad_sync()
                    native_moe_losses = read_native_moe_losses()
                    step_result = optimizer.step()
                    successful, grad_norm = _optimizer_step_result(step_result)
                    if not successful:
                        local_blockers.append("optimizer_step_failed")
                    torch.cuda.synchronize(runtime.local_rank)
                    elapsed = time.perf_counter() - started
                    if elapsed <= 0:
                        local_blockers.append("non_positive_step_duration")
                        elapsed = 1.0e-9

                    local_tokens = catalog.sequence_length * config.microbatches_per_step
                    global_tokens = local_tokens * dp_size
                    router_snapshot = router_probe.snapshot()
                    activation_snapshot = activation_probe.snapshot()
                    stability = collect_mcore_stability_telemetry(
                        model=model,
                        global_step=completed_step + 1,
                        local_tokens=local_tokens,
                        global_tokens=global_tokens,
                        elapsed_seconds=elapsed,
                        grad_norm=grad_norm,
                        thresholds=config.stability_thresholds,
                        local_rank=runtime.local_rank,
                        aux_loss=native_moe_losses.aux_loss,
                        z_loss=native_moe_losses.z_loss,
                        router_layers=[
                            (layer.module_name, layer.tokens_per_expert)
                            for layer in router_snapshot.layers
                        ],
                    )
                    trend = trend_tracker.preview(
                        global_step=completed_step + 1,
                        lm_loss=lm_loss,
                        grad_norm=grad_norm,
                        max_activation_rms=activation_snapshot.max_rms,
                        max_activation_abs=activation_snapshot.max_abs,
                        worst_router_utilization_fraction=stability.worst_router_utilization_fraction,
                    )
                    local_blockers.extend(stability.blockers)
                    local_blockers.extend(trend.blockers)
                except Exception as exc:
                    local_exception = exc
                    local_blockers.append(f"rank_exception:{type(exc).__name__}")

                globally_unstable = distributed_any_unstable(bool(local_blockers), device=device)
                if globally_unstable:
                    reason_codes = local_blockers or ["remote_rank_instability"]
                    decision: RecoveryDecision = build_recovery_decision(
                        globally_unstable=True,
                        batch_fingerprint=batch_fingerprint,
                        current_learning_rate=current_lr,
                        policy=config.recovery_policy,
                        ledger=recovery_ledger,
                        reason_codes=reason_codes,
                    )
                    restored = _restore_last_good(
                        model,
                        optimizer,
                        checkpoint_dir=final_checkpoint,
                        global_seed=config.global_seed,
                    )
                    if restored is None:
                        raise RuntimeError(
                            "distributed recovery required but no last-good checkpoint exists"
                        ) from local_exception
                    state = restored
                    current_lr = decision.next_learning_rate
                    _set_optimizer_lr(optimizer, current_lr)
                    if decision.retry_allowed:
                        retry_attempts += 1
                        total_retries = recovery_ledger.total_retries
                        continue
                    if decision.quarantine_batch:
                        event = CatalogQuarantineEvent(
                            batch_fingerprint=batch_fingerprint,
                            global_sequence_offset=global_offset,
                            global_window=window,
                            failure_count=decision.failure_count,
                            skipped=decision.skip_batch,
                            reason_codes=decision.reason_codes,
                        )
                        quarantines.append(event)
                        if decision.skip_batch:
                            global_offset += window
                            cursor = FoundationCatalogCursor(
                                global_sequence_offset=global_offset,
                                catalog_sha256=catalog.catalog_sha256,
                                data_parallel_size=dp_size,
                            )
                            state = state.model_copy(
                                update={
                                    "data_state": cursor.model_dump(mode="json"),
                                    "learning_rate": current_lr,
                                }
                            )
                            state, final_checkpoint = _save_runtime_checkpoint(
                                model,
                                optimizer,
                                state,
                                checkpoint_dir=root / f"recovery-quarantine-{global_offset:016d}",
                                catalog_sha256=catalog.catalog_sha256,
                                current_learning_rate=current_lr,
                                ledger=recovery_ledger,
                                trend_tracker=trend_tracker,
                                total_retries=total_retries,
                                quarantine_events=quarantines,
                            )
                            step_committed = True
                            break
                    raise RuntimeError(
                        f"distributed recovery exhausted for batch {batch_fingerprint}: {decision.reason_codes}"
                    ) from local_exception

                if stability is None or trend is None:
                    raise RuntimeError("stable consensus reached without local telemetry")

                trend_tracker.commit_values(trend, grad_norm=grad_norm)
                elapsed = max(1.0e-9, time.perf_counter() - started)
                local_tokens = catalog.sequence_length * config.microbatches_per_step
                global_tokens = local_tokens * dp_size
                global_offset += window
                cursor = FoundationCatalogCursor(
                    global_sequence_offset=global_offset,
                    catalog_sha256=catalog.catalog_sha256,
                    data_parallel_size=dp_size,
                )
                completed_step += 1
                metrics.append(
                    CatalogTrainingStepMetrics(
                        global_step=completed_step,
                        local_tokens=local_tokens,
                        global_tokens=global_tokens,
                        elapsed_seconds=elapsed,
                        local_tokens_per_second=local_tokens / elapsed,
                        global_tokens_per_second=global_tokens / elapsed,
                        lm_loss=lm_loss,
                        stability=stability,
                        trend=trend,
                        recovery_attempts_before_success=retry_attempts,
                        effective_learning_rate=current_lr,
                    )
                )
                state = MCoreTrainingState(
                    global_step=completed_step,
                    consumed_microbatches=state.consumed_microbatches + config.microbatches_per_step,
                    learning_rate=current_lr,
                    global_seed=config.global_seed,
                    data_state=cursor.model_dump(mode="json"),
                )
                regular_checkpoint = (
                    state.global_step % config.checkpoint_every_steps == 0
                    or state.global_step == config.max_steps
                )
                checkpoint_dir = root / (
                    f"step-{state.global_step:08d}"
                    if regular_checkpoint
                    else f"recovery-step-{state.global_step:08d}"
                )
                state, final_checkpoint = _save_runtime_checkpoint(
                    model,
                    optimizer,
                    state,
                    checkpoint_dir=checkpoint_dir,
                    catalog_sha256=catalog.catalog_sha256,
                    current_learning_rate=current_lr,
                    ledger=recovery_ledger,
                    trend_tracker=trend_tracker,
                    total_retries=total_retries,
                    quarantine_events=quarantines,
                )
                step_committed = True
    finally:
        activation_probe.detach()
        router_probe.detach()

    return CatalogTrainingResult(
        start_step=start_step,
        final_step=config.max_steps,
        resumed_from_checkpoint=resumed,
        catalog_sha256=catalog.catalog_sha256,
        step_metrics=metrics,
        quarantine_events=quarantines,
        total_retries=total_retries,
        final_learning_rate=current_lr,
        final_checkpoint_dir=final_checkpoint,
    )
