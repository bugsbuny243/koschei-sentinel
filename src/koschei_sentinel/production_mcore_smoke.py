from __future__ import annotations

from functools import partial
from typing import Literal

from pydantic import Field

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_mcore_distributed_runtime import MCoreDistributedRuntime
from koschei_sentinel.production_mcore_native_checkpoint import initialize_mcore_native_parameters
from koschei_sentinel.production_megatron_model_spec import ProductionMegatronModelSpec


class MCoreSmokeResult(StrictModel):
    schema_version: Literal["sentinel.mcore-smoke.v1"] = "sentinel.mcore-smoke.v1"
    status: Literal["forward_backward_verified"] = "forward_backward_verified"
    global_rank: int = Field(ge=0)
    world_size: int = Field(gt=0)
    micro_batch_size: Literal[1] = 1
    seq_length: int = Field(gt=0)
    num_microbatches: int = Field(gt=0)
    forward_verified: Literal[True] = True
    backward_verified: Literal[True] = True
    optimizer_step_verified: Literal[False] = False
    execution_authorized: Literal[False] = False


def run_mcore_forward_backward_smoke(
    runtime: MCoreDistributedRuntime,
    spec: ProductionMegatronModelSpec,
    *,
    seq_length: int = 64,
    global_seed: int = 39735,
    num_microbatches: int = 1,
    initialize_parameters: bool = True,
) -> MCoreSmokeResult:
    """Run synthetic microbatches through the actual MCore PP/TP/CP/EP graph.

    This verifies graph construction plus autograd. When ``initialize_parameters`` is false,
    existing model weights are preserved so the same path can be reused by the resumable
    training loop for gradient accumulation.
    """
    try:
        import torch
        from megatron.core import parallel_state
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
        from megatron.core.utils import get_batch_on_this_cp_rank
    except ImportError as exc:  # pragma: no cover - production dependency path
        raise RuntimeError("PyTorch and Megatron-Core are required for smoke execution") from exc

    cp_size = spec.topology.context_parallel_size
    if seq_length <= 0 or seq_length > spec.max_position_embeddings:
        raise ValueError("smoke seq_length outside model context window")
    if seq_length % (2 * cp_size):
        raise ValueError("smoke seq_length must be divisible by 2*context_parallel_size")
    if num_microbatches <= 0:
        raise ValueError("num_microbatches must be positive")

    model = runtime.model
    if initialize_parameters:
        initialize_mcore_native_parameters(model, global_seed=global_seed)
    model.train()

    device = torch.device("cuda", runtime.local_rank)

    def make_batch(microbatch_index: int):
        generator = torch.Generator(device=device)
        generator.manual_seed(global_seed + microbatch_index)
        tokens = torch.randint(
            low=0,
            high=spec.transformer.vocab_size,
            size=(1, seq_length),
            device=device,
            dtype=torch.long,
            generator=generator,
        )
        labels = tokens.roll(shifts=-1, dims=1)
        position_ids = torch.arange(seq_length, device=device, dtype=torch.long).unsqueeze(0)
        loss_mask = torch.ones((1, seq_length), device=device, dtype=torch.float32)
        batch = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "attention_mask": None,
            "cu_seqlens": None,
        }
        if cp_size > 1:
            batch = get_batch_on_this_cp_rank(
                batch,
                is_hybrid_cp=False,
                cp_group=parallel_state.get_context_parallel_group(),
            )
        return batch

    batches = [make_batch(index) for index in range(num_microbatches)]

    def loss_func(local_loss_mask, output_tensor):
        losses = output_tensor.float().view(-1)
        mask = local_loss_mask.float().view(-1)
        if losses.numel() != mask.numel():
            raise RuntimeError("smoke loss tensor and loss mask shape mismatch")
        denominator = mask.sum()
        if denominator.item() <= 0:
            raise RuntimeError("smoke loss mask is empty")
        loss = torch.sum(losses * mask) / denominator
        return loss, {"smoke_lm_loss": loss.detach()}

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

    schedule = get_forward_backward_func()
    schedule(
        forward_step_func=forward_step,
        data_iterator=iter(batches),
        model=model,
        num_microbatches=num_microbatches,
        seq_length=seq_length,
        micro_batch_size=1,
        forward_only=False,
        collect_non_loss_data=False,
    )

    def gradient_for(parameter):
        main_grad = getattr(parameter, "main_grad", None)
        return main_grad if main_grad is not None else parameter.grad

    has_grad = any(
        gradient_for(parameter) is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if not has_grad:
        raise RuntimeError("MCore smoke backward completed without any parameter gradient on this rank")

    return MCoreSmokeResult(
        global_rank=runtime.global_rank,
        world_size=runtime.world_size,
        seq_length=seq_length,
        num_microbatches=num_microbatches,
    )
