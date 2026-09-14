from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from koschei_sentinel.production_megatron_model_spec import ProductionMegatronModelSpec
from koschei_sentinel.production_megatron_runtime_adapter import instantiate_megatron_core_model


@dataclass(frozen=True)
class MCoreDistributedRuntime:
    local_rank: int
    global_rank: int
    world_size: int
    data_parallel_size: int
    model: Any


def initialize_mcore_distributed_runtime(spec: ProductionMegatronModelSpec) -> MCoreDistributedRuntime:
    """Bind an already torchrun-initialized 1024-rank job to the Sentinel MCore graph.

    This function does not call torch.distributed.init_process_group; torchrun/the cluster
    launcher owns that lifecycle. It only creates Megatron model-parallel groups and then
    constructs the native GPT/MoE graph on the current CUDA rank.
    """
    try:
        import torch
        import torch.distributed as dist
        from megatron.core import parallel_state
    except ImportError as exc:  # pragma: no cover - production dependency path
        raise RuntimeError("PyTorch distributed and Megatron-Core are required") from exc

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized by torchrun before MCore bootstrap")
    if not torch.cuda.is_available():
        raise RuntimeError("production MCore runtime requires CUDA")

    topo = spec.topology
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    if world_size != topo.world_size:
        raise RuntimeError(
            f"distributed world size mismatch: expected {topo.world_size}, observed {world_size}"
        )

    local_rank = int(os.environ.get("LOCAL_RANK", global_rank % torch.cuda.device_count()))
    if not 0 <= local_rank < torch.cuda.device_count():
        raise RuntimeError("LOCAL_RANK is outside visible CUDA devices")
    torch.cuda.set_device(local_rank)

    model_parallel_size = (
        topo.tensor_model_parallel_size
        * topo.pipeline_model_parallel_size
        * topo.context_parallel_size
    )
    if world_size % model_parallel_size:
        raise RuntimeError("world size is not divisible by TP*PP*CP")
    data_parallel_size = world_size // model_parallel_size
    if data_parallel_size % topo.expert_model_parallel_size:
        raise RuntimeError("data parallel size must be divisible by expert parallel size")

    if parallel_state.model_parallel_is_initialized():
        raise RuntimeError("Megatron model-parallel state is already initialized")

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=topo.tensor_model_parallel_size,
        pipeline_model_parallel_size=topo.pipeline_model_parallel_size,
        context_parallel_size=topo.context_parallel_size,
        expert_model_parallel_size=topo.expert_model_parallel_size,
        expert_tensor_parallel_size=topo.tensor_model_parallel_size,
        order="tp-cp-ep-dp-pp",
        local_world_size=torch.cuda.device_count(),
    )

    runtime_objects = instantiate_megatron_core_model(spec, transformer_impl="transformer_engine")
    model = runtime_objects.model.cuda(local_rank)

    return MCoreDistributedRuntime(
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
        data_parallel_size=data_parallel_size,
        model=model,
    )
