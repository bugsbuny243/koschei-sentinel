from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_megatron_model_spec import ProductionMegatronModelSpec
from koschei_sentinel.production_tensor_layout import TensorLayoutEntry, build_tensor_layout


class MetaFactoryResult(StrictModel):
    schema_version: Literal["sentinel.production-meta-factory-result.v1"] = (
        "sentinel.production-meta-factory-result.v1"
    )
    status: Literal["meta_only"] = "meta_only"
    tensor_count: int = Field(gt=0)
    local_elements: int = Field(gt=0)
    device: Literal["meta"] = "meta"
    dtype: Literal["bfloat16"] = "bfloat16"
    materialized_real_storage: Literal[False] = False
    execution_authorized: Literal[False] = False


def _owned_by_rank(
    entry: TensorLayoutEntry,
    *,
    pipeline_rank: int,
    tensor_rank: int,
    expert_rank: int,
) -> bool:
    if entry.pipeline_stage != pipeline_rank:
        return False
    if tensor_rank >= entry.tensor_parallel_shards:
        return False
    if expert_rank >= entry.expert_parallel_shards:
        return False
    return True


def instantiate_meta_rank(
    spec: ProductionMegatronModelSpec,
    *,
    pipeline_rank: int,
    tensor_rank: int,
    expert_rank: int,
) -> tuple[dict[str, Any], MetaFactoryResult]:
    topo = spec.topology
    if not 0 <= pipeline_rank < topo.pipeline_model_parallel_size:
        raise ValueError("pipeline_rank outside configured pipeline parallel size")
    if not 0 <= tensor_rank < topo.tensor_model_parallel_size:
        raise ValueError("tensor_rank outside configured tensor parallel size")
    if not 0 <= expert_rank < topo.expert_model_parallel_size:
        raise ValueError("expert_rank outside configured expert parallel size")

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependency path
        raise RuntimeError("PyTorch is required for meta-device construction") from exc

    layout = build_tensor_layout(spec)
    tensors: dict[str, Any] = {}
    local_elements = 0
    for entry in layout.entries:
        if not _owned_by_rank(
            entry,
            pipeline_rank=pipeline_rank,
            tensor_rank=tensor_rank,
            expert_rank=expert_rank,
        ):
            continue
        tensor = torch.empty(entry.shard_shape, device="meta", dtype=torch.bfloat16)
        tensors[entry.name] = tensor
        local_elements += tensor.numel()

    if not tensors:
        raise ValueError("rank owns no tensors; topology/layout mismatch")

    return tensors, MetaFactoryResult(
        tensor_count=len(tensors),
        local_elements=local_elements,
    )
