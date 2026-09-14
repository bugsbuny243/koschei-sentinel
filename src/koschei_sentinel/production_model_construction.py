from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_megatron_model_spec import ProductionMegatronModelSpec
from koschei_sentinel.production_transformer_budget import estimate_transformer_budget

_GIB = 1024 ** 3


class ProductionConstructionPlan(StrictModel):
    schema_version: Literal["sentinel.production-model-construction.v1"] = (
        "sentinel.production-model-construction.v1"
    )
    status: Literal["dry_run_only"] = "dry_run_only"
    initialization: Literal["from_scratch"] = "from_scratch"
    parameter_dtype: Literal["bf16"] = "bf16"
    gradient_dtype: Literal["bf16"] = "bf16"
    optimizer: Literal["adamw"] = "adamw"
    optimizer_master_dtype: Literal["fp32"] = "fp32"
    distributed_optimizer: Literal[True] = True
    checkpoint_format: Literal["torch_dist"] = "torch_dist"
    checkpoint_shards: int = Field(gt=0)
    dense_parameter_shards: int = Field(gt=0)
    expert_parameter_shards: int = Field(gt=0)
    estimated_dense_weight_gib_per_rank: float = Field(gt=0)
    estimated_expert_weight_gib_per_rank: float = Field(gt=0)
    estimated_model_weight_gib_per_rank: float = Field(gt=0)
    estimated_optimizer_state_gib_per_rank_lower_bound: float = Field(gt=0)
    activation_memory_estimated: Literal[False] = False
    execution_authorized: Literal[False] = False
    blockers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def cannot_authorize_execution(self) -> "ProductionConstructionPlan":
        if self.execution_authorized is not False:
            raise ValueError("construction planning cannot authorize GPU execution")
        return self


def build_construction_plan(spec: ProductionMegatronModelSpec) -> ProductionConstructionPlan:
    budget = estimate_transformer_budget(spec.transformer)
    if not budget.within_target_tolerance:
        raise ValueError("transformer budget must remain within 397B/35B target tolerance")

    topo = spec.topology
    # Context parallelism partitions activations, not model weights. Dense weights are
    # therefore sharded by TP*PP, while routed experts are additionally sharded by EP.
    dense_shards = topo.tensor_model_parallel_size * topo.pipeline_model_parallel_size
    expert_shards = dense_shards * topo.expert_model_parallel_size

    dense_weight_bytes_per_rank = (
        budget.dense_shared_parameters_billion * 1e9 * 2 / dense_shards
    )
    expert_weight_bytes_per_rank = (
        budget.routed_expert_parameters_billion * 1e9 * 2 / expert_shards
    )
    weight_gib = (dense_weight_bytes_per_rank + expert_weight_bytes_per_rank) / _GIB

    # AdamW lower bound: fp32 master parameter + fp32 m + fp32 v = 12 bytes/parameter,
    # distributed across the data-parallel group. This intentionally excludes gradients,
    # activations, fragmentation, CUDA/NCCL workspaces, TE buffers and checkpoint staging.
    dp = topo.world_size // topo.model_parallel_size
    optimizer_bytes = (
        (budget.dense_shared_parameters_billion * 1e9 / dense_shards)
        + (budget.routed_expert_parameters_billion * 1e9 / expert_shards)
    ) * 12 / dp

    blockers = [
        "activation memory has not been measured on a real Megatron-Core graph",
        "CUDA/NCCL/Transformer Engine workspace memory has not been measured",
        "no 397B/35B checkpoint has been initialized on GPU",
        "no optimizer step has been executed on the target cluster",
    ]

    return ProductionConstructionPlan(
        checkpoint_shards=topo.world_size,
        dense_parameter_shards=dense_shards,
        expert_parameter_shards=expert_shards,
        estimated_dense_weight_gib_per_rank=round(dense_weight_bytes_per_rank / _GIB, 4),
        estimated_expert_weight_gib_per_rank=round(expert_weight_bytes_per_rank / _GIB, 4),
        estimated_model_weight_gib_per_rank=round(weight_gib, 4),
        estimated_optimizer_state_gib_per_rank_lower_bound=round(optimizer_bytes / _GIB, 4),
        blockers=blockers,
    )
