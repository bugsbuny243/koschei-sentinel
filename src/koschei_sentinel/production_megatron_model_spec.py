from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_megatron_config import ProductionMegatronTopology
from koschei_sentinel.production_transformer_budget import (
    TransformerBudgetSpec,
    estimate_transformer_budget,
)


class ProductionMegatronModelSpec(StrictModel):
    schema_version: Literal["sentinel.production-megatron-model-spec.v1"] = (
        "sentinel.production-megatron-model-spec.v1"
    )
    status: Literal["research_candidate"] = "research_candidate"

    transformer: TransformerBudgetSpec

    num_attention_heads: int = Field(gt=0)
    num_query_groups: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    normalization: Literal["RMSNorm"] = "RMSNorm"
    norm_epsilon: float = Field(gt=0.0, le=1e-3)
    position_embedding: Literal["rope"] = "rope"
    rotary_percent: Literal[1.0] = 1.0
    rotary_base: float = Field(gt=0.0)
    max_position_embeddings: int = Field(ge=8192)

    moe_router_topk: int = Field(gt=0)
    moe_router_score_function: Literal["softmax"] = "softmax"
    moe_router_pre_softmax: bool = False
    moe_aux_loss_coeff: float = Field(ge=0.0, le=1.0)
    moe_z_loss_coeff: float = Field(ge=0.0, le=1.0)
    moe_grouped_gemm: Literal[True] = True
    moe_token_dispatcher_type: Literal["alltoall"] = "alltoall"
    moe_expert_capacity_factor: float | None = Field(default=None, gt=0.0)
    moe_pad_expert_input_to_capacity: bool = False

    recompute_granularity: Literal["full"] = "full"
    recompute_method: Literal["uniform"] = "uniform"
    recompute_num_layers: int = Field(gt=0)

    topology: ProductionMegatronTopology

    @model_validator(mode="after")
    def validate_megatron_shape(self) -> "ProductionMegatronModelSpec":
        t = self.transformer
        tp = self.topology.tensor_model_parallel_size
        pp = self.topology.pipeline_model_parallel_size
        cp = self.topology.context_parallel_size
        ep = self.topology.expert_model_parallel_size

        if self.num_attention_heads * self.head_dim != t.hidden_size:
            raise ValueError("attention heads * head_dim must equal hidden_size")
        if self.num_attention_heads % self.num_query_groups:
            raise ValueError("num_attention_heads must be divisible by num_query_groups")
        if self.num_attention_heads % tp:
            raise ValueError("num_attention_heads must be divisible by tensor parallel size")
        if self.num_query_groups % tp:
            raise ValueError("num_query_groups must be divisible by tensor parallel size")

        expected_kv_ratio = self.num_query_groups / self.num_attention_heads
        if abs(expected_kv_ratio - t.attention_kv_ratio) > 1e-12:
            raise ValueError("GQA query-group ratio disagrees with transformer budget")

        if self.moe_router_topk != t.experts_per_token:
            raise ValueError("Megatron router top-k must match transformer experts_per_token")
        if t.num_experts % ep:
            raise ValueError("num_experts must be divisible by expert parallel size")
        if t.num_layers % pp:
            raise ValueError("num_layers must be divisible by pipeline parallel size")
        if self.max_position_embeddings % cp:
            raise ValueError("max_position_embeddings must be divisible by context parallel size")
        if self.recompute_num_layers > t.num_layers // pp:
            raise ValueError("recompute_num_layers cannot exceed local pipeline-stage depth")

        budget = estimate_transformer_budget(t)
        if not budget.within_target_tolerance:
            raise ValueError("transformer budget must remain within 397B/35B target tolerance")

        return self

    @property
    def layers_per_pipeline_stage(self) -> int:
        return self.transformer.num_layers // self.topology.pipeline_model_parallel_size

    @property
    def experts_per_ep_rank(self) -> int:
        return self.transformer.num_experts // self.topology.expert_model_parallel_size


def load_production_megatron_model_spec(path: str | Path) -> ProductionMegatronModelSpec:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid production Megatron model spec: {source}") from exc
    return ProductionMegatronModelSpec.model_validate(payload)
