from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_megatron_model_spec import ProductionMegatronModelSpec
from koschei_sentinel.production_transformer_budget import estimate_transformer_budget


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ProductionMegatronLaunchPlan(StrictModel):
    schema_version: Literal["sentinel.production-megatron-launch-plan.v1"] = (
        "sentinel.production-megatron-launch-plan.v1"
    )
    status: Literal["dry_run_only"] = "dry_run_only"
    trainer: Literal["megatron-core"] = "megatron-core"
    world_size: int = Field(gt=0)
    model_parallel_size: int = Field(gt=0)
    data_parallel_size: int = Field(gt=0)
    layers_per_pipeline_stage: int = Field(gt=0)
    experts_per_ep_rank: int = Field(gt=0)
    estimated_total_parameters_billion: float
    estimated_active_parameters_billion: float
    argv: list[str] = Field(min_length=1)
    blockers: list[str] = Field(default_factory=list)
    launchable: Literal[False] = False
    plan_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def no_accidental_execution_claim(self) -> "ProductionMegatronLaunchPlan":
        if self.status != "dry_run_only" or self.launchable is not False:
            raise ValueError("research launch plan cannot authorize execution")
        return self


def compile_production_megatron_launch_plan(
    spec: ProductionMegatronModelSpec,
) -> ProductionMegatronLaunchPlan:
    t = spec.transformer
    topo = spec.topology
    budget = estimate_transformer_budget(t)

    blockers: list[str] = []
    if not budget.within_target_tolerance:
        blockers.extend(budget.blockers)
    if spec.status != "research_candidate":
        blockers.append("only a research_candidate model spec may enter dry-run compilation")

    world_size = topo.world_size
    model_parallel = topo.model_parallel_size
    data_parallel = world_size // model_parallel

    argv = [
        "--num-layers", str(t.num_layers),
        "--hidden-size", str(t.hidden_size),
        "--num-attention-heads", str(spec.num_attention_heads),
        "--group-query-attention",
        "--num-query-groups", str(spec.num_query_groups),
        "--ffn-hidden-size", str(t.shared_intermediate_size),
        "--num-experts", str(t.num_experts),
        "--moe-router-topk", str(spec.moe_router_topk),
        "--moe-ffn-hidden-size", str(t.expert_intermediate_size),
        "--tensor-model-parallel-size", str(topo.tensor_model_parallel_size),
        "--pipeline-model-parallel-size", str(topo.pipeline_model_parallel_size),
        "--context-parallel-size", str(topo.context_parallel_size),
        "--expert-model-parallel-size", str(topo.expert_model_parallel_size),
        "--sequence-parallel",
        "--max-position-embeddings", str(spec.max_position_embeddings),
        "--position-embedding-type", "rope",
        "--rotary-percent", str(spec.rotary_percent),
        "--rotary-base", str(spec.rotary_base),
        "--normalization", spec.normalization,
        "--norm-epsilon", str(spec.norm_epsilon),
        "--moe-token-dispatcher-type", spec.moe_token_dispatcher_type,
        "--moe-grouped-gemm",
        "--recompute-granularity", spec.recompute_granularity,
        "--recompute-method", spec.recompute_method,
        "--recompute-num-layers", str(spec.recompute_num_layers),
    ]
    if spec.moe_router_pre_softmax:
        argv.append("--moe-router-pre-softmax")
    if spec.moe_aux_loss_coeff:
        argv.extend(["--moe-aux-loss-coeff", str(spec.moe_aux_loss_coeff)])
    if spec.moe_z_loss_coeff:
        argv.extend(["--moe-z-loss-coeff", str(spec.moe_z_loss_coeff)])
    if spec.moe_expert_capacity_factor is not None:
        argv.extend(["--moe-expert-capacity-factor", str(spec.moe_expert_capacity_factor)])
    if spec.moe_pad_expert_input_to_capacity:
        argv.append("--moe-pad-expert-input-to-capacity")

    payload = {
        "schema_version": "sentinel.production-megatron-launch-plan.v1",
        "status": "dry_run_only",
        "trainer": "megatron-core",
        "world_size": world_size,
        "model_parallel_size": model_parallel,
        "data_parallel_size": data_parallel,
        "layers_per_pipeline_stage": spec.layers_per_pipeline_stage,
        "experts_per_ep_rank": spec.experts_per_ep_rank,
        "estimated_total_parameters_billion": budget.total_parameters_billion,
        "estimated_active_parameters_billion": budget.active_parameters_billion,
        "argv": argv,
        "blockers": blockers,
        "launchable": False,
    }
    return ProductionMegatronLaunchPlan.model_validate(
        {**payload, "plan_sha256": _digest(payload)}
    )
