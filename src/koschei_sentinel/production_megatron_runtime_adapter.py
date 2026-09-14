from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_megatron_model_spec import ProductionMegatronModelSpec


class MegatronCoreFactoryPlan(StrictModel):
    schema_version: Literal["sentinel.megatron-core-factory-plan.v1"] = (
        "sentinel.megatron-core-factory-plan.v1"
    )
    status: Literal["runtime_adapter"] = "runtime_adapter"
    transformer_config_kwargs: dict[str, Any]
    gpt_model_kwargs: dict[str, Any]
    required_transformer_fields: list[str] = Field(min_length=1)
    required_gpt_fields: list[str] = Field(min_length=1)
    transformer_impl: Literal["transformer_engine", "local"]
    model_instantiated: Literal[False] = False
    checkpoint_loaded: Literal[False] = False
    forward_verified: Literal[False] = False
    backward_verified: Literal[False] = False
    execution_authorized: Literal[False] = False


def build_megatron_core_factory_plan(
    spec: ProductionMegatronModelSpec,
    *,
    transformer_impl: Literal["transformer_engine", "local"] = "transformer_engine",
) -> MegatronCoreFactoryPlan:
    t = spec.transformer
    topo = spec.topology

    transformer = {
        "num_layers": t.num_layers,
        "hidden_size": t.hidden_size,
        "num_attention_heads": spec.num_attention_heads,
        "num_query_groups": spec.num_query_groups,
        "kv_channels": spec.head_dim,
        "ffn_hidden_size": t.shared_intermediate_size,
        "layernorm_epsilon": spec.norm_epsilon,
        "tensor_model_parallel_size": topo.tensor_model_parallel_size,
        "pipeline_model_parallel_size": topo.pipeline_model_parallel_size,
        "context_parallel_size": topo.context_parallel_size,
        "expert_model_parallel_size": topo.expert_model_parallel_size,
        "sequence_parallel": topo.sequence_parallel,
        "num_moe_experts": t.num_experts,
        "moe_router_topk": spec.moe_router_topk,
        "moe_ffn_hidden_size": t.expert_intermediate_size,
        "moe_shared_expert_intermediate_size": t.shared_intermediate_size,
        "moe_router_load_balancing_type": "aux_loss" if spec.moe_aux_loss_coeff else "none",
        "moe_aux_loss_coeff": spec.moe_aux_loss_coeff,
        "moe_z_loss_coeff": spec.moe_z_loss_coeff,
        "moe_token_dispatcher_type": spec.moe_token_dispatcher_type,
        "moe_grouped_gemm": spec.moe_grouped_gemm,
        "recompute_granularity": spec.recompute_granularity,
        "recompute_method": spec.recompute_method,
        "recompute_num_layers": spec.recompute_num_layers,
        "bf16": True,
        "transformer_impl": transformer_impl,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
    }

    gpt = {
        "vocab_size": t.vocab_size,
        "max_sequence_length": spec.max_position_embeddings,
        "pre_process": True,
        "post_process": True,
        "fp16_lm_cross_entropy": False,
        "parallel_output": True,
        "share_embeddings_and_output_weights": t.tied_embeddings,
        "position_embedding_type": spec.position_embedding,
        "rotary_percent": spec.rotary_percent,
        "rotary_base": int(spec.rotary_base),
    }

    return MegatronCoreFactoryPlan(
        transformer_config_kwargs=transformer,
        gpt_model_kwargs=gpt,
        required_transformer_fields=[
            "num_layers",
            "hidden_size",
            "num_attention_heads",
            "num_query_groups",
            "kv_channels",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "expert_model_parallel_size",
            "num_moe_experts",
            "moe_router_topk",
            "moe_ffn_hidden_size",
            "moe_shared_expert_intermediate_size",
        ],
        required_gpt_fields=[
            "vocab_size",
            "max_sequence_length",
            "position_embedding_type",
            "rotary_percent",
            "rotary_base",
        ],
        transformer_impl=transformer_impl,
    )


@dataclass(frozen=True)
class MegatronRuntimeObjects:
    config: Any
    layer_spec: Any
    model: Any


def _supported_kwargs(callable_obj: Any, payload: dict[str, Any], required: list[str]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    parameters = signature.parameters
    accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    if accepts_var_kw:
        return dict(payload)
    missing = [name for name in required if name not in parameters]
    if missing:
        raise RuntimeError(f"installed Megatron-Core API lacks required fields: {missing}")
    return {name: value for name, value in payload.items() if name in parameters}


def instantiate_megatron_core_model(
    spec: ProductionMegatronModelSpec,
    *,
    transformer_impl: Literal["transformer_engine", "local"] = "transformer_engine",
) -> MegatronRuntimeObjects:
    """Instantiate the actual Megatron-Core GPT graph for the production spec.

    The caller must initialize torch.distributed and Megatron model-parallel groups first.
    Pipeline-stage ownership is derived from Megatron parallel_state at runtime.
    """
    try:
        from megatron.core import parallel_state
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_local_spec,
            get_gpt_layer_with_transformer_engine_spec,
        )
        from megatron.core.models.gpt.gpt_model import GPTModel
        from megatron.core.transformer.transformer_config import TransformerConfig
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("Megatron-Core is required for production model construction") from exc

    if not parallel_state.model_parallel_is_initialized():
        raise RuntimeError("Megatron model-parallel groups must be initialized before GPT construction")

    plan = build_megatron_core_factory_plan(spec, transformer_impl=transformer_impl)
    config_kwargs = _supported_kwargs(
        TransformerConfig,
        plan.transformer_config_kwargs,
        plan.required_transformer_fields,
    )
    config = TransformerConfig(**config_kwargs)

    layer_factory = (
        get_gpt_layer_with_transformer_engine_spec
        if transformer_impl == "transformer_engine"
        else get_gpt_layer_local_spec
    )
    layer_signature = inspect.signature(layer_factory)
    layer_kwargs: dict[str, Any] = {}
    if "num_experts" in layer_signature.parameters:
        layer_kwargs["num_experts"] = spec.transformer.num_experts
    if "moe_grouped_gemm" in layer_signature.parameters:
        layer_kwargs["moe_grouped_gemm"] = spec.moe_grouped_gemm
    layer_spec = layer_factory(**layer_kwargs)

    stage_kwargs = dict(plan.gpt_model_kwargs)
    stage_kwargs["pre_process"] = parallel_state.is_pipeline_first_stage()
    stage_kwargs["post_process"] = parallel_state.is_pipeline_last_stage()
    model_kwargs = _supported_kwargs(GPTModel, stage_kwargs, plan.required_gpt_fields)
    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        **model_kwargs,
    )
    return MegatronRuntimeObjects(config=config, layer_spec=layer_spec, model=model)
