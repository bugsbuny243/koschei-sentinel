from pathlib import Path

from koschei_sentinel.production_megatron_model_spec import load_production_megatron_model_spec
from koschei_sentinel.production_megatron_runtime_adapter import build_megatron_core_factory_plan


CANDIDATE = Path("configs/training/production-megatron-model.397b-35b.candidate.json")


def test_factory_plan_preserves_production_architecture() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    plan = build_megatron_core_factory_plan(spec)
    cfg = plan.transformer_config_kwargs
    gpt = plan.gpt_model_kwargs

    assert cfg["num_layers"] == 60
    assert cfg["hidden_size"] == 8192
    assert cfg["num_attention_heads"] == 128
    assert cfg["num_query_groups"] == 16
    assert cfg["kv_channels"] == 64
    assert cfg["num_moe_experts"] == 128
    assert cfg["moe_router_topk"] == 8
    assert cfg["moe_ffn_hidden_size"] == 2048
    assert cfg["moe_shared_expert_intermediate_size"] == 384
    assert cfg["tensor_model_parallel_size"] == 8
    assert cfg["pipeline_model_parallel_size"] == 4
    assert cfg["context_parallel_size"] == 2
    assert cfg["expert_model_parallel_size"] == 16
    assert cfg["sequence_parallel"] is True
    assert cfg["bf16"] is True

    assert gpt["vocab_size"] == 151936
    assert gpt["max_sequence_length"] == 32768
    assert gpt["position_embedding_type"] == "rope"
    assert gpt["rotary_percent"] == 1.0
    assert gpt["rotary_base"] == 1_000_000
    assert gpt["share_embeddings_and_output_weights"] is True


def test_factory_plan_never_claims_runtime_success() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    plan = build_megatron_core_factory_plan(spec, transformer_impl="local")
    assert plan.model_instantiated is False
    assert plan.checkpoint_loaded is False
    assert plan.forward_verified is False
    assert plan.backward_verified is False
    assert plan.execution_authorized is False
