from pathlib import Path

from koschei_sentinel.production_megatron_model_spec import load_production_megatron_model_spec
from koschei_sentinel.production_tensor_layout import build_tensor_layout


CANDIDATE = Path("configs/training/production-megatron-model.397b-35b.candidate.json")


def test_tensor_layout_emits_expected_count_and_pipeline_placement() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    plan = build_tensor_layout(spec)
    assert plan.tensor_count == 662
    assert plan.entries[0].name == "embedding.word_embeddings.weight"
    assert plan.entries[0].pipeline_stage == 0
    assert plan.entries[-1].name == "decoder.final_layernorm.weight"
    assert plan.entries[-1].pipeline_stage == 3
    assert plan.execution_authorized is False


def test_attention_and_expert_shards_match_tp_ep_shape() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    plan = build_tensor_layout(spec)
    by_name = {entry.name: entry for entry in plan.entries}

    q = by_name["decoder.layers.0.self_attention.q_proj.weight"]
    assert q.global_shape == (8192, 8192)
    assert q.shard_shape == (1024, 8192)
    assert q.tensor_parallel_shards == 8

    k = by_name["decoder.layers.0.self_attention.k_proj.weight"]
    assert k.global_shape == (1024, 8192)
    assert k.shard_shape == (128, 8192)

    gate_up = by_name["decoder.layers.0.moe.experts.gate_up.weight"]
    assert gate_up.global_shape == (128, 4096, 8192)
    assert gate_up.shard_shape == (8, 512, 8192)
    assert gate_up.tensor_parallel_shards == 8
    assert gate_up.expert_parallel_shards == 16


def test_pipeline_stage_changes_every_fifteen_layers() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    plan = build_tensor_layout(spec)
    by_name = {entry.name: entry for entry in plan.entries}
    assert by_name["decoder.layers.14.self_attention.q_proj.weight"].pipeline_stage == 0
    assert by_name["decoder.layers.15.self_attention.q_proj.weight"].pipeline_stage == 1
    assert by_name["decoder.layers.30.self_attention.q_proj.weight"].pipeline_stage == 2
    assert by_name["decoder.layers.45.self_attention.q_proj.weight"].pipeline_stage == 3
