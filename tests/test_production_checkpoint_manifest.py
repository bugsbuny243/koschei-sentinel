from __future__ import annotations

from pathlib import Path

import pytest

from koschei_sentinel.production_checkpoint_manifest import (
    RankCheckpointManifest,
    build_rank_checkpoint_manifest,
)
from koschei_sentinel.production_megatron_model_spec import load_production_megatron_model_spec
from koschei_sentinel.production_meta_factory import instantiate_meta_rank


CANDIDATE = Path("configs/training/production-megatron-model.397b-35b.candidate.json")


def test_dense_tensors_exist_on_nonzero_expert_rank() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    tensors, result = instantiate_meta_rank(
        spec,
        pipeline_rank=0,
        tensor_rank=3,
        expert_rank=7,
    )
    assert result.tensor_count > 0
    assert "embedding.word_embeddings.weight" in tensors
    assert "decoder.layers.0.self_attention.q_proj.weight" in tensors
    assert "decoder.layers.0.input_layernorm.weight" in tensors
    assert "decoder.layers.0.moe.experts.gate_up.weight" in tensors


def test_replicated_norm_and_router_exist_on_nonzero_tp_ep_rank() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    tensors, _ = instantiate_meta_rank(
        spec,
        pipeline_rank=2,
        tensor_rank=7,
        expert_rank=15,
    )
    assert "decoder.layers.30.input_layernorm.weight" in tensors
    assert "decoder.layers.30.moe.router.weight" in tensors


def test_rank_checkpoint_manifest_is_deterministic() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    first = build_rank_checkpoint_manifest(
        spec,
        pipeline_rank=1,
        tensor_rank=4,
        expert_rank=9,
        global_seed=39735,
    )
    second = build_rank_checkpoint_manifest(
        spec,
        pipeline_rank=1,
        tensor_rank=4,
        expert_rank=9,
        global_seed=39735,
    )
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.tensors[0].recipe_sha256 == second.tensors[0].recipe_sha256
    assert first.materialized_weights is False
    assert first.execution_authorized is False


def test_tensor_parallel_shard_changes_dense_seed_but_ep_replica_does_not() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    base = build_rank_checkpoint_manifest(
        spec,
        pipeline_rank=0,
        tensor_rank=0,
        expert_rank=0,
    )
    other_tp = build_rank_checkpoint_manifest(
        spec,
        pipeline_rank=0,
        tensor_rank=1,
        expert_rank=0,
    )
    other_ep = build_rank_checkpoint_manifest(
        spec,
        pipeline_rank=0,
        tensor_rank=0,
        expert_rank=7,
    )
    get_q = lambda manifest: next(
        item for item in manifest.tensors if item.name.endswith("self_attention.q_proj.weight")
    )
    assert get_q(base).seed != get_q(other_tp).seed
    assert get_q(base).seed == get_q(other_ep).seed


def test_replicated_router_seed_is_identical_across_tp_and_ep() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    left = build_rank_checkpoint_manifest(
        spec,
        pipeline_rank=2,
        tensor_rank=0,
        expert_rank=0,
    )
    right = build_rank_checkpoint_manifest(
        spec,
        pipeline_rank=2,
        tensor_rank=7,
        expert_rank=15,
    )
    get_router = lambda manifest: next(
        item for item in manifest.tensors if item.name.endswith("moe.router.weight")
    )
    assert get_router(left).seed == get_router(right).seed


def test_expert_shard_seed_changes_across_ep() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    left = build_rank_checkpoint_manifest(
        spec,
        pipeline_rank=1,
        tensor_rank=4,
        expert_rank=2,
    )
    right = build_rank_checkpoint_manifest(
        spec,
        pipeline_rank=1,
        tensor_rank=4,
        expert_rank=3,
    )
    get_expert = lambda manifest: next(
        item for item in manifest.tensors if item.name.endswith("moe.experts.gate_up.weight")
    )
    assert get_expert(left).seed != get_expert(right).seed


def test_manifest_digest_rejects_mutation() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    manifest = build_rank_checkpoint_manifest(
        spec,
        pipeline_rank=3,
        tensor_rank=7,
        expert_rank=15,
    )
    payload = manifest.model_dump(mode="json")
    payload["global_seed"] += 1
    with pytest.raises(ValueError, match="manifest digest mismatch"):
        RankCheckpointManifest.model_validate(payload)
