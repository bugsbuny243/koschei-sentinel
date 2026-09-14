from __future__ import annotations

from pathlib import Path

import pytest

from koschei_sentinel.production_global_checkpoint import (
    GlobalCheckpointIndex,
    build_global_checkpoint_index,
)
from koschei_sentinel.production_megatron_model_spec import load_production_megatron_model_spec


CANDIDATE = Path("configs/training/production-megatron-model.397b-35b.candidate.json")


def test_global_index_has_512_unique_model_shards() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    index = build_global_checkpoint_index(spec)
    assert index.unique_model_shard_count == 4 * 8 * 16 == 512
    assert index.process_world_size == 1024
    assert len(index.shards) == 512
    assert len({(s.pipeline_rank, s.tensor_rank, s.expert_rank) for s in index.shards}) == 512
    assert index.fully_materialized is False
    assert index.loadable_checkpoint is False
    assert index.execution_authorized is False


def test_global_index_paths_are_deterministic_and_unique() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    left = build_global_checkpoint_index(spec)
    right = build_global_checkpoint_index(spec)
    assert left.index_sha256 == right.index_sha256
    paths = [s.relative_path for s in left.shards]
    assert len(paths) == len(set(paths)) == 512
    assert paths[0] == "model-shards/pp-00/tp-00-ep-00.pt"
    assert paths[-1] == "model-shards/pp-03/tp-07-ep-15.pt"


def test_global_index_digest_rejects_mutation() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    index = build_global_checkpoint_index(spec)
    payload = index.model_dump(mode="json")
    payload["shards"][0]["local_elements"] += 1
    with pytest.raises(ValueError, match="index digest mismatch"):
        GlobalCheckpointIndex.model_validate(payload)


def test_global_index_does_not_confuse_world_processes_with_weight_shards() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    index = build_global_checkpoint_index(spec)
    assert index.process_world_size == 1024
    assert index.unique_model_shard_count == 512
    assert index.context_parallel_size == 2
    assert index.data_parallel_size == 16
