from __future__ import annotations

from pathlib import Path

import pytest

from koschei_sentinel.production_global_checkpoint import GlobalCheckpointIndex
from koschei_sentinel.production_global_checkpoint_materialization import (
    GlobalCheckpointMaterializationReceipt,
    finalize_global_checkpoint_materialization,
)


def _tiny_index(tmp_path: Path) -> GlobalCheckpointIndex:
    shard_payload = {
        "model_shard_rank": 0,
        "pipeline_rank": 0,
        "tensor_rank": 0,
        "expert_rank": 0,
        "tensor_count": 1,
        "local_elements": 1,
        "initialization_manifest_sha256": "a" * 64,
        "relative_path": "model-shards/pp-00/tp-00-ep-00.pt",
        "materialized_file_sha256": None,
    }
    payload = {
        "schema_version": "sentinel.global-checkpoint-index.v1",
        "status": "initialization_index",
        "checkpoint_format": "torch_dist",
        "global_seed": 39735,
        "pipeline_parallel_size": 1,
        "tensor_parallel_size": 1,
        "expert_parallel_size": 1,
        "unique_model_shard_count": 1,
        "process_world_size": 1,
        "context_parallel_size": 1,
        "data_parallel_size": 1,
        "shards": [shard_payload],
        "fully_materialized": False,
        "loadable_checkpoint": False,
        "execution_authorized": False,
    }
    import hashlib, json
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload["index_sha256"] = hashlib.sha256(encoded).hexdigest()
    return GlobalCheckpointIndex.model_validate(payload)


def test_finalize_hashes_materialized_shard(tmp_path: Path) -> None:
    index = _tiny_index(tmp_path)
    shard = tmp_path / "model-shards/pp-00/tp-00-ep-00.pt"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"sentinel-checkpoint-shard")
    receipt = finalize_global_checkpoint_materialization(index, checkpoint_root=tmp_path)
    assert receipt.shard_count == 1
    assert receipt.fully_materialized is True
    assert receipt.megatron_load_verified is False
    assert receipt.execution_authorized is False
    assert receipt.shards[0].file_size_bytes == len(b"sentinel-checkpoint-shard")


def test_finalize_rejects_missing_shard(tmp_path: Path) -> None:
    index = _tiny_index(tmp_path)
    with pytest.raises(ValueError, match="missing checkpoint shard"):
        finalize_global_checkpoint_materialization(index, checkpoint_root=tmp_path)


def test_materialization_receipt_digest_rejects_mutation(tmp_path: Path) -> None:
    index = _tiny_index(tmp_path)
    shard = tmp_path / "model-shards/pp-00/tp-00-ep-00.pt"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"sentinel-checkpoint-shard")
    receipt = finalize_global_checkpoint_materialization(index, checkpoint_root=tmp_path)
    payload = receipt.model_dump(mode="json")
    payload["shards"][0]["file_size_bytes"] += 1
    with pytest.raises(ValueError, match="receipt digest mismatch"):
        GlobalCheckpointMaterializationReceipt.model_validate(payload)
