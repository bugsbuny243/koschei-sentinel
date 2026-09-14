from __future__ import annotations

from pathlib import Path

import pytest

from koschei_sentinel.production_checkpoint_load_verify import verify_rank_checkpoint_load
from koschei_sentinel.production_checkpoint_manifest import build_rank_checkpoint_manifest
from koschei_sentinel.production_megatron_model_spec import load_production_megatron_model_spec


CANDIDATE = Path("configs/training/production-megatron-model.397b-35b.candidate.json")


def test_rank_checkpoint_load_verifies_meta_shapes(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    spec = load_production_megatron_model_spec(CANDIDATE)
    manifest = build_rank_checkpoint_manifest(
        spec,
        pipeline_rank=0,
        tensor_rank=0,
        expert_rank=0,
    )
    state_dict = {
        item.name: torch.empty(item.shard_shape, dtype=torch.bfloat16)
        for item in manifest.tensors
    }
    path = tmp_path / "rank.pt"
    torch.save(
        {
            "schema_version": "sentinel.rank-weight-checkpoint.v1",
            "manifest_sha256": manifest.manifest_sha256,
            "pipeline_rank": 0,
            "tensor_rank": 0,
            "expert_rank": 0,
            "dtype": "bfloat16",
            "state_dict": state_dict,
        },
        path,
    )
    result = verify_rank_checkpoint_load(manifest, checkpoint_path=path)
    assert result.status == "metadata_verified"
    assert result.map_location == "meta"
    assert result.real_storage_allocated is False
    assert result.forward_verified is False
    assert result.backward_verified is False


def test_rank_checkpoint_load_rejects_manifest_mismatch(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    spec = load_production_megatron_model_spec(CANDIDATE)
    manifest = build_rank_checkpoint_manifest(
        spec,
        pipeline_rank=3,
        tensor_rank=7,
        expert_rank=15,
    )
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "schema_version": "sentinel.rank-weight-checkpoint.v1",
            "manifest_sha256": "0" * 64,
            "pipeline_rank": 3,
            "tensor_rank": 7,
            "expert_rank": 15,
            "dtype": "bfloat16",
            "state_dict": {},
        },
        path,
    )
    with pytest.raises(ValueError, match="manifest digest mismatch"):
        verify_rank_checkpoint_load(manifest, checkpoint_path=path)
