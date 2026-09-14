from __future__ import annotations

from pathlib import Path

import pytest

from koschei_sentinel.production_checkpoint_manifest import build_rank_checkpoint_manifest
from koschei_sentinel.production_megatron_model_spec import load_production_megatron_model_spec
from koschei_sentinel.production_weight_materializer import (
    materialize_initialization_record,
    materialize_rank_weights,
)


CANDIDATE = Path("configs/training/production-megatron-model.397b-35b.candidate.json")


def test_normalization_materializes_to_bf16_ones_on_cpu() -> None:
    torch = pytest.importorskip("torch")
    spec = load_production_megatron_model_spec(CANDIDATE)
    manifest = build_rank_checkpoint_manifest(
        spec,
        pipeline_rank=0,
        tensor_rank=7,
        expert_rank=15,
    )
    record = next(item for item in manifest.tensors if item.category == "normalization")
    tensor = materialize_initialization_record(record, device="cpu")
    assert tensor.dtype == torch.bfloat16
    assert tuple(tensor.shape) == record.shard_shape
    assert torch.all(tensor == 1)


def test_normal_initialization_is_deterministic_for_same_recipe() -> None:
    torch = pytest.importorskip("torch")
    spec = load_production_megatron_model_spec(CANDIDATE)
    manifest = build_rank_checkpoint_manifest(
        spec,
        pipeline_rank=0,
        tensor_rank=7,
        expert_rank=15,
    )
    record = next(item for item in manifest.tensors if item.category == "shared_mlp")
    first = materialize_initialization_record(record, device="cpu")
    second = materialize_initialization_record(record, device="cpu")
    assert torch.equal(first, second)


def test_full_rank_materialization_respects_memory_cap_before_allocation() -> None:
    pytest.importorskip("torch")
    spec = load_production_megatron_model_spec(CANDIDATE)
    manifest = build_rank_checkpoint_manifest(
        spec,
        pipeline_rank=1,
        tensor_rank=4,
        expert_rank=9,
    )
    with pytest.raises(RuntimeError, match="exceeds materialization cap"):
        materialize_rank_weights(
            manifest,
            device="cpu",
            max_weight_gib=0.001,
        )


def test_production_write_path_can_require_cuda() -> None:
    pytest.importorskip("torch")
    spec = load_production_megatron_model_spec(CANDIDATE)
    manifest = build_rank_checkpoint_manifest(
        spec,
        pipeline_rank=3,
        tensor_rank=0,
        expert_rank=0,
    )
    with pytest.raises(ValueError, match="requires a CUDA device"):
        materialize_rank_weights(
            manifest,
            device="cpu",
            require_cuda=True,
        )
