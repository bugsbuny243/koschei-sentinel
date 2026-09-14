from pathlib import Path

import pytest

from koschei_sentinel.production_megatron_model_spec import load_production_megatron_model_spec
from koschei_sentinel.production_meta_factory import instantiate_meta_rank


CANDIDATE = Path("configs/training/production-megatron-model.397b-35b.candidate.json")


def test_meta_factory_builds_rank_zero_without_real_storage() -> None:
    pytest.importorskip("torch")
    spec = load_production_megatron_model_spec(CANDIDATE)
    tensors, result = instantiate_meta_rank(
        spec,
        pipeline_rank=0,
        tensor_rank=0,
        expert_rank=0,
    )
    assert tensors
    assert all(t.device.type == "meta" for t in tensors.values())
    assert result.device == "meta"
    assert result.materialized_real_storage is False
    assert result.execution_authorized is False


def test_meta_factory_rejects_out_of_range_rank() -> None:
    pytest.importorskip("torch")
    spec = load_production_megatron_model_spec(CANDIDATE)
    with pytest.raises(ValueError):
        instantiate_meta_rank(spec, pipeline_rank=4, tensor_rank=0, expert_rank=0)
