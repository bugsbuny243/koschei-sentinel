from __future__ import annotations

import pytest
from pydantic import ValidationError

from koschei_sentinel.production_megatron_config import ProductionMegatronConfig, ProductionMegatronTopology


def _topology() -> ProductionMegatronTopology:
    return ProductionMegatronTopology(
        nodes=8,
        gpus_per_node=8,
        minimum_gpu_memory_gib=80,
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=8,
    )


def _config(**overrides):
    payload = dict(
        run_id="sentinel-prod-397b-35b-001",
        model_ref="koschei/sentinel-397b-35b",
        model_revision="a" * 64,
        trainer_adapter_version="4.5.2",
        corpus_dir="build/prod/train",
        validation_corpus_dir="build/prod/validation",
        dataset_dir="build/prod/dataset",
        output_dir="build/prod/output",
        topology=_topology(),
        production_target_binding_sha256="b" * 64,
        production_topology_verification_sha256="c" * 64,
    )
    payload.update(overrides)
    return ProductionMegatronConfig(**payload)


def test_production_config_is_exactly_397b_35b() -> None:
    config = _config()
    assert config.total_parameters == "397B"
    assert config.active_parameters == "35B"
    assert config.lane == "production-target-397b-35b"


def test_validation_and_training_corpus_must_differ() -> None:
    with pytest.raises(ValidationError):
        _config(validation_corpus_dir="build/prod/train")


def test_a17b_cannot_masquerade_as_production_active_width() -> None:
    with pytest.raises(ValidationError):
        _config(active_parameters="17B")
