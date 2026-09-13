from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel

_DIGEST = r"^[a-f0-9]{64}$"


def _relative_path(value: str, field_name: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.startswith("~"):
        raise ValueError(f"{field_name} must stay inside the repository root")


class ProductionMegatronTopology(StrictModel):
    nodes: int = Field(ge=1, le=128)
    gpus_per_node: int = Field(ge=1, le=16)
    minimum_gpu_memory_gib: int = Field(ge=40, le=256)
    tensor_model_parallel_size: int = Field(ge=1, le=64)
    pipeline_model_parallel_size: int = Field(ge=1, le=64)
    context_parallel_size: int = Field(default=1, ge=1, le=64)
    expert_model_parallel_size: int = Field(ge=1, le=4096)
    sequence_parallel: Literal[True] = True

    @property
    def world_size(self) -> int:
        return self.nodes * self.gpus_per_node

    @property
    def model_parallel_size(self) -> int:
        return self.tensor_model_parallel_size * self.pipeline_model_parallel_size * self.context_parallel_size

    @model_validator(mode="after")
    def topology_is_coherent(self) -> "ProductionMegatronTopology":
        if self.world_size % self.model_parallel_size:
            raise ValueError("world size must be divisible by TP * PP * CP")
        data_parallel = self.world_size // self.model_parallel_size
        if data_parallel % self.expert_model_parallel_size:
            raise ValueError("data parallel size must be divisible by expert model parallel size")
        return self


class ProductionMegatronConfig(StrictModel):
    schema_version: Literal["sentinel.production-megatron-config.v1"] = "sentinel.production-megatron-config.v1"
    lane: Literal["production-target-397b-35b"] = "production-target-397b-35b"
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    model_ref: str = Field(min_length=1)
    model_revision: str = Field(pattern=_DIGEST)
    total_parameters: Literal["397B"] = "397B"
    active_parameters: Literal["35B"] = "35B"
    architecture_class: Literal["sparse-moe"] = "sparse-moe"
    trainer_adapter: Literal["megatron-swift"] = "megatron-swift"
    trainer_adapter_version: str = Field(min_length=1)
    corpus_dir: str = Field(min_length=1)
    validation_corpus_dir: str = Field(min_length=1)
    dataset_dir: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    topology: ProductionMegatronTopology
    launch_approval_env: Literal["KOSCHEI_397B_35B_LAUNCH_APPROVED"] = "KOSCHEI_397B_35B_LAUNCH_APPROVED"
    production_target_binding_sha256: str = Field(pattern=_DIGEST)
    production_topology_verification_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def config_is_safe(self) -> "ProductionMegatronConfig":
        for name, value in (("corpus_dir", self.corpus_dir), ("validation_corpus_dir", self.validation_corpus_dir), ("dataset_dir", self.dataset_dir), ("output_dir", self.output_dir)):
            _relative_path(value, name)
        if self.corpus_dir == self.validation_corpus_dir:
            raise ValueError("production training and validation corpora must differ")
        if self.dataset_dir == self.output_dir:
            raise ValueError("dataset_dir must differ from output_dir")
        return self


def load_production_megatron_config(path: str | Path) -> ProductionMegatronConfig:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid production Megatron config: {source}") from exc
    return ProductionMegatronConfig.model_validate(payload)
