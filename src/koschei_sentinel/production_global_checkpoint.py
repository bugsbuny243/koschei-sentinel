from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_checkpoint_manifest import build_rank_checkpoint_manifest
from koschei_sentinel.production_megatron_model_spec import ProductionMegatronModelSpec

_DIGEST = r"^[a-f0-9]{64}$"


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.startswith("~"):
        raise ValueError("checkpoint shard path must stay relative to the checkpoint root")


class GlobalCheckpointShard(StrictModel):
    model_shard_rank: int = Field(ge=0)
    pipeline_rank: int = Field(ge=0)
    tensor_rank: int = Field(ge=0)
    expert_rank: int = Field(ge=0)
    tensor_count: int = Field(gt=0)
    local_elements: int = Field(gt=0)
    initialization_manifest_sha256: str = Field(pattern=_DIGEST)
    relative_path: str = Field(min_length=1)
    materialized_file_sha256: str | None = Field(default=None, pattern=_DIGEST)

    @model_validator(mode="after")
    def shard_path_is_local(self) -> "GlobalCheckpointShard":
        _relative_path(self.relative_path)
        return self


class GlobalCheckpointIndex(StrictModel):
    schema_version: Literal["sentinel.global-checkpoint-index.v1"] = "sentinel.global-checkpoint-index.v1"
    status: Literal["initialization_index"] = "initialization_index"
    checkpoint_format: Literal["torch_dist"] = "torch_dist"
    global_seed: int = Field(ge=0, lt=2**63)
    pipeline_parallel_size: int = Field(gt=0)
    tensor_parallel_size: int = Field(gt=0)
    expert_parallel_size: int = Field(gt=0)
    unique_model_shard_count: int = Field(gt=0)
    process_world_size: int = Field(gt=0)
    context_parallel_size: int = Field(gt=0)
    data_parallel_size: int = Field(gt=0)
    shards: list[GlobalCheckpointShard] = Field(min_length=1)
    fully_materialized: bool = False
    loadable_checkpoint: Literal[False] = False
    execution_authorized: Literal[False] = False
    index_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def verify_index(self) -> "GlobalCheckpointIndex":
        expected = self.pipeline_parallel_size * self.tensor_parallel_size * self.expert_parallel_size
        if self.unique_model_shard_count != expected:
            raise ValueError("unique_model_shard_count disagrees with PP*TP*EP")
        if len(self.shards) != expected:
            raise ValueError("global checkpoint index is incomplete")
        coords = {(s.pipeline_rank, s.tensor_rank, s.expert_rank) for s in self.shards}
        if len(coords) != expected:
            raise ValueError("duplicate or missing PP/TP/EP checkpoint coordinates")
        ranks = {s.model_shard_rank for s in self.shards}
        if ranks != set(range(expected)):
            raise ValueError("model_shard_rank must be contiguous from zero")
        if self.fully_materialized != all(s.materialized_file_sha256 is not None for s in self.shards):
            raise ValueError("fully_materialized disagrees with shard file digests")
        if self.loadable_checkpoint is not False or self.execution_authorized is not False:
            raise ValueError("initialization index cannot claim load or execution authority")
        payload = self.model_dump(mode="json", exclude={"index_sha256"})
        if _digest(payload) != self.index_sha256:
            raise ValueError("global checkpoint index digest mismatch")
        return self


def build_global_checkpoint_index(
    spec: ProductionMegatronModelSpec,
    *,
    global_seed: int = 39735,
    shard_dir: str = "model-shards",
) -> GlobalCheckpointIndex:
    _relative_path(shard_dir)
    topo = spec.topology
    pp = topo.pipeline_model_parallel_size
    tp = topo.tensor_model_parallel_size
    ep = topo.expert_model_parallel_size
    unique = pp * tp * ep
    data_parallel = topo.world_size // topo.model_parallel_size

    shards: list[dict[str, object]] = []
    model_shard_rank = 0
    for pipeline_rank in range(pp):
        for tensor_rank in range(tp):
            for expert_rank in range(ep):
                manifest = build_rank_checkpoint_manifest(
                    spec,
                    pipeline_rank=pipeline_rank,
                    tensor_rank=tensor_rank,
                    expert_rank=expert_rank,
                    global_seed=global_seed,
                )
                relative_path = (
                    f"{shard_dir}/pp-{pipeline_rank:02d}/"
                    f"tp-{tensor_rank:02d}-ep-{expert_rank:02d}.pt"
                )
                shards.append(
                    {
                        "model_shard_rank": model_shard_rank,
                        "pipeline_rank": pipeline_rank,
                        "tensor_rank": tensor_rank,
                        "expert_rank": expert_rank,
                        "tensor_count": manifest.tensor_count,
                        "local_elements": manifest.local_elements,
                        "initialization_manifest_sha256": manifest.manifest_sha256,
                        "relative_path": relative_path,
                        "materialized_file_sha256": None,
                    }
                )
                model_shard_rank += 1

    payload = {
        "schema_version": "sentinel.global-checkpoint-index.v1",
        "status": "initialization_index",
        "checkpoint_format": "torch_dist",
        "global_seed": global_seed,
        "pipeline_parallel_size": pp,
        "tensor_parallel_size": tp,
        "expert_parallel_size": ep,
        "unique_model_shard_count": unique,
        "process_world_size": topo.world_size,
        "context_parallel_size": topo.context_parallel_size,
        "data_parallel_size": data_parallel,
        "shards": shards,
        "fully_materialized": False,
        "loadable_checkpoint": False,
        "execution_authorized": False,
    }
    return GlobalCheckpointIndex.model_validate({**payload, "index_sha256": _digest(payload)})
