from __future__ import annotations

import hashlib
import json
from math import prod
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_megatron_model_spec import ProductionMegatronModelSpec
from koschei_sentinel.production_tensor_layout import TensorLayoutEntry, build_tensor_layout

_DIGEST = r"^[a-f0-9]{64}$"


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seed64(*parts: object) -> int:
    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


class TensorInitializationRecord(StrictModel):
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    shard_shape: tuple[int, ...]
    local_elements: int = Field(gt=0)
    initialization: Literal["normal", "ones"]
    mean: float | None = None
    std: float | None = Field(default=None, gt=0.0)
    seed: int = Field(ge=0, lt=2**63)
    recipe_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def initialization_parameters_are_coherent(self) -> "TensorInitializationRecord":
        if self.initialization == "normal":
            if self.mean is None or self.std is None:
                raise ValueError("normal initialization requires mean and std")
        elif self.mean is not None or self.std is not None:
            raise ValueError("ones initialization cannot carry normal mean/std")
        payload = self.model_dump(mode="json", exclude={"recipe_sha256"})
        if _canonical_digest(payload) != self.recipe_sha256:
            raise ValueError("tensor initialization recipe digest mismatch")
        return self


class RankCheckpointManifest(StrictModel):
    schema_version: Literal["sentinel.rank-checkpoint-manifest.v1"] = (
        "sentinel.rank-checkpoint-manifest.v1"
    )
    status: Literal["initialization_recipe"] = "initialization_recipe"
    global_seed: int = Field(ge=0, lt=2**63)
    pipeline_rank: int = Field(ge=0)
    tensor_rank: int = Field(ge=0)
    expert_rank: int = Field(ge=0)
    tensor_count: int = Field(gt=0)
    local_elements: int = Field(gt=0)
    dtype: Literal["bfloat16"] = "bfloat16"
    checkpoint_format: Literal["torch_dist"] = "torch_dist"
    tensors: list[TensorInitializationRecord] = Field(min_length=1)
    materialized_weights: Literal[False] = False
    execution_authorized: Literal[False] = False
    manifest_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def verify_manifest(self) -> "RankCheckpointManifest":
        if self.tensor_count != len(self.tensors):
            raise ValueError("tensor_count disagrees with tensors")
        if self.local_elements != sum(item.local_elements for item in self.tensors):
            raise ValueError("local_elements disagrees with tensors")
        payload = self.model_dump(mode="json", exclude={"manifest_sha256"})
        if _canonical_digest(payload) != self.manifest_sha256:
            raise ValueError("rank checkpoint manifest digest mismatch")
        return self


def _rank_owns(entry: TensorLayoutEntry, *, pp: int, tp: int, ep: int) -> bool:
    if entry.pipeline_stage != pp:
        return False
    if not entry.replicated_across_tensor_parallel and tp >= entry.tensor_parallel_shards:
        return False
    if not entry.replicated_across_expert_parallel and ep >= entry.expert_parallel_shards:
        return False
    return True


def _initialization_for(
    entry: TensorLayoutEntry,
    *,
    global_seed: int,
    pp: int,
    tp: int,
    ep: int,
    hidden_size: int,
) -> TensorInitializationRecord:
    seed = _seed64(global_seed, entry.name, pp, tp, ep)
    local_elements = prod(entry.shard_shape)
    if entry.category == "normalization":
        base = {
            "name": entry.name,
            "category": entry.category,
            "shard_shape": entry.shard_shape,
            "local_elements": local_elements,
            "initialization": "ones",
            "mean": None,
            "std": None,
            "seed": seed,
        }
    else:
        # Conservative transformer-from-scratch recipe. This is a deterministic recipe,
        # not evidence that the 397B/35B model has converged or even been materialized.
        base = {
            "name": entry.name,
            "category": entry.category,
            "shard_shape": entry.shard_shape,
            "local_elements": local_elements,
            "initialization": "normal",
            "mean": 0.0,
            "std": hidden_size ** -0.5,
            "seed": seed,
        }
    return TensorInitializationRecord.model_validate(
        {**base, "recipe_sha256": _canonical_digest(base)}
    )


def build_rank_checkpoint_manifest(
    spec: ProductionMegatronModelSpec,
    *,
    pipeline_rank: int,
    tensor_rank: int,
    expert_rank: int,
    global_seed: int = 39735,
) -> RankCheckpointManifest:
    topo = spec.topology
    if not 0 <= pipeline_rank < topo.pipeline_model_parallel_size:
        raise ValueError("pipeline_rank outside configured pipeline parallel size")
    if not 0 <= tensor_rank < topo.tensor_model_parallel_size:
        raise ValueError("tensor_rank outside configured tensor parallel size")
    if not 0 <= expert_rank < topo.expert_model_parallel_size:
        raise ValueError("expert_rank outside configured expert parallel size")
    if not 0 <= global_seed < 2**63:
        raise ValueError("global_seed outside signed 63-bit range")

    layout = build_tensor_layout(spec)
    tensors = [
        _initialization_for(
            entry,
            global_seed=global_seed,
            pp=pipeline_rank,
            tp=tensor_rank,
            ep=expert_rank,
            hidden_size=spec.transformer.hidden_size,
        )
        for entry in layout.entries
        if _rank_owns(
            entry,
            pp=pipeline_rank,
            tp=tensor_rank,
            ep=expert_rank,
        )
    ]
    if not tensors:
        raise ValueError("rank owns no tensors; topology/layout mismatch")

    payload = {
        "schema_version": "sentinel.rank-checkpoint-manifest.v1",
        "status": "initialization_recipe",
        "global_seed": global_seed,
        "pipeline_rank": pipeline_rank,
        "tensor_rank": tensor_rank,
        "expert_rank": expert_rank,
        "tensor_count": len(tensors),
        "local_elements": sum(item.local_elements for item in tensors),
        "dtype": "bfloat16",
        "checkpoint_format": "torch_dist",
        "tensors": [item.model_dump(mode="json") for item in tensors],
        "materialized_weights": False,
        "execution_authorized": False,
    }
    return RankCheckpointManifest.model_validate(
        {**payload, "manifest_sha256": _canonical_digest(payload)}
    )
