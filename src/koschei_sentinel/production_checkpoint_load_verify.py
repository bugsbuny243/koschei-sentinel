from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_checkpoint_manifest import RankCheckpointManifest


class RankCheckpointLoadVerification(StrictModel):
    schema_version: Literal["sentinel.rank-checkpoint-load-verification.v1"] = (
        "sentinel.rank-checkpoint-load-verification.v1"
    )
    status: Literal["metadata_verified"] = "metadata_verified"
    pipeline_rank: int = Field(ge=0)
    tensor_rank: int = Field(ge=0)
    expert_rank: int = Field(ge=0)
    tensor_count: int = Field(gt=0)
    local_elements: int = Field(gt=0)
    map_location: Literal["meta"] = "meta"
    real_storage_allocated: Literal[False] = False
    forward_verified: Literal[False] = False
    backward_verified: Literal[False] = False
    execution_authorized: Literal[False] = False


def verify_rank_checkpoint_load(
    manifest: RankCheckpointManifest,
    *,
    checkpoint_path: str | Path,
) -> RankCheckpointLoadVerification:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for checkpoint load verification") from exc

    path = Path(checkpoint_path)
    if not path.is_file():
        raise ValueError(f"checkpoint shard does not exist: {path}")

    try:
        payload = torch.load(path, map_location="meta", weights_only=False)
    except Exception as exc:
        raise ValueError(f"unable to load checkpoint shard metadata: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError("checkpoint shard payload must be a mapping")
    if payload.get("schema_version") != "sentinel.rank-weight-checkpoint.v1":
        raise ValueError("unsupported checkpoint shard schema")
    if payload.get("manifest_sha256") != manifest.manifest_sha256:
        raise ValueError("checkpoint shard manifest digest mismatch")
    for field, expected in (
        ("pipeline_rank", manifest.pipeline_rank),
        ("tensor_rank", manifest.tensor_rank),
        ("expert_rank", manifest.expert_rank),
        ("dtype", manifest.dtype),
    ):
        if payload.get(field) != expected:
            raise ValueError(f"checkpoint shard {field} mismatch")

    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint shard state_dict must be a mapping")

    expected = {record.name: record for record in manifest.tensors}
    if set(state_dict) != set(expected):
        missing = sorted(set(expected) - set(state_dict))
        extra = sorted(set(state_dict) - set(expected))
        raise ValueError(f"checkpoint tensor-key mismatch; missing={missing[:3]} extra={extra[:3]}")

    local_elements = 0
    for name, record in expected.items():
        tensor = state_dict[name]
        if not hasattr(tensor, "shape") or not hasattr(tensor, "dtype"):
            raise ValueError(f"checkpoint entry is not tensor-like: {name}")
        if tuple(tensor.shape) != tuple(record.shard_shape):
            raise ValueError(f"checkpoint tensor shape mismatch: {name}")
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"checkpoint tensor dtype mismatch: {name}")
        if getattr(tensor, "device", None) is None or tensor.device.type != "meta":
            raise ValueError(f"checkpoint verification must remain on meta device: {name}")
        local_elements += tensor.numel()

    if local_elements != manifest.local_elements:
        raise ValueError("checkpoint local element count mismatch")

    return RankCheckpointLoadVerification(
        pipeline_rank=manifest.pipeline_rank,
        tensor_rank=manifest.tensor_rank,
        expert_rank=manifest.expert_rank,
        tensor_count=len(state_dict),
        local_elements=local_elements,
    )
