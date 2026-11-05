from __future__ import annotations

import hashlib
import os
import re
import stat
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


def _hash_open_file(handle) -> tuple[str, int]:
    handle.seek(0)
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def verify_rank_checkpoint_load(
    manifest: RankCheckpointManifest,
    *,
    checkpoint_path: str | Path,
    expected_file_sha256: str | None = None,
    expected_file_size_bytes: int | None = None,
) -> RankCheckpointLoadVerification:
    """Validate tensor metadata; optional artifact pin is checked before loading.

    Passing only a digest does not establish who approved it. The signed
    checkpoint intake verifies the trust root before invoking this function.
    """
    if (expected_file_sha256 is None) != (expected_file_size_bytes is None):
        raise ValueError("checkpoint artifact hash and byte size must be supplied together")
    if expected_file_sha256 is not None and re.fullmatch(
        r"[a-f0-9]{64}", expected_file_sha256
    ) is None:
        raise ValueError("checkpoint artifact SHA-256 must be lowercase hex")
    if expected_file_size_bytes is not None and (
        type(expected_file_size_bytes) is not int or expected_file_size_bytes <= 0
    ):
        raise ValueError("checkpoint artifact size must be a positive integer")

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for checkpoint load verification") from exc

    path = Path(checkpoint_path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"checkpoint shard cannot be opened safely: {path}") from exc

    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("checkpoint shard must be a regular file")
        if expected_file_sha256 is not None:
            observed_digest, observed_bytes = _hash_open_file(handle)
            if observed_bytes != expected_file_size_bytes:
                raise ValueError("checkpoint artifact byte size mismatch")
            if observed_digest != expected_file_sha256:
                raise ValueError("checkpoint artifact SHA-256 mismatch")
            handle.seek(0)

        # Inspect tensor metadata without allowing arbitrary Python object unpickling.
        # Meta tensors do not authenticate the checkpoint or prevent resource exhaustion.
        try:
            payload = torch.load(handle, map_location="meta", weights_only=True)
        except Exception as exc:
            raise ValueError(f"unable to load checkpoint shard metadata: {path}") from exc

        if expected_file_sha256 is not None:
            # Detect in-place replacement or mutation during metadata inspection.
            observed_digest, observed_bytes = _hash_open_file(handle)
            if observed_bytes != expected_file_size_bytes or observed_digest != expected_file_sha256:
                raise ValueError("checkpoint artifact changed during metadata inspection")

    if not isinstance(payload, dict):
        raise ValueError("checkpoint shard payload must be a mapping")  # noqa: TRY004 - preserve ValueError validation/manifest API contract
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
        raise ValueError("checkpoint shard state_dict must be a mapping")  # noqa: TRY004 - preserve ValueError validation/manifest API contract

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
