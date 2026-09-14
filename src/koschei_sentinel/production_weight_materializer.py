from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_checkpoint_manifest import (
    RankCheckpointManifest,
    TensorInitializationRecord,
)

_GIB = 1024 ** 3


class RankWeightMaterializationResult(StrictModel):
    schema_version: Literal["sentinel.rank-weight-materialization.v1"] = (
        "sentinel.rank-weight-materialization.v1"
    )
    status: Literal["rank_weights_materialized"] = "rank_weights_materialized"
    device: str = Field(min_length=1)
    dtype: Literal["bfloat16"] = "bfloat16"
    tensor_count: int = Field(gt=0)
    local_elements: int = Field(gt=0)
    estimated_weight_gib: float = Field(gt=0.0)
    checkpoint_written: bool = False
    checkpoint_path: str | None = None
    execution_authorized: Literal[False] = False


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("PyTorch is required for weight materialization") from exc
    return torch


def materialize_initialization_record(
    record: TensorInitializationRecord,
    *,
    device: str,
) -> Any:
    torch = _torch()
    generator_device = device if device.startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(record.seed)

    if record.initialization == "ones":
        return torch.ones(record.shard_shape, device=device, dtype=torch.bfloat16)

    tensor = torch.empty(record.shard_shape, device=device, dtype=torch.bfloat16)
    tensor.normal_(
        mean=float(record.mean),
        std=float(record.std),
        generator=generator,
    )
    return tensor


def materialize_rank_weights(
    manifest: RankCheckpointManifest,
    *,
    device: str,
    max_weight_gib: float = 8.0,
    require_cuda: bool = False,
) -> tuple[dict[str, Any], RankWeightMaterializationResult]:
    if max_weight_gib <= 0:
        raise ValueError("max_weight_gib must be positive")
    if require_cuda and not device.startswith("cuda"):
        raise ValueError("production materialization requires a CUDA device")

    torch = _torch()
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        device_index = torch.device(device).index
        if device_index is None:
            device_index = torch.cuda.current_device()
        free_bytes, _ = torch.cuda.mem_get_info(device_index)
    else:
        free_bytes = None

    estimated_bytes = manifest.local_elements * 2
    estimated_gib = estimated_bytes / _GIB
    if estimated_gib > max_weight_gib:
        raise RuntimeError(
            f"rank weight estimate {estimated_gib:.4f} GiB exceeds materialization cap {max_weight_gib:.4f} GiB"
        )
    if free_bytes is not None and estimated_bytes > int(free_bytes * 0.85):
        raise RuntimeError("insufficient free CUDA memory for guarded rank materialization")

    tensors: dict[str, Any] = {}
    for record in manifest.tensors:
        tensors[record.name] = materialize_initialization_record(record, device=device)

    return tensors, RankWeightMaterializationResult(
        device=device,
        tensor_count=len(tensors),
        local_elements=manifest.local_elements,
        estimated_weight_gib=round(estimated_gib, 6),
    )


def write_rank_checkpoint(
    manifest: RankCheckpointManifest,
    *,
    output_path: str | Path,
    device: str,
    max_weight_gib: float = 8.0,
    require_cuda: bool = True,
) -> RankWeightMaterializationResult:
    """Materialize one PP/TP/EP rank and atomically write its BF16 state shard.

    This function writes model-initialization weights only. It does not create optimizer
    state, run collectives, authorize a training launch, or claim a valid global checkpoint.
    """
    torch = _torch()
    tensors, result = materialize_rank_weights(
        manifest,
        device=device,
        max_weight_gib=max_weight_gib,
        require_cuda=require_cuda,
    )

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if target.exists():
        raise FileExistsError(f"checkpoint shard already exists: {target}")
    try:
        torch.save(
            {
                "schema_version": "sentinel.rank-weight-checkpoint.v1",
                "manifest_sha256": manifest.manifest_sha256,
                "pipeline_rank": manifest.pipeline_rank,
                "tensor_rank": manifest.tensor_rank,
                "expert_rank": manifest.expert_rank,
                "dtype": manifest.dtype,
                "state_dict": tensors,
            },
            temporary,
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()

    return result.model_copy(
        update={
            "checkpoint_written": True,
            "checkpoint_path": target.as_posix(),
        }
    )
