from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_mcore_native_checkpoint import save_mcore_native_checkpoint
from koschei_sentinel.production_mcore_native_load import load_mcore_native_checkpoint


class MCoreCheckpointRoundtripResult(StrictModel):
    schema_version: Literal["sentinel.mcore-checkpoint-roundtrip.v1"] = (
        "sentinel.mcore-checkpoint-roundtrip.v1"
    )
    status: Literal["checkpoint_roundtrip_verified"] = "checkpoint_roundtrip_verified"
    checkpoint_dir: str = Field(min_length=1)
    parameter_name: str = Field(min_length=1)
    saved_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    restored_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    checkpoint_saved: Literal[True] = True
    checkpoint_loaded: Literal[True] = True
    digest_restored: Literal[True] = True
    execution_authorized: Literal[False] = False


def _digest_sample(tensor: Any) -> str:
    flat = tensor.detach().reshape(-1)
    if flat.numel() == 0:
        raise RuntimeError("cannot digest empty checkpoint parameter")
    sample = flat[: min(4096, flat.numel())].float().cpu().contiguous()
    return hashlib.sha256(sample.numpy().tobytes()).hexdigest()


def _resolve_parameter(model: Any, parameter_name: str) -> Any:
    named = dict(model.named_parameters())
    candidates = [parameter_name]
    if parameter_name.startswith("module."):
        candidates.append(parameter_name[len("module."):])
    else:
        candidates.append(f"module.{parameter_name}")
    for candidate in candidates:
        if candidate in named:
            return named[candidate]
    raise ValueError(f"checkpoint roundtrip parameter not found: {parameter_name}")


def verify_mcore_checkpoint_roundtrip(
    model: Any,
    *,
    checkpoint_dir: str | Path,
    parameter_name: str,
) -> MCoreCheckpointRoundtripResult:
    """Save, perturb, reload, and verify one native MCore parameter digest.

    Reuses the existing graph to avoid doubling 397B model memory. All ranks must call this
    function collectively because Megatron torch_dist checkpoint save/load is distributed.
    """
    native_model = getattr(model, "module", model)
    parameter = _resolve_parameter(native_model, parameter_name)
    saved_digest = _digest_sample(parameter)

    save_mcore_native_checkpoint(
        native_model,
        checkpoint_dir=checkpoint_dir,
        content_metadata={
            "schema_version": "sentinel.mcore-native-checkpoint.v1",
            "purpose": "optimizer-smoke-roundtrip",
            "target_total_parameters_billion": 397.0,
            "target_active_parameters_billion": 35.0,
        },
        verify_integrity=True,
    )

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for checkpoint roundtrip") from exc

    with torch.no_grad():
        flat = parameter.reshape(-1)
        flat[: min(4096, flat.numel())].zero_()
    perturbed_digest = _digest_sample(parameter)
    if perturbed_digest == saved_digest:
        raise RuntimeError("checkpoint roundtrip perturbation did not change sampled parameter")

    load_mcore_native_checkpoint(
        native_model,
        checkpoint_dir=checkpoint_dir,
        verify_integrity=True,
    )
    restored_digest = _digest_sample(_resolve_parameter(native_model, parameter_name))
    if restored_digest != saved_digest:
        raise RuntimeError("native checkpoint reload did not restore sampled parameter digest")

    return MCoreCheckpointRoundtripResult(
        checkpoint_dir=Path(checkpoint_dir).as_posix(),
        parameter_name=parameter_name,
        saved_digest=saved_digest,
        restored_digest=restored_digest,
    )
