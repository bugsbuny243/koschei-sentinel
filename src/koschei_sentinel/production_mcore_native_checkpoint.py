from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from koschei_sentinel.models import StrictModel


class MCoreNativeInitializationResult(StrictModel):
    schema_version: Literal["sentinel.mcore-native-init.v1"] = "sentinel.mcore-native-init.v1"
    status: Literal["native_parameters_initialized"] = "native_parameters_initialized"
    parameter_count: int = Field(gt=0)
    local_elements: int = Field(gt=0)
    global_seed: int = Field(ge=0, lt=2**63)
    dtype: Literal["bfloat16"] = "bfloat16"
    mcore_native_layout: Literal[True] = True
    checkpoint_written: bool = False
    checkpoint_dir: str | None = None
    forward_verified: Literal[False] = False
    backward_verified: Literal[False] = False
    execution_authorized: Literal[False] = False


def _seed64(global_seed: int, name: str) -> int:
    payload = f"{global_seed}\x1f{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _is_norm_parameter(name: str) -> bool:
    lowered = name.lower()
    return (
        "layernorm" in lowered
        or ".norm." in lowered
        or lowered.endswith("norm.weight")
        or lowered.endswith("layer_norm_weight")
    )


def initialize_mcore_native_parameters(
    model: Any,
    *,
    global_seed: int = 39735,
    std: float | None = None,
) -> MCoreNativeInitializationResult:
    """Initialize the actual Megatron-Core model parameters in their native layout.

    This intentionally initializes the graph's own fused QKV/grouped-expert parameters
    rather than converting Sentinel's architecture-recipe checkpoint names into MCore.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for MCore native initialization") from exc

    if not 0 <= global_seed < 2**63:
        raise ValueError("global_seed outside signed 63-bit range")
    if not hasattr(model, "named_parameters"):
        raise TypeError("model must expose named_parameters")

    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden_size is None or int(hidden_size) <= 0:
        raise ValueError("Megatron-Core model config must expose a positive hidden_size")
    init_std = float(std) if std is not None else int(hidden_size) ** -0.5
    if init_std <= 0:
        raise ValueError("initialization std must be positive")

    parameter_count = 0
    local_elements = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter is None:
                continue
            if parameter.device.type == "meta":
                raise RuntimeError(f"cannot initialize meta parameter without materialization: {name}")
            parameter_count += 1
            local_elements += parameter.numel()
            if _is_norm_parameter(name):
                parameter.fill_(1.0)
                continue
            generator_device = parameter.device if parameter.device.type == "cuda" else torch.device("cpu")
            generator = torch.Generator(device=generator_device)
            generator.manual_seed(_seed64(global_seed, name))
            parameter.normal_(mean=0.0, std=init_std, generator=generator)

    if parameter_count == 0 or local_elements == 0:
        raise ValueError("Megatron-Core model exposed no parameters")

    return MCoreNativeInitializationResult(
        parameter_count=parameter_count,
        local_elements=local_elements,
        global_seed=global_seed,
    )


def save_mcore_native_checkpoint(
    model: Any,
    *,
    checkpoint_dir: str | Path,
    content_metadata: dict[str, Any] | None = None,
    verify_integrity: bool = True,
) -> None:
    """Save the model using Megatron-Core's native torch_dist sharded checkpoint API."""
    try:
        from megatron.core.dist_checkpointing.serialization import save
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Megatron-Core distributed checkpointing is required") from exc

    if not hasattr(model, "sharded_state_dict"):
        raise TypeError("model must expose sharded_state_dict")

    target = Path(checkpoint_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"checkpoint directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)

    sharded = model.sharded_state_dict()
    save(
        sharded,
        target.as_posix(),
        validate_access_integrity=True,
        async_sharded_save=False,
        content_metadata=content_metadata or {
            "schema_version": "sentinel.mcore-native-checkpoint.v1",
            "target_total_parameters_billion": 397.0,
            "target_active_parameters_billion": 35.0,
        },
        verify_integrity=verify_integrity,
    )
