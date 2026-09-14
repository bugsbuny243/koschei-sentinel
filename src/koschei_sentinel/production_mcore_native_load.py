from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from koschei_sentinel.models import StrictModel


class MCoreNativeLoadResult(StrictModel):
    schema_version: Literal["sentinel.mcore-native-load.v1"] = "sentinel.mcore-native-load.v1"
    status: Literal["native_checkpoint_loaded"] = "native_checkpoint_loaded"
    checkpoint_dir: str = Field(min_length=1)
    missing_keys: list[str] = Field(default_factory=list)
    unexpected_keys: list[str] = Field(default_factory=list)
    megatron_load_verified: Literal[True] = True
    forward_verified: Literal[False] = False
    backward_verified: Literal[False] = False
    execution_authorized: Literal[False] = False


def load_mcore_native_checkpoint(
    model: Any,
    *,
    checkpoint_dir: str | Path,
    verify_integrity: bool = True,
) -> MCoreNativeLoadResult:
    try:
        from megatron.core.dist_checkpointing.serialization import load
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Megatron-Core distributed checkpointing is required") from exc

    path = Path(checkpoint_dir)
    if not path.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {path}")
    if not hasattr(model, "sharded_state_dict") or not hasattr(model, "load_state_dict"):
        raise TypeError("model must expose sharded_state_dict and load_state_dict")

    template = model.sharded_state_dict()
    loaded = load(
        template,
        path.as_posix(),
        validate_access_integrity=True,
        strict="return_all",
        verify_integrity=verify_integrity,
    )

    missing: list[str] = []
    unexpected: list[str] = []
    state = loaded
    if isinstance(loaded, tuple):
        if len(loaded) != 3:
            raise RuntimeError("unexpected Megatron-Core checkpoint load return shape")
        state, missing_raw, unexpected_raw = loaded
        missing = sorted(str(item) for item in missing_raw)
        unexpected = sorted(str(item) for item in unexpected_raw)
    if missing or unexpected:
        raise ValueError(
            f"Megatron-Core checkpoint mismatch; missing={missing[:5]} unexpected={unexpected[:5]}"
        )

    incompatible = model.load_state_dict(state, strict=True)
    if getattr(incompatible, "missing_keys", None) or getattr(incompatible, "unexpected_keys", None):
        raise ValueError("Megatron-Core model load_state_dict reported incompatible keys")

    return MCoreNativeLoadResult(
        checkpoint_dir=path.as_posix(),
        missing_keys=missing,
        unexpected_keys=unexpected,
    )
