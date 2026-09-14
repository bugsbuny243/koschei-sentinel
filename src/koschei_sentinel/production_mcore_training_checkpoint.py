from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from koschei_sentinel.models import StrictModel


class MCoreTrainingState(StrictModel):
    schema_version: Literal["sentinel.mcore-training-state.v1"] = "sentinel.mcore-training-state.v1"
    global_step: int = Field(ge=0)
    consumed_microbatches: int = Field(ge=0)
    learning_rate: float = Field(gt=0.0)
    global_seed: int = Field(ge=0, lt=2**63)
    data_state: dict[str, Any] | None = None


class MCoreTrainingCheckpointResult(StrictModel):
    schema_version: Literal["sentinel.mcore-training-checkpoint.v1"] = "sentinel.mcore-training-checkpoint.v1"
    status: Literal["training_checkpoint_saved", "training_checkpoint_loaded"]
    checkpoint_dir: str = Field(min_length=1)
    global_step: int = Field(ge=0)
    model_state_included: Literal[True] = True
    optimizer_state_included: Literal[True] = True
    training_state_included: Literal[True] = True
    execution_authorized: Literal[False] = False


def _combined_sharded_state(model: Any, optimizer: Any, training_state: MCoreTrainingState, *, is_loading: bool):
    if not hasattr(model, "sharded_state_dict"):
        raise TypeError("model must expose sharded_state_dict")
    if not hasattr(optimizer, "sharded_state_dict"):
        raise TypeError("optimizer must expose sharded_state_dict")
    model_state = model.sharded_state_dict()
    optimizer_state = optimizer.sharded_state_dict(model_state, is_loading=is_loading)
    return {
        "model": model_state,
        "optimizer": optimizer_state,
        "training_state": training_state.model_dump(mode="json"),
    }


def save_mcore_training_checkpoint(model: Any, optimizer: Any, training_state: MCoreTrainingState, *, checkpoint_dir: str | Path) -> MCoreTrainingCheckpointResult:
    try:
        from megatron.core.dist_checkpointing.serialization import save
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Megatron-Core distributed checkpointing is required") from exc
    target = Path(checkpoint_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"training checkpoint directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    state = _combined_sharded_state(model, optimizer, training_state, is_loading=False)
    save(
        state,
        target.as_posix(),
        validate_access_integrity=True,
        async_sharded_save=False,
        content_metadata={
            "schema_version": "sentinel.mcore-training-checkpoint.v1",
            "global_step": training_state.global_step,
            "has_data_state": training_state.data_state is not None,
            "target_total_parameters_billion": 397.0,
            "target_active_parameters_billion": 35.0,
        },
        verify_integrity=True,
    )
    return MCoreTrainingCheckpointResult(status="training_checkpoint_saved", checkpoint_dir=target.as_posix(), global_step=training_state.global_step)


def load_mcore_training_checkpoint(model: Any, optimizer: Any, *, checkpoint_dir: str | Path, expected_global_seed: int) -> tuple[MCoreTrainingState, MCoreTrainingCheckpointResult]:
    try:
        from megatron.core.dist_checkpointing.serialization import load
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Megatron-Core distributed checkpointing is required") from exc
    source = Path(checkpoint_dir)
    if not source.is_dir():
        raise ValueError(f"training checkpoint directory does not exist: {source}")
    template_training_state = MCoreTrainingState(global_step=0, consumed_microbatches=0, learning_rate=1.0, global_seed=expected_global_seed, data_state=None)
    template = _combined_sharded_state(model, optimizer, template_training_state, is_loading=True)
    loaded = load(template, source.as_posix(), validate_access_integrity=True, strict="return_all", verify_integrity=True)
    missing: list[str] = []
    unexpected: list[str] = []
    state = loaded
    if isinstance(loaded, tuple):
        if len(loaded) != 3:
            raise RuntimeError("unexpected MCore training checkpoint load return shape")
        state, missing_raw, unexpected_raw = loaded
        missing = sorted(str(item) for item in missing_raw)
        unexpected = sorted(str(item) for item in unexpected_raw)
    if missing or unexpected:
        raise ValueError(f"training checkpoint mismatch; missing={missing[:5]} unexpected={unexpected[:5]}")
    if not isinstance(state, dict):
        raise ValueError("training checkpoint state must be a mapping")
    model_state, optimizer_state, training_state_raw = state.get("model"), state.get("optimizer"), state.get("training_state")
    if model_state is None or optimizer_state is None or training_state_raw is None:
        raise ValueError("training checkpoint missing model, optimizer, or training state")
    incompatible = model.load_state_dict(model_state, strict=True)
    if getattr(incompatible, "missing_keys", None) or getattr(incompatible, "unexpected_keys", None):
        raise ValueError("model state is incompatible during training resume")
    optimizer.load_state_dict(optimizer_state)
    if hasattr(optimizer, "reload_model_params"):
        optimizer.reload_model_params()
    training_state = MCoreTrainingState.model_validate(training_state_raw)
    if training_state.global_seed != expected_global_seed:
        raise ValueError("training checkpoint global seed mismatch")
    return training_state, MCoreTrainingCheckpointResult(status="training_checkpoint_loaded", checkpoint_dir=source.as_posix(), global_step=training_state.global_step)
