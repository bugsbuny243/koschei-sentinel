from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_mcore_training_checkpoint import MCoreTrainingState, _combined_sharded_state


class AsyncCheckpointConfig(StrictModel):
    schema_version: Literal["sentinel.mcore-async-checkpoint-config.v1"] = "sentinel.mcore-async-checkpoint-config.v1"
    enabled: bool = True
    strategy: Literal["nvrx", "mcore"] = "nvrx"
    persistent_queue: bool = True
    max_unfinalized: int = Field(default=1, ge=1, le=4)
    verify_integrity: bool = False


@dataclass
class MCoreAsyncCheckpointQueue:
    config: AsyncCheckpointConfig
    _queue: Any = None

    def _ensure_queue(self):
        if self._queue is None:
            try:
                from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Megatron-Core async checkpoint queue is unavailable") from exc
            self._queue = AsyncCallsQueue(persistent=self.config.persistent_queue)
        return self._queue

    def maybe_finalize(self, *, blocking: bool = False) -> list[int]:
        if self._queue is None:
            return []
        return list(self._queue.maybe_finalize_async_calls(blocking=blocking))

    def wait(self) -> list[int]:
        return self.maybe_finalize(blocking=True)

    def close(self, *, abort: bool = False) -> None:
        if self._queue is not None:
            self._queue.close(abort=abort)
            self._queue = None

    def save(
        self,
        model: Any,
        optimizer: Any,
        training_state: MCoreTrainingState,
        *,
        checkpoint_dir: str | Path,
    ) -> int | None:
        if not self.config.enabled:
            raise RuntimeError("async checkpoint queue is disabled; use synchronous checkpoint save")
        try:
            from megatron.core.dist_checkpointing.serialization import save
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Megatron-Core distributed checkpointing is required") from exc

        queue = self._ensure_queue()
        # Bound outstanding host-memory snapshots. This blocks before scheduling a new
        # checkpoint instead of allowing unbounded 397B checkpoint staging pressure.
        if queue.get_num_unfinalized_calls() >= self.config.max_unfinalized:
            queue.maybe_finalize_async_calls(blocking=True)

        target = Path(checkpoint_dir)
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(f"training checkpoint directory is not empty: {target}")
        target.mkdir(parents=True, exist_ok=True)
        state = _combined_sharded_state(model, optimizer, training_state, is_loading=False)
        request = save(
            state,
            target.as_posix(),
            validate_access_integrity=True,
            async_sharded_save=True,
            content_metadata={
                "schema_version": "sentinel.mcore-training-checkpoint.v2",
                "training_state_schema_version": training_state.schema_version,
                "global_step": training_state.global_step,
                "has_data_state": training_state.data_state is not None,
                "has_recovery_state": training_state.recovery_state is not None,
                "has_trend_state": training_state.trend_state is not None,
                "target_total_parameters_billion": 397.0,
                "target_active_parameters_billion": 35.0,
            },
            async_strategy=self.config.strategy,
            verify_integrity=self.config.verify_integrity,
        )
        if request is None:
            raise RuntimeError("Megatron-Core async checkpoint save returned no AsyncRequest")
        return int(queue.schedule_async_request(request))
