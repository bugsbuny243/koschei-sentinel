from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from koschei_sentinel.production_mcore_async_checkpoint import AsyncCheckpointConfig, MCoreAsyncCheckpointQueue
from koschei_sentinel.production_mcore_checkpoint_retention import (
    RecoveryCheckpointEntry,
    RecoveryCheckpointIndex,
    load_recovery_checkpoint_index,
    record_recovery_checkpoint,
)
from koschei_sentinel.production_mcore_training_checkpoint import MCoreTrainingState


@dataclass
class RecoveryCheckpointManager:
    root: Path
    async_config: AsyncCheckpointConfig
    keep_recovery_slots: int = 2
    queue: MCoreAsyncCheckpointQueue = field(init=False)
    index: RecoveryCheckpointIndex = field(init=False)
    pending: dict[int, RecoveryCheckpointEntry] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.root = self.root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.queue = MCoreAsyncCheckpointQueue(self.async_config)
        self.index = load_recovery_checkpoint_index(
            self.root,
            keep_recovery_slots=self.keep_recovery_slots,
        )

    def _finalize_ids(self, ids: list[int]) -> list[str]:
        deleted: list[str] = []
        for request_id in ids:
            entry = self.pending.pop(int(request_id), None)
            if entry is None:
                raise RuntimeError(f"async checkpoint finalized unknown request id {request_id}")
            self.index, removed = record_recovery_checkpoint(self.root, self.index, entry)
            deleted.extend(removed)
        return deleted

    def poll(self) -> list[str]:
        return self._finalize_ids(self.queue.maybe_finalize(blocking=False))

    def wait(self) -> list[str]:
        return self._finalize_ids(self.queue.wait())

    def save_async(
        self,
        model,
        optimizer,
        state: MCoreTrainingState,
        *,
        checkpoint_dir: str | Path,
        kind: str,
        durable: bool = False,
    ) -> int:
        # Finalize completed saves before allocating another host-side snapshot.
        self.poll()
        target = Path(checkpoint_dir).resolve()
        entry = RecoveryCheckpointEntry(
            global_step=state.global_step,
            checkpoint_dir=target.as_posix(),
            kind=kind,
            durable=durable,
        )
        request_id = self.queue.save(model, optimizer, state, checkpoint_dir=target)
        if request_id is None:
            raise RuntimeError("async checkpoint manager expected a request id")
        if request_id in self.pending:
            raise RuntimeError("async checkpoint request id collision")
        self.pending[request_id] = entry
        return request_id

    def close(self, *, abort: bool = False) -> list[str]:
        deleted: list[str] = []
        if abort:
            self.queue.close(abort=True)
            self.pending.clear()
            return deleted
        deleted.extend(self.wait())
        if self.pending:
            raise RuntimeError("async checkpoint queue closed with unfinalized recovery entries")
        self.queue.close(abort=False)
        return deleted
