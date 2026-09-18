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

    @property
    def committed_last_good(self) -> RecoveryCheckpointEntry | None:
        """Newest finalized checkpoint only; pending async saves are never recoverable."""
        return self.index.latest

    @property
    def committed_last_good_dir(self) -> str | None:
        entry = self.committed_last_good
        return None if entry is None else entry.checkpoint_dir

    @property
    def pending_steps(self) -> tuple[int, ...]:
        return tuple(sorted(entry.global_step for entry in self.pending.values()))

    def _next_commit_generation(self) -> int:
        generations = [entry.commit_generation for entry in self.index.entries]
        generations.extend(entry.commit_generation for entry in self.pending.values())
        return 0 if not generations else max(generations) + 1

    def _finalize_ids(self, ids: list[int] | tuple[int, ...]) -> list[str]:
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

    def register_committed(
        self,
        state: MCoreTrainingState,
        *,
        checkpoint_dir: str | Path,
        kind: str,
        durable: bool = True,
    ) -> RecoveryCheckpointEntry:
        """Publish an already synchronously committed checkpoint into the same generation order as async saves."""
        self.poll()
        target = Path(checkpoint_dir).resolve()
        if not target.is_dir() or not any(target.iterdir()):
            raise RuntimeError(f"cannot register missing or empty committed checkpoint: {target}")
        entry = RecoveryCheckpointEntry(
            global_step=state.global_step,
            commit_generation=self._next_commit_generation(),
            checkpoint_dir=target.as_posix(),
            kind=kind,
            durable=durable,
        )
        self.index, _ = record_recovery_checkpoint(self.root, self.index, entry)
        return entry

    def wait_for_committed_step(self, global_step: int) -> RecoveryCheckpointEntry:
        if global_step < 0:
            raise ValueError("global_step must be non-negative")
        self.poll()
        matches = [entry for entry in self.index.entries if entry.global_step == global_step]
        if matches:
            return max(matches, key=lambda entry: entry.commit_generation)
        if global_step not in self.pending_steps:
            raise RuntimeError(f"recovery checkpoint step {global_step} is neither committed nor pending")
        self.wait()
        matches = [entry for entry in self.index.entries if entry.global_step == global_step]
        if not matches:
            raise RuntimeError(f"async checkpoint step {global_step} finalized without an index entry")
        return max(matches, key=lambda entry: entry.commit_generation)

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
        self.poll()
        target = Path(checkpoint_dir).resolve()
        entry = RecoveryCheckpointEntry(
            global_step=state.global_step,
            commit_generation=self._next_commit_generation(),
            checkpoint_dir=target.as_posix(),
            kind=kind,
            durable=durable,
        )
        result = self.queue.save(model, optimizer, state, checkpoint_dir=target)
        self._finalize_ids(result.finalized_request_ids)
        request_id = result.request_id
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
