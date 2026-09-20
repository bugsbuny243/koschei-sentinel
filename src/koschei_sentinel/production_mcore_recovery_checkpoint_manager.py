from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

    def _agree_entry(self, entry: RecoveryCheckpointEntry) -> RecoveryCheckpointEntry:
        """Assign rank-0 generation and validate logical checkpoint metadata in one collective phase."""
        try:
            import torch.distributed as dist
        except ImportError:
            return entry
        if not dist.is_initialized():
            return entry
        candidate = entry.model_dump(mode="json")
        gathered: list[Any] | None = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(candidate, gathered, dst=0)
        payload: list[Any] = [None]
        if dist.get_rank() == 0:
            try:
                if gathered is None or not gathered:
                    raise RuntimeError("recovery checkpoint agreement gathered no rank proposals")
                logical = ("global_step", "checkpoint_dir", "kind", "durable")
                first = gathered[0]
                if any(any(candidate[key] != first[key] for key in logical) for candidate in gathered[1:]):
                    raise RuntimeError("recovery checkpoint proposals diverged across ranks")
                first["commit_generation"] = self._next_commit_generation()
                payload[0] = {"ok": True, "entry": first}
            except Exception as exc:
                payload[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        dist.broadcast_object_list(payload, src=0)
        result = payload[0]
        if not isinstance(result, dict) or not result.get("ok"):
            detail = result.get("error") if isinstance(result, dict) else "invalid rank-0 agreement result"
            raise RuntimeError(f"recovery checkpoint agreement failed: {detail}")
        return RecoveryCheckpointEntry.model_validate(result["entry"])

    def _publish_entry(self, entry: RecoveryCheckpointEntry) -> list[str]:
        """All ranks participate; rank 0 alone mutates recovery metadata and retention."""
        try:
            import torch.distributed as dist
        except ImportError as exc:
            raise RuntimeError("PyTorch distributed is required for recovery checkpoint publication") from exc
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before recovery checkpoint publication")
        proposed: list[Any] = [entry.model_dump(mode="json")]
        gathered: list[Any] | None = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(proposed[0], gathered, dst=0)

        payload: list[Any] = [None]
        if dist.get_rank() == 0:
            try:
                if gathered is None or not gathered:
                    raise RuntimeError("recovery checkpoint publication gathered no rank proposals")
                first = gathered[0]
                if any(candidate != first for candidate in gathered[1:]):
                    raise RuntimeError("recovery checkpoint publication proposals diverged across ranks")
                agreed = RecoveryCheckpointEntry.model_validate(first)
                self.index, removed = record_recovery_checkpoint(self.root, self.index, agreed)
                payload[0] = {"ok": True, "removed": removed}
            except Exception as exc:
                payload[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        dist.broadcast_object_list(payload, src=0)
        result = payload[0]
        if not isinstance(result, dict) or not result.get("ok"):
            detail = result.get("error") if isinstance(result, dict) else "invalid rank-0 publication result"
            raise RuntimeError(f"recovery checkpoint publication failed: {detail}")

        self.index = load_recovery_checkpoint_index(self.root, keep_recovery_slots=self.keep_recovery_slots)
        return list(result.get("removed", []))

    def _finalize_ids(self, ids: list[int] | tuple[int, ...]) -> list[str]:
        """Translate rank-local queue IDs to logical entries, then publish in generation order."""
        finalized: list[RecoveryCheckpointEntry] = []
        for request_id in ids:
            entry = self.pending.pop(int(request_id), None)
            if entry is None:
                raise RuntimeError(f"async checkpoint finalized unknown request id {request_id}")
            finalized.append(entry)
        deleted: list[str] = []
        for entry in sorted(finalized, key=lambda item: item.commit_generation):
            deleted.extend(self._publish_entry(entry))
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
        entry = self._agree_entry(entry)
        self._publish_entry(entry)
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
        entry = self._agree_entry(entry)
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
