from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel


class RecoveryCheckpointEntry(StrictModel):
    schema_version: Literal["sentinel.recovery-checkpoint-entry.v1"] = "sentinel.recovery-checkpoint-entry.v1"
    global_step: int = Field(ge=0)
    checkpoint_dir: str = Field(min_length=1)
    kind: Literal["base", "recovery", "periodic", "quarantine", "skip"]
    durable: bool = False


class RecoveryCheckpointIndex(StrictModel):
    schema_version: Literal["sentinel.recovery-checkpoint-index.v1"] = "sentinel.recovery-checkpoint-index.v1"
    keep_recovery_slots: int = Field(default=2, ge=2, le=16)
    entries: list[RecoveryCheckpointEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent(self) -> "RecoveryCheckpointIndex":
        recovery_steps = [item.global_step for item in self.entries if item.kind == "recovery"]
        if len(recovery_steps) != len(set(recovery_steps)):
            raise ValueError("recovery checkpoint index contains duplicate recovery steps")
        return self

    @property
    def latest(self) -> RecoveryCheckpointEntry | None:
        return max(self.entries, key=lambda item: item.global_step, default=None)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_recovery_checkpoint_index(root: str | Path, *, keep_recovery_slots: int = 2) -> RecoveryCheckpointIndex:
    path = Path(root) / "recovery-index.json"
    if not path.exists():
        return RecoveryCheckpointIndex(keep_recovery_slots=keep_recovery_slots)
    raw = json.loads(path.read_text(encoding="utf-8"))
    index = RecoveryCheckpointIndex.model_validate(raw)
    if index.keep_recovery_slots != keep_recovery_slots:
        raise ValueError("recovery checkpoint retention setting changed across resume")
    return index


def record_recovery_checkpoint(
    root: str | Path,
    index: RecoveryCheckpointIndex,
    entry: RecoveryCheckpointEntry,
) -> tuple[RecoveryCheckpointIndex, list[str]]:
    root_path = Path(root).resolve()
    candidate = Path(entry.checkpoint_dir).resolve()
    if candidate != root_path and root_path not in candidate.parents:
        raise ValueError("checkpoint retention refuses path outside checkpoint root")

    entries = [item for item in index.entries if item.checkpoint_dir != entry.checkpoint_dir]
    entries.append(entry)
    recovery = sorted((item for item in entries if item.kind == "recovery"), key=lambda item: item.global_step)
    removable = recovery[:-index.keep_recovery_slots]
    removable_paths = {item.checkpoint_dir for item in removable if not item.durable}
    kept = [item for item in entries if item.checkpoint_dir not in removable_paths]
    updated = RecoveryCheckpointIndex(keep_recovery_slots=index.keep_recovery_slots, entries=kept)

    # Publish the new index before deleting old slots. A crash can leave extra old data,
    # but never an index pointing at a checkpoint deleted by this function.
    _atomic_write_json(root_path / "recovery-index.json", updated.model_dump(mode="json"))
    deleted: list[str] = []
    for raw_path in sorted(removable_paths):
        path = Path(raw_path).resolve()
        if path == root_path or root_path not in path.parents:
            raise ValueError("checkpoint retention refuses deletion outside checkpoint root")
        if path.is_dir():
            shutil.rmtree(path)
            deleted.append(path.as_posix())
    return updated, deleted
