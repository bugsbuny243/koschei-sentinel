from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel


class RecoveryCheckpointEntry(StrictModel):
    schema_version: Literal["sentinel.recovery-checkpoint-entry.v2"] = "sentinel.recovery-checkpoint-entry.v2"
    global_step: int = Field(ge=0)
    commit_generation: int = Field(ge=0)
    checkpoint_dir: str = Field(min_length=1)
    kind: Literal["base", "recovery", "periodic", "quarantine", "skip"]
    durable: bool = False


class RecoveryCheckpointIndex(StrictModel):
    schema_version: Literal["sentinel.recovery-checkpoint-index.v2"] = "sentinel.recovery-checkpoint-index.v2"
    keep_recovery_slots: int = Field(default=2, ge=2, le=16)
    entries: list[RecoveryCheckpointEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent(self) -> "RecoveryCheckpointIndex":
        generations = [item.commit_generation for item in self.entries]
        if len(generations) != len(set(generations)):
            raise ValueError("recovery checkpoint index contains duplicate commit generations")
        recovery_generations = [item.commit_generation for item in self.entries if item.kind == "recovery"]
        if len(recovery_generations) != len(set(recovery_generations)):
            raise ValueError("recovery checkpoint index contains duplicate recovery generations")
        return self

    @property
    def latest(self) -> RecoveryCheckpointEntry | None:
        return max(self.entries, key=lambda item: item.commit_generation, default=None)

    @property
    def next_commit_generation(self) -> int:
        latest = self.latest
        return 0 if latest is None else latest.commit_generation + 1


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        # Make the directory entry replacement durable where the platform supports it.
        try:
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass
    finally:
        if tmp.exists():
            tmp.unlink()


def _migrate_v1(raw: dict) -> dict:
    if raw.get("schema_version") != "sentinel.recovery-checkpoint-index.v1":
        return raw
    entries = raw.get("entries", [])
    # v1 had no ordering independent of global_step. Preserve its observable ordering
    # deterministically, using step first and original list order as the tie breaker.
    ordered = sorted(enumerate(entries), key=lambda pair: (int(pair[1]["global_step"]), pair[0]))
    generation_by_index = {original_index: generation for generation, (original_index, _) in enumerate(ordered)}
    migrated = []
    for index, entry in enumerate(entries):
        item = dict(entry)
        item["schema_version"] = "sentinel.recovery-checkpoint-entry.v2"
        item["commit_generation"] = generation_by_index[index]
        migrated.append(item)
    return {
        "schema_version": "sentinel.recovery-checkpoint-index.v2",
        "keep_recovery_slots": raw.get("keep_recovery_slots", 2),
        "entries": migrated,
    }


def load_recovery_checkpoint_index(root: str | Path, *, keep_recovery_slots: int = 2) -> RecoveryCheckpointIndex:
    path = Path(root) / "recovery-index.json"
    if not path.exists():
        return RecoveryCheckpointIndex(keep_recovery_slots=keep_recovery_slots)
    raw = _migrate_v1(json.loads(path.read_text(encoding="utf-8")))
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
    if any(item.commit_generation == entry.commit_generation and item.checkpoint_dir != entry.checkpoint_dir for item in index.entries):
        raise ValueError("checkpoint retention refuses duplicate commit generation")

    entries = [item for item in index.entries if item.checkpoint_dir != entry.checkpoint_dir]
    entries.append(entry)
    recovery = sorted((item for item in entries if item.kind == "recovery"), key=lambda item: item.commit_generation)
    removable = recovery[:-index.keep_recovery_slots]
    removable_paths = {item.checkpoint_dir for item in removable if not item.durable}
    kept = [item for item in entries if item.checkpoint_dir not in removable_paths]
    updated = RecoveryCheckpointIndex(keep_recovery_slots=index.keep_recovery_slots, entries=kept)

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
