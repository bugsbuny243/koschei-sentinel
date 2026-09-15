from __future__ import annotations

from pathlib import Path

import pytest

from koschei_sentinel.production_mcore_checkpoint_retention import (
    RecoveryCheckpointEntry,
    RecoveryCheckpointIndex,
    load_recovery_checkpoint_index,
    record_recovery_checkpoint,
)


def _entry(path: Path, step: int, kind: str = "recovery", durable: bool = False):
    path.mkdir(parents=True, exist_ok=True)
    (path / "marker").write_text(str(step), encoding="utf-8")
    return RecoveryCheckpointEntry(
        global_step=step,
        checkpoint_dir=path.as_posix(),
        kind=kind,
        durable=durable,
    )


def test_retention_keeps_two_latest_recovery_slots(tmp_path: Path):
    index = RecoveryCheckpointIndex(keep_recovery_slots=2)
    paths = [tmp_path / f"recovery-step-{step:08d}" for step in (1, 2, 3)]
    for step, path in enumerate(paths, start=1):
        index, _ = record_recovery_checkpoint(tmp_path, index, _entry(path, step))
    assert paths[0].exists() is False
    assert paths[1].is_dir()
    assert paths[2].is_dir()
    assert [item.global_step for item in index.entries if item.kind == "recovery"] == [2, 3]


def test_periodic_checkpoint_is_not_pruned_as_recovery_slot(tmp_path: Path):
    index = RecoveryCheckpointIndex(keep_recovery_slots=2)
    periodic = tmp_path / "step-00000001"
    index, _ = record_recovery_checkpoint(tmp_path, index, _entry(periodic, 1, kind="periodic", durable=True))
    for step in (2, 3, 4):
        index, _ = record_recovery_checkpoint(
            tmp_path,
            index,
            _entry(tmp_path / f"recovery-step-{step:08d}", step),
        )
    assert periodic.is_dir()
    assert any(item.kind == "periodic" and item.global_step == 1 for item in index.entries)


def test_retention_refuses_checkpoint_outside_root(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="outside checkpoint root"):
        record_recovery_checkpoint(root, RecoveryCheckpointIndex(), _entry(outside, 1))


def test_index_roundtrip_and_retention_setting_are_bound(tmp_path: Path):
    index = RecoveryCheckpointIndex(keep_recovery_slots=3)
    index, _ = record_recovery_checkpoint(
        tmp_path,
        index,
        _entry(tmp_path / "recovery-step-00000001", 1),
    )
    loaded = load_recovery_checkpoint_index(tmp_path, keep_recovery_slots=3)
    assert loaded.model_dump(mode="json") == index.model_dump(mode="json")
    with pytest.raises(ValueError, match="retention setting changed"):
        load_recovery_checkpoint_index(tmp_path, keep_recovery_slots=2)
