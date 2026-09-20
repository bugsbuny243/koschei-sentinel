from __future__ import annotations

from pathlib import Path

import pytest

from koschei_sentinel.production_mcore_async_checkpoint import AsyncCheckpointScheduleResult
from koschei_sentinel.production_mcore_checkpoint_retention import RecoveryCheckpointEntry
from koschei_sentinel.production_mcore_recovery_checkpoint_manager import RecoveryCheckpointManager


def test_schedule_result_preserves_backpressure_finalizations():
    result = AsyncCheckpointScheduleResult(request_id=7, finalized_request_ids=(3, 5))
    assert result.request_id == 7
    assert result.finalized_request_ids == (3, 5)


def test_committed_last_good_never_reads_pending_entry(tmp_path: Path):
    manager = object.__new__(RecoveryCheckpointManager)
    manager.root = tmp_path
    manager.pending = {
        9: RecoveryCheckpointEntry(
            global_step=9,
            commit_generation=9,
            checkpoint_dir=(tmp_path / "pending-9").as_posix(),
            kind="recovery",
            durable=False,
        )
    }

    class Index:
        latest = RecoveryCheckpointEntry(
            global_step=8,
            commit_generation=8,
            checkpoint_dir=(tmp_path / "committed-8").as_posix(),
            kind="recovery",
            durable=False,
        )

    manager.index = Index()
    assert manager.committed_last_good.global_step == 8
    assert manager.committed_last_good_dir.endswith("committed-8")
    assert manager.pending_steps == (9,)


def test_committed_last_good_prefers_newer_generation_at_same_step(tmp_path: Path):
    manager = object.__new__(RecoveryCheckpointManager)
    manager.root = tmp_path
    manager.pending = {}
    older = RecoveryCheckpointEntry(global_step=8, commit_generation=10, checkpoint_dir=(tmp_path / "older").as_posix(), kind="recovery", durable=False)
    newer = RecoveryCheckpointEntry(global_step=8, commit_generation=11, checkpoint_dir=(tmp_path / "skip").as_posix(), kind="skip", durable=True)

    class Index:
        entries = [older, newer]
        latest = newer

    manager.index = Index()
    assert manager.committed_last_good.commit_generation == 11
    assert manager.committed_last_good.kind == "skip"


def test_next_generation_accounts_for_pending_async_entry(tmp_path: Path):
    manager = object.__new__(RecoveryCheckpointManager)
    manager.root = tmp_path
    committed = RecoveryCheckpointEntry(global_step=4, commit_generation=20, checkpoint_dir=(tmp_path / "committed").as_posix(), kind="recovery", durable=False)
    pending = RecoveryCheckpointEntry(global_step=5, commit_generation=22, checkpoint_dir=(tmp_path / "pending").as_posix(), kind="recovery", durable=False)

    class Index:
        entries = [committed]
        latest = committed

    manager.index = Index()
    manager.pending = {7: pending}
    assert manager._next_commit_generation() == 23


class _FakeDist:
    def __init__(self, *, gathered=None, broadcast_result=None):
        self.gathered = gathered
        self.broadcast_result = broadcast_result
        self.broadcast_calls = 0

    def is_initialized(self): return True
    def get_rank(self): return 0
    def get_world_size(self): return 2
    def gather_object(self, value, output, dst=0):
        assert dst == 0
        values = self.gathered if self.gathered is not None else [value, value]
        output[:] = values
    def broadcast_object_list(self, payload, src=0):
        assert src == 0
        self.broadcast_calls += 1
        if self.broadcast_result is not None:
            payload[0] = self.broadcast_result


def _manager_for_publication(tmp_path: Path):
    manager = object.__new__(RecoveryCheckpointManager)
    manager.root = tmp_path.resolve()
    manager.keep_recovery_slots = 2
    manager.pending = {}
    return manager


def test_publication_rejects_divergent_rank_metadata(tmp_path: Path, monkeypatch):
    import torch.distributed as dist
    from koschei_sentinel.production_mcore_checkpoint_retention import RecoveryCheckpointIndex

    manager = _manager_for_publication(tmp_path)
    manager.index = RecoveryCheckpointIndex(keep_recovery_slots=2, entries=[])
    entry = RecoveryCheckpointEntry(global_step=8, commit_generation=1, checkpoint_dir=(tmp_path / "rank0").as_posix(), kind="recovery", durable=False)
    other = entry.model_copy(update={"checkpoint_dir": (tmp_path / "rank1").as_posix()})
    fake = _FakeDist(gathered=[entry.model_dump(mode="json"), other.model_dump(mode="json")])
    monkeypatch.setattr(dist, "is_initialized", fake.is_initialized)
    monkeypatch.setattr(dist, "get_rank", fake.get_rank)
    monkeypatch.setattr(dist, "get_world_size", fake.get_world_size)
    monkeypatch.setattr(dist, "gather_object", fake.gather_object)
    monkeypatch.setattr(dist, "broadcast_object_list", fake.broadcast_object_list)

    with pytest.raises(RuntimeError, match="proposals diverged across ranks"):
        manager._publish_entry(entry)
    assert fake.broadcast_calls == 1


def test_publication_propagates_rank_zero_failure(tmp_path: Path, monkeypatch):
    import torch.distributed as dist
    import koschei_sentinel.production_mcore_recovery_checkpoint_manager as module
    from koschei_sentinel.production_mcore_checkpoint_retention import RecoveryCheckpointIndex

    manager = _manager_for_publication(tmp_path)
    manager.index = RecoveryCheckpointIndex(keep_recovery_slots=2, entries=[])
    entry = RecoveryCheckpointEntry(global_step=8, commit_generation=1, checkpoint_dir=(tmp_path / "rank0").as_posix(), kind="recovery", durable=False)
    fake = _FakeDist()
    monkeypatch.setattr(dist, "is_initialized", fake.is_initialized)
    monkeypatch.setattr(dist, "get_rank", fake.get_rank)
    monkeypatch.setattr(dist, "get_world_size", fake.get_world_size)
    monkeypatch.setattr(dist, "gather_object", fake.gather_object)
    monkeypatch.setattr(dist, "broadcast_object_list", fake.broadcast_object_list)
    monkeypatch.setattr(module, "record_recovery_checkpoint", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(RuntimeError, match="OSError: disk full"):
        manager._publish_entry(entry)
    assert fake.broadcast_calls == 1


def test_finalize_ids_uses_generation_order_not_request_id_order(tmp_path: Path, monkeypatch):
    manager = object.__new__(RecoveryCheckpointManager)
    manager.root = tmp_path
    older = RecoveryCheckpointEntry(global_step=4, commit_generation=30, checkpoint_dir=(tmp_path / "older").as_posix(), kind="recovery", durable=False)
    newer = RecoveryCheckpointEntry(global_step=5, commit_generation=31, checkpoint_dir=(tmp_path / "newer").as_posix(), kind="recovery", durable=False)
    manager.pending = {900: newer, 12: older}
    published = []
    monkeypatch.setattr(manager, "_publish_entry", lambda entry: published.append(entry.commit_generation) or [])

    manager._finalize_ids((900, 12))
    assert published == [30, 31]
    assert manager.pending == {}
