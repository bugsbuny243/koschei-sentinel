from __future__ import annotations

from pathlib import Path

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
