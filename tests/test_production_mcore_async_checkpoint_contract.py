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
            checkpoint_dir=(tmp_path / "pending-9").as_posix(),
            kind="recovery",
            durable=False,
        )
    }

    class Index:
        latest = RecoveryCheckpointEntry(
            global_step=8,
            checkpoint_dir=(tmp_path / "committed-8").as_posix(),
            kind="recovery",
            durable=False,
        )

    manager.index = Index()
    assert manager.committed_last_good.global_step == 8
    assert manager.committed_last_good_dir.endswith("committed-8")
    assert manager.pending_steps == (9,)
