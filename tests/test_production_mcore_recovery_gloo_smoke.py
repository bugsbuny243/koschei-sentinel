from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from koschei_sentinel.production_mcore_checkpoint_retention import RecoveryCheckpointEntry, RecoveryCheckpointIndex
from koschei_sentinel.production_mcore_recovery_checkpoint_manager import RecoveryCheckpointManager


def _worker(rank: int, world_size: int, init_file: str, root: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        manager = object.__new__(RecoveryCheckpointManager)
        manager.root = Path(root).resolve()
        manager.keep_recovery_slots = 2
        manager.pending = {}
        manager.index = RecoveryCheckpointIndex(keep_recovery_slots=2, entries=[])

        target = manager.root / "checkpoint-4"
        target.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            (target / "rank0.marker").write_text("committed", encoding="utf-8")
        dist.barrier()

        candidate = RecoveryCheckpointEntry(
            global_step=4,
            commit_generation=0,
            checkpoint_dir=target.as_posix(),
            kind="recovery",
            durable=False,
        )
        agreed = manager._agree_entry(candidate)
        assert agreed.commit_generation == 0
        manager._publish_entry(agreed)
        assert manager.index.latest is not None
        assert manager.index.latest.global_step == 4
        assert manager.index.latest.commit_generation == 0
        assert manager.index.latest.checkpoint_dir == target.as_posix()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_two_process_gloo_recovery_agreement_and_publication(tmp_path: Path):
    if not dist.is_gloo_available():
        pytest.skip("Gloo backend unavailable")
    init_file = tmp_path / "gloo-init"
    root = tmp_path / "recovery"
    root.mkdir()
    mp.spawn(
        _worker,
        args=(2, init_file.as_posix(), root.as_posix()),
        nprocs=2,
        join=True,
    )
