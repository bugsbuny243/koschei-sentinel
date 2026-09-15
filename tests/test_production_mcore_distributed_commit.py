from pathlib import Path
import pytest
from koschei_sentinel.production_mcore_distributed_commit import DistributedCommitResult


def test_distributed_commit_result_marks_ready_checkpoint(tmp_path: Path):
    result=DistributedCommitResult(checkpoint_dir=(tmp_path/"checkpoint").as_posix(),all_ranks_ready=True)
    assert result.all_ranks_ready is True
    assert result.checkpoint_dir.endswith("checkpoint")


def test_commit_result_is_immutable(tmp_path: Path):
    result=DistributedCommitResult(checkpoint_dir=tmp_path.as_posix(),all_ranks_ready=True)
    with pytest.raises(Exception):
        result.all_ranks_ready=False
