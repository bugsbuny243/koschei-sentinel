from pathlib import Path
from types import SimpleNamespace

import pytest

import koschei_sentinel.cyber_sft_candidate_snapshot as snapshot_module


def test_candidate_snapshot_copies_then_reverifies(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "candidate-export"
    source.mkdir()
    (source / "training-config.json").write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "snapshot"
    observed: list[Path] = []

    def fake_verify(path):
        observed.append(Path(path))
        return SimpleNamespace(valid=True, violations=[])

    monkeypatch.setattr(snapshot_module, "verify_cyber_sft_export", fake_verify)

    snapshot = snapshot_module.snapshot_verified_cyber_sft_export(
        source,
        destination,
    )

    assert snapshot == destination
    assert observed == [destination]
    assert (snapshot / "training-config.json").read_text(encoding="utf-8") == "{}\n"


def test_candidate_snapshot_failure_removes_untrusted_copy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidate-export"
    source.mkdir()
    (source / "training-config.json").write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "snapshot"

    monkeypatch.setattr(
        snapshot_module,
        "verify_cyber_sft_export",
        lambda *_args: SimpleNamespace(
            valid=False,
            violations=["adapter digest does not verify"],
        ),
    )

    with pytest.raises(ValueError, match="adapter digest does not verify"):
        snapshot_module.snapshot_verified_cyber_sft_export(source, destination)

    assert not destination.exists()
