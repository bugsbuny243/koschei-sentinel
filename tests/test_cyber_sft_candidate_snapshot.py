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
    assert observed == [source, destination]
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


def test_candidate_snapshot_rejects_source_mutation_during_verification(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidate-export"
    source.mkdir()
    config = source / "training-config.json"
    config.write_text("before\n", encoding="utf-8")
    destination = tmp_path / "snapshot"

    def mutate_source(path):
        assert Path(path) == source
        config.write_text("after\n", encoding="utf-8")
        return SimpleNamespace(valid=True, violations=[])

    monkeypatch.setattr(snapshot_module, "verify_cyber_sft_export", mutate_source)

    with pytest.raises(ValueError, match="source changed while it was being verified"):
        snapshot_module.snapshot_verified_cyber_sft_export(source, destination)

    assert not destination.exists()


def test_candidate_snapshot_rejects_copy_that_differs_from_verified_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidate-export"
    source.mkdir()
    (source / "training-config.json").write_text("source\n", encoding="utf-8")
    destination = tmp_path / "snapshot"

    def mutate_snapshot(path):
        path = Path(path)
        if path == destination:
            (path / "training-config.json").write_text("mutated\n", encoding="utf-8")
        return SimpleNamespace(valid=True, violations=[])

    monkeypatch.setattr(snapshot_module, "verify_cyber_sft_export", mutate_snapshot)

    with pytest.raises(ValueError, match="snapshot bytes differ from verified source"):
        snapshot_module.snapshot_verified_cyber_sft_export(source, destination)

    assert not destination.exists()
