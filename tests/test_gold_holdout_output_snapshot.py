from pathlib import Path
from types import SimpleNamespace

import pytest

import koschei_sentinel.gold_holdout_output_snapshot as snapshot_module


def test_output_snapshot_copies_then_offline_verifies(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "output"
    source.mkdir()
    (source / "receipt.json").write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "snapshot"
    observed = None

    def fake_verify(output_dir, inference_pack_dir, candidate_export_dir):
        nonlocal observed
        observed = (
            Path(output_dir),
            Path(inference_pack_dir),
            Path(candidate_export_dir),
        )
        return SimpleNamespace(valid=True, violations=[])

    monkeypatch.setattr(
        snapshot_module,
        "verify_gold_holdout_inference_output",
        fake_verify,
    )

    snapshot, verification = snapshot_module.snapshot_verified_gold_holdout_inference_output(
        source,
        destination,
        inference_pack_dir="pack-snapshot",
        candidate_export_dir="candidate-snapshot",
    )

    assert snapshot == destination
    assert verification.valid is True
    assert observed == (
        destination,
        Path("pack-snapshot"),
        Path("candidate-snapshot"),
    )


def test_output_snapshot_verification_failure_removes_copy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "output"
    source.mkdir()
    (source / "receipt.json").write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "snapshot"

    monkeypatch.setattr(
        snapshot_module,
        "verify_gold_holdout_inference_output",
        lambda *_args, **_kwargs: SimpleNamespace(
            valid=False,
            violations=["receipt binding mismatch"],
        ),
    )

    with pytest.raises(ValueError, match="receipt binding mismatch"):
        snapshot_module.snapshot_verified_gold_holdout_inference_output(
            source,
            destination,
            inference_pack_dir="pack-snapshot",
            candidate_export_dir="candidate-snapshot",
        )

    assert not destination.exists()
