from pathlib import Path
from types import SimpleNamespace

import pytest

import koschei_sentinel.gold_release_snapshot as snapshot_module


def test_gold_release_snapshot_binds_structural_and_signed_review_audits(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "release"
    source.mkdir()
    (source / "release-manifest.json").write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "snapshot"

    monkeypatch.setattr(
        snapshot_module,
        "audit_gold_defense_release",
        lambda path: SimpleNamespace(
            valid=True,
            violations=[],
            audit_sha256="a" * 64,
        ),
    )
    monkeypatch.setattr(
        snapshot_module,
        "audit_gold_release_review_signatures",
        lambda path, key: SimpleNamespace(
            valid=True,
            violations=[],
            audit_sha256="b" * 64,
        ),
    )

    snapshot, verification = snapshot_module.snapshot_verified_gold_release(
        source,
        destination,
        reviewer_public_key=object(),
        expected_release_audit_sha256="a" * 64,
        expected_review_signature_audit_sha256="b" * 64,
    )

    assert snapshot == destination
    assert verification.release_audit.audit_sha256 == "a" * 64
    assert verification.review_signature_audit.audit_sha256 == "b" * 64


def test_gold_release_snapshot_digest_mismatch_removes_copy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "release"
    source.mkdir()
    (source / "release-manifest.json").write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "snapshot"

    monkeypatch.setattr(
        snapshot_module,
        "audit_gold_defense_release",
        lambda path: SimpleNamespace(
            valid=True,
            violations=[],
            audit_sha256="f" * 64,
        ),
    )

    with pytest.raises(ValueError, match="different release audit"):
        snapshot_module.snapshot_verified_gold_release(
            source,
            destination,
            reviewer_public_key=object(),
            expected_release_audit_sha256="a" * 64,
            expected_review_signature_audit_sha256="b" * 64,
        )

    assert not destination.exists()
