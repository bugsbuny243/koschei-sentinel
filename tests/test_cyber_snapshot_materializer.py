from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from koschei_sentinel.cyber_corpus_catalog import CyberSource
from koschei_sentinel.cyber_snapshot_materializer import (
    _verify_git_revision,
    materialize_snapshot,
)


def _git(path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "sentinel@example.invalid")
    _git(root, "config", "user.name", "Sentinel Test")
    (root / "README.md").write_text("security\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "commit", "-m", "first")
    head = _git(root, "rev-parse", "HEAD")
    return root, head


def test_revision_verifier_accepts_exact_commit(tmp_path: Path):
    root, head = _repo(tmp_path)
    observed, verified = _verify_git_revision(root, head)
    assert observed == head
    assert verified is True


def test_revision_verifier_accepts_tag_pointing_to_head(tmp_path: Path):
    root, head = _repo(tmp_path)
    _git(root, "tag", "v-test")
    observed, verified = _verify_git_revision(root, "v-test")
    assert observed == head
    assert verified is True


def test_revision_verifier_rejects_checkout_moved_past_tag(tmp_path: Path):
    root, _ = _repo(tmp_path)
    _git(root, "tag", "v-test")
    (root / "SECOND.md").write_text("new\n", encoding="utf-8")
    _git(root, "add", "SECOND.md")
    _git(root, "commit", "-m", "second")
    _, verified = _verify_git_revision(root, "v-test")
    assert verified is False


def test_materializer_rejects_wrong_pinned_revision_before_extraction(tmp_path: Path):
    root, head = _repo(tmp_path)
    _git(root, "tag", "approved")
    (root / "SECOND.md").write_text("new\n", encoding="utf-8")
    _git(root, "add", "SECOND.md")
    _git(root, "commit", "-m", "second")

    source = CyberSource.model_validate(
        {
            "source_id": "yara.official.rules.docs",
            "source_class": "MALWARE_ANALYSIS",
            "domain_families": ["MALWARE_REVERSE_ENGINEERING"],
            "provenance_tier": "T0_PRIMARY",
            "license_status": "ALLOW_WITH_ATTRIBUTION",
            "license_scope": "UNIFORM",
            "license_reference": "license",
            "acquisition_mode": "PINNED_GIT",
            "canonical_locator": "https://example.invalid/yara",
            "pinned_revision": "approved",
            "training_authorization": True,
            "eval_exclusion": True,
            "benchmark_overlap_risk": "LOW",
            "review_status": "APPROVED",
        }
    )

    with pytest.raises(ValueError, match="snapshot git revision mismatch"):
        materialize_snapshot(source=source, snapshot_path=root, output_root=tmp_path / "out")

    assert head != _git(root, "rev-parse", "HEAD")
