import hashlib
from pathlib import Path

from tests.test_cyber_sft_export_verify import _build_export


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_gold_bound_candidate_export(tmp_path, monkeypatch, release: Path) -> Path:
    fixture_root = tmp_path / "gold-bound-candidate"
    fixture_root.mkdir()
    return _build_export(
        fixture_root,
        monkeypatch,
        promotion_eligible=True,
        corpus_examples_raw=(release / "train" / "examples.jsonl").read_bytes(),
        corpus_manifest_raw=(release / "train" / "manifest.json").read_bytes(),
        validation_examples_sha256=_sha(release / "validation" / "examples.jsonl"),
        validation_manifest_sha256=_sha(release / "validation" / "manifest.json"),
    )
