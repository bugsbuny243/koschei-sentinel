from __future__ import annotations

import json
from pathlib import Path

from koschei_sentinel.cyber_sft_export_verify import verify_cyber_sft_export
from koschei_sentinel.cyber_sft_run_attestation import _attestation_digest
from koschei_sentinel.cyber_sft_training_source import _source_digest
from tests.test_cyber_sft_export_verify import _build_export, _sha, _write_json


def test_rehashed_forged_training_source_still_fails_export_semantics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = _build_export(tmp_path, monkeypatch)

    source_path = root / "training-source.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["repository_commit"] = "0" * 40
    source["source_binding_sha256"] = _source_digest(source)
    source_raw = _write_json(source_path, source)

    attestation_path = root / "run-attestation.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["training_source_sha256"] = _sha(source_raw)
    attestation["attestation_sha256"] = _attestation_digest(attestation)
    _write_json(attestation_path, attestation)

    report = verify_cyber_sft_export(root)

    assert report.valid is False
    assert report.training_source_sha256_verified is True
    assert report.repository_commit_binding_verified is True
    assert any("training source semantic bindings" in row for row in report.violations)
