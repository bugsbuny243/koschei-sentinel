from __future__ import annotations

import json
from pathlib import Path

from test_cyber_sft_export_verify import _build_export, _write_json

from koschei_sentinel.cyber_sft_export_verify import verify_cyber_sft_export


def test_portable_export_rejects_adapter_example_count_drift(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = _build_export(tmp_path, monkeypatch)
    manifest_path = root / "run" / "adapter-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["training_examples"] += 1
    _write_json(manifest_path, manifest)

    report = verify_cyber_sft_export(root)

    assert report.valid is False
    assert any(
        "TRAIN/VALIDATION example counts differ" in violation
        for violation in report.violations
    )
