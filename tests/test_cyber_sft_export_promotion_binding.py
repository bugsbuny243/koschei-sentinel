import json

from koschei_sentinel.cyber_sft_export_verify import verify_cyber_sft_export
from tests.test_cyber_sft_export_verify import _build_export, _write_json


def test_portable_export_rejects_manifest_promotion_eligibility_drift(
    tmp_path,
    monkeypatch,
) -> None:
    root = _build_export(tmp_path, monkeypatch)
    manifest_path = root / "run" / "adapter-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["corpus_promotion_eligible"] = True
    _write_json(manifest_path, manifest)

    report = verify_cyber_sft_export(root)

    assert report.valid is False
    assert any("promotion eligibility differs" in row for row in report.violations)
