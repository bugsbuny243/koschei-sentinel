from pathlib import Path

from koschei_sentinel.cyber_sft_export_verify import verify_cyber_sft_export


def test_missing_portable_export_artifacts_fail_closed(tmp_path: Path) -> None:
    export = tmp_path / "export"
    export.mkdir()

    report = verify_cyber_sft_export(export)

    assert report.valid is False
    assert report.run_id is None
    assert report.violations
    assert report.violations[0].startswith("missing export files:")
    assert "attestation" in report.violations[0]
    assert "corpus_examples" in report.violations[0]
    assert report.attestation_sha256_verified is False
    assert report.fresh_run_verification_valid is False
