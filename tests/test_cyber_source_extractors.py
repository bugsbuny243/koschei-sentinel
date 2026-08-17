from __future__ import annotations

import json
from pathlib import Path

from koschei_sentinel.cyber_corpus_catalog import audit_artifacts, load_catalog
from koschei_sentinel.cyber_source_extractors import (
    extract_nvd_snapshot,
    extract_osv_snapshot,
    extract_rustsec_snapshot,
)


REVISION = "deadbeef"
RUSTSEC_APPROVED_REVISION = "69f93e1d081d8b6fbee010e48f0b5e0d13661415"
DIGEST = "a" * 64


def _write_rustsec(
    path: Path,
    advisory_id: str,
    *,
    license_name: str | None = None,
    url: str | None = None,
) -> None:
    optional = ""
    if license_name:
        optional += f'license = "{license_name}"\n'
    if url:
        optional += f'url = "{url}"\n'
    path.write_text(
        "```toml\n"
        "[advisory]\n"
        f'id = "{advisory_id}"\n'
        'package = "demo"\n'
        'date = "2026-01-01"\n'
        f"{optional}"
        "```\n\n"
        "A defensive advisory body.\n",
        encoding="utf-8",
    )


def test_rustsec_default_cc0_is_trainable(tmp_path: Path):
    advisory = tmp_path / "RUSTSEC-2026-0001.md"
    _write_rustsec(advisory, "RUSTSEC-2026-0001")
    row = extract_rustsec_snapshot(
        tmp_path,
        source_id="rustsec.advisory.database",
        source_revision=REVISION,
        snapshot_digest=DIGEST,
    )[0]
    assert row.artifact.training_authorization is True
    assert row.artifact.license_status.value == "ALLOW_TRAINING"
    assert row.metadata["declared_license"] == "CC0-1.0"


def test_rustsec_cc_by_requires_and_keeps_attribution(tmp_path: Path):
    advisory = tmp_path / "RUSTSEC-2026-0002.md"
    _write_rustsec(
        advisory,
        "RUSTSEC-2026-0002",
        license_name="CC-BY-4.0",
        url="https://github.com/advisories/GHSA-demo",
    )
    row = extract_rustsec_snapshot(
        tmp_path,
        source_id="rustsec.advisory.database",
        source_revision=REVISION,
        snapshot_digest=DIGEST,
    )[0]
    assert row.artifact.training_authorization is True
    assert row.artifact.license_status.value == "ALLOW_WITH_ATTRIBUTION"
    assert row.artifact.license_reference == "https://github.com/advisories/GHSA-demo"


def test_rustsec_unknown_license_is_fail_closed(tmp_path: Path):
    advisory = tmp_path / "RUSTSEC-2026-0003.md"
    _write_rustsec(advisory, "RUSTSEC-2026-0003", license_name="UNKNOWN")
    row = extract_rustsec_snapshot(
        tmp_path,
        source_id="rustsec.advisory.database",
        source_revision=REVISION,
        snapshot_digest=DIGEST,
    )[0]
    assert row.artifact.training_authorization is False
    assert row.artifact.license_status.value == "REVIEW_REQUIRED"


def test_rustsec_extractor_output_passes_approved_source_audit(tmp_path: Path):
    advisory = tmp_path / "RUSTSEC-2026-0004.md"
    _write_rustsec(advisory, "RUSTSEC-2026-0004")
    row = extract_rustsec_snapshot(
        tmp_path,
        source_id="rustsec.advisory.database",
        source_revision=RUSTSEC_APPROVED_REVISION,
        snapshot_digest=DIGEST,
    )[0]
    sources = load_catalog("configs/corpus/cyber-v3.sources.approved.jsonl")
    result = audit_artifacts([row.artifact], sources)
    assert result.ready_for_ingestion is True
    assert result.training_authorized_artifacts == 1
    assert result.violations == []


def test_osv_without_explicit_artifact_license_is_not_trainable(tmp_path: Path):
    payload = {
        "id": "OSV-2026-1",
        "summary": "demo",
        "details": "defensive vulnerability record",
    }
    path = tmp_path / "osv.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    row = extract_osv_snapshot(
        path,
        source_id="osv.vulnerability.database",
        source_revision=REVISION,
        snapshot_digest=DIGEST,
    )[0]
    assert row.artifact.training_authorization is False
    assert row.artifact.license_status.value == "REVIEW_REQUIRED"


def test_nvd_is_extracted_but_not_auto_authorized(tmp_path: Path):
    payload = {
        "vulnerabilities": [
            {
                "cve": {
                    "id": "CVE-2026-0001",
                    "descriptions": [{"lang": "en", "value": "demo description"}],
                }
            }
        ]
    }
    path = tmp_path / "nvd.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    row = extract_nvd_snapshot(
        path,
        source_id="nvd.vulnerability.records",
        source_revision=REVISION,
        snapshot_digest=DIGEST,
    )[0]
    assert row.artifact.training_authorization is False
    assert row.artifact.license_status.value == "REVIEW_REQUIRED"
    assert row.metadata["provider"] == "NVD"
