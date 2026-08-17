from __future__ import annotations

import json
from pathlib import Path

from koschei_sentinel.cyber_source_extractors import (
    extract_nvd_snapshot,
    extract_osv_snapshot,
    extract_rustsec_snapshot,
)


REVISION = "deadbeef"
DIGEST = "a" * 64


def test_rustsec_default_cc0_is_trainable(tmp_path: Path):
    advisory = tmp_path / "RUSTSEC-2026-0001.md"
    advisory.write_text(
        "```toml\n[advisory]\nid = \"RUSTSEC-2026-0001\"\npackage = \"demo\"\ndate = \"2026-01-01\"\n```\n\nA defensive advisory body.\n",
        encoding="utf-8",
    )
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
    advisory.write_text(
        "```toml\n[advisory]\nid = \"RUSTSEC-2026-0002\"\npackage = \"demo\"\ndate = \"2026-01-01\"\nlicense = \"CC-BY-4.0\"\nurl = \"https://github.com/advisories/GHSA-demo\"\n```\n\nImported advisory body.\n",
        encoding="utf-8",
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
    advisory.write_text(
        "```toml\n[advisory]\nid = \"RUSTSEC-2026-0003\"\npackage = \"demo\"\ndate = \"2026-01-01\"\nlicense = \"UNKNOWN\"\n```\n\nBody.\n",
        encoding="utf-8",
    )
    row = extract_rustsec_snapshot(
        tmp_path,
        source_id="rustsec.advisory.database",
        source_revision=REVISION,
        snapshot_digest=DIGEST,
    )[0]
    assert row.artifact.training_authorization is False
    assert row.artifact.license_status.value == "REVIEW_REQUIRED"


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
