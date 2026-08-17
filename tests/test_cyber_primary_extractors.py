from __future__ import annotations

import json

from koschei_sentinel.cyber_primary_extractors import (
    extract_attack_stix_snapshot,
    extract_kubernetes_security_snapshot,
    extract_yara_snapshot,
)


def test_attack_extractor_keeps_supported_non_revoked_objects(tmp_path):
    payload = {
        "type": "bundle",
        "objects": [
            {
                "type": "attack-pattern",
                "id": "attack-pattern--11111111-1111-4111-8111-111111111111",
                "name": "Credential Access Example",
                "description": "A defensive description of credential access behavior.",
                "x_mitre_platforms": ["Windows"],
            },
            {
                "type": "attack-pattern",
                "id": "attack-pattern--22222222-2222-4222-8222-222222222222",
                "name": "Deprecated Example",
                "description": "Should not be emitted.",
                "x_mitre_deprecated": True,
            },
        ],
    }
    (tmp_path / "enterprise-attack.json").write_text(json.dumps(payload), encoding="utf-8")
    rows = extract_attack_stix_snapshot(
        tmp_path,
        source_id="mitre.attack.knowledge",
        source_revision="v19.2",
        snapshot_digest="a" * 64,
    )
    assert len(rows) == 1
    assert rows[0].artifact.training_authorization is True
    assert rows[0].metadata["provider"] == "MITRE_ATTACK"


def test_kubernetes_extractor_only_keeps_english_security_scope(tmp_path):
    security = tmp_path / "content/en/docs/concepts/security"
    other = tmp_path / "content/en/docs/concepts/workloads"
    localized = tmp_path / "content/tr/docs/concepts/security"
    security.mkdir(parents=True)
    other.mkdir(parents=True)
    localized.mkdir(parents=True)
    (security / "security-checklist.md").write_text(
        "# Security checklist\nUse authentication, authorization and RBAC controls.",
        encoding="utf-8",
    )
    (other / "pods.md").write_text("# Pods\nGeneral workload material.", encoding="utf-8")
    (localized / "security.md").write_text(
        "# Security\nAuthentication guidance.", encoding="utf-8"
    )
    rows = extract_kubernetes_security_snapshot(
        tmp_path,
        source_id="kubernetes.security.docs",
        source_revision="2a031a5f0a9382d26c25cb1e58016001a21ce7b6",
        snapshot_digest="b" * 64,
    )
    assert len(rows) == 1
    assert rows[0].artifact.locator == "content/en/docs/concepts/security/security-checklist.md"


def test_yara_extractor_excludes_vendor_and_keeps_docs_and_engine(tmp_path):
    docs = tmp_path / "docs"
    engine = tmp_path / "libyara"
    vendor = tmp_path / "vendor/library"
    docs.mkdir(parents=True)
    engine.mkdir(parents=True)
    vendor.mkdir(parents=True)
    (docs / "writingrules.rst").write_text(
        "YARA rule documentation for defensive malware analysis.", encoding="utf-8"
    )
    (engine / "scanner.c").write_text(
        "/* first-party YARA scanning engine source for detection */\nint scan(void) { return 0; }",
        encoding="utf-8",
    )
    (vendor / "third.c").write_text(
        "/* third-party source must not be included */\nint x(void) { return 0; }",
        encoding="utf-8",
    )
    rows = extract_yara_snapshot(
        tmp_path,
        source_id="yara.official.rules.docs",
        source_revision="604822da04103d13812dbcb08f4d7d42b61f94a8",
        snapshot_digest="c" * 64,
    )
    assert {row.artifact.locator for row in rows} == {"docs/writingrules.rst", "libyara/scanner.c"}
