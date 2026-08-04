from __future__ import annotations

import json

import pytest

from koschei_sentinel.dataset import export_record
from koschei_sentinel.neon_cli import main as neon_main
from koschei_sentinel.neon_source import (
    _REQUIRED_COLUMNS,
    _validate_schema,
    read_neon_records,
    records_from_neon_rows,
)

SALT = "unit-test-salt-with-minimum-length"
TARGET = "8KxV7JGnVKXHh4UauuMxoPjzD4hnvy9dPp7Z4hXqW2aB"


def source_row() -> dict:
    return {
        "id": "4bd2e5b7-a048-4502-8dc2-301868810834",
        "network": "solana-mainnet",
        "target_kind": "token",
        "target_id": TARGET,
        "grade": "D",
        "verdict": "A deterministic holder pressure condition was observed.",
        "signature": "signed-unified-verdict-0001",
        "fingerprint": "fingerprint-unified-verdict-0001",
        "triggered_rules": [
            {
                "rule_id": "URD-C002",
                "title": "Dominant-holder position",
                "summary": "The bounded position exceeded the configured pressure rule.",
                "evidence_status": "verified",
                "evidence_keys": ["holder-liquidity:private-source-key"],
                "grade_effect": "cap_d",
            }
        ],
        "updated_at": "2026-08-04T00:22:34Z",
        "modules": [
            {
                "id": "bf3a75e9-678f-4fca-aec1-14093eb93b18",
                "module_id": "holder_concentration",
                "signature": "signed-holder-module-0001",
                "risk_index": 82,
                "risk_level": "high",
                "grade": "D",
                "verdict": "Holder concentration was material.",
                "recommendation": "Review the bounded holder evidence.",
                "evidence": ["The largest holder controlled a material supply share."],
                "signals": {"evidence_status": "verified"},
                "rule_version": "holder-v1",
            },
            {
                "id": "386805d7-b167-443d-89e4-9b30c6f39caf",
                "module_id": "token_authority_scanner",
                "signature": "signed-authority-module-0001",
                "risk_index": 20,
                "risk_level": "low",
                "grade": "B",
                "verdict": "Authority controls were bounded.",
                "evidence": [],
                "signals": {},
                "rule_version": "authority-v1",
            },
        ],
        "holder_count": 10,
        "top_percentage": "42.5",
        "holder_scanned_at": "2026-08-04T00:20:00Z",
    }


def test_neon_rows_become_private_arvis_records() -> None:
    records, stats = records_from_neon_rows([source_row()])

    assert stats.selected_targets == 1
    assert stats.rule_evidence_items == 1
    assert stats.module_evidence_items == 2
    assert stats.holder_evidence_items == 1
    assert stats.fallback_evidence_items == 0

    raw = records[0]
    assert raw["signed_verdict"]["triggered_rules"] == ["URD-C002"]
    assert {item["kind"] for item in raw["evidence"]} == {
        "unified_rule",
        "holder_concentration",
        "token_authority_scanner",
        "holder_snapshot",
    }

    private = export_record(raw, salt=SALT)
    serialized = private.model_dump_json()
    assert TARGET not in serialized
    assert private.case.target_ref.startswith("target_")
    assert private.case.evidence[0].evidence_id.startswith("evidence_")


def test_neon_json_strings_are_supported() -> None:
    row = source_row()
    row["triggered_rules"] = json.dumps(row["triggered_rules"])
    row["modules"] = json.dumps(row["modules"])

    records, stats = records_from_neon_rows([row])

    assert len(records[0]["evidence"]) == 4
    assert stats.emitted_records == 1


def test_neon_schema_gate_rejects_missing_columns() -> None:
    complete = [
        {"table_name": table, "column_name": column}
        for table, columns in _REQUIRED_COLUMNS.items()
        for column in columns
    ]
    _validate_schema(complete)

    incomplete = [item for item in complete if item["column_name"] != "triggered_rules"]
    with pytest.raises(ValueError, match="triggered_rules"):
        _validate_schema(incomplete)


def test_neon_limit_is_checked_before_loading_driver() -> None:
    with pytest.raises(ValueError, match="between 1 and 10000"):
        read_neon_records("postgresql://example.invalid/db", limit=0)


def test_neon_cli_requires_secret_without_echoing_value(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("NEON_DATABASE_URL", raising=False)

    exit_code = neon_main(["--salt-version", "test-v1", "--dry-run"])

    assert exit_code == 2
    error = capsys.readouterr().err
    assert "NEON_DATABASE_URL is required" in error
    assert "postgresql://" not in error
