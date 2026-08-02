from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from koschei_sentinel.dataset import export_jsonl, export_record

SALT = "unit-test-salt-with-minimum-length"
RAW_TARGET = "8KxV7JGnVKXHh4UauuMxoPjzD4hnvy9dPp7Z4hXqW2aB"
RAW_SIGNATURE = "raw-signed-verdict-signature-123456789"


def source_record() -> dict:
    return {
        "schema_version": "arvis.export.v1",
        "case_id": "case-for-" + RAW_TARGET,
        "network": "solana-mainnet",
        "target": RAW_TARGET,
        "signed_verdict": {
            "grade": "D",
            "signature": RAW_SIGNATURE,
            "triggered_rules": ["KS-HOLDER-001"],
            "summary": f"Target {RAW_TARGET} was reviewed; contact person@example.com.",
        },
        "evidence": [
            {
                "evidence_id": "evidence-raw-001",
                "kind": "holder_intelligence",
                "statement": f"Wallet {RAW_TARGET} has a concentrated position.",
                "confidence": "VERIFIED",
                "rule_ids": ["KS-HOLDER-001"],
                "attributes": {
                    "note": "Bearer abcdefghijklmnopqrstuvwxyz0123456789",
                    "share_bps": 4200,
                },
            }
        ],
        "limitations": ["No phone +90 555 555 55 55 was retained."],
        "lineage_ids": [RAW_TARGET, "incident-family-001"],
    }


def test_export_record_is_deterministic_and_removes_raw_identifiers() -> None:
    first = export_record(source_record(), salt=SALT)
    second = export_record(source_record(), salt=SALT)
    assert first == second
    serialized = first.model_dump_json()
    assert RAW_TARGET not in serialized
    assert RAW_SIGNATURE not in serialized
    assert "person@example.com" not in serialized
    assert "Bearer abcdef" not in serialized
    assert first.case.target_ref.startswith("target_")
    assert first.case.signed_verdict.signature.startswith("signature_")
    assert first.case.evidence[0].evidence_id.startswith("evidence_")


def test_duplicate_source_evidence_ids_fail_closed() -> None:
    record = source_record()
    record["evidence"].append(dict(record["evidence"][0]))
    with pytest.raises(ValidationError):
        export_record(record, salt=SALT)


def test_unsupported_fields_are_rejected() -> None:
    record = source_record()
    record["unexpected"] = "not allowed"
    with pytest.raises(ValidationError):
        export_record(record, salt=SALT)


def test_dry_run_returns_manifest_without_writing_training_data(tmp_path: Path) -> None:
    output = tmp_path / "train.jsonl"
    manifest = export_jsonl([source_record()], output_path=output, salt=SALT, dry_run=True)
    assert manifest.accepted_records == 1
    assert manifest.rejected_records == 0
    assert manifest.output_digest
    assert not output.exists()


def test_invalid_batch_writes_no_partial_output(tmp_path: Path) -> None:
    invalid = source_record()
    invalid["signed_verdict"].pop("signature")
    output = tmp_path / "train.jsonl"
    manifest = export_jsonl([source_record(), invalid], output_path=output, salt=SALT)
    assert manifest.accepted_records == 1
    assert manifest.rejected_records == 1
    assert not output.exists()


def test_jsonl_output_is_canonical_and_sorted(tmp_path: Path) -> None:
    second = source_record()
    second["case_id"] = "another-case"
    second["target"] = "7Lvc7jgQy4A7tFnXbUnJ8wuDkNnEnLrTj6Y3hFrkNmPq"
    output = tmp_path / "train.jsonl"
    manifest = export_jsonl([source_record(), second], output_path=output, salt=SALT)
    assert manifest.rejected_records == 0
    lines = output.read_text().splitlines()
    parsed = [json.loads(line) for line in lines]
    assert [item["example_id"] for item in parsed] == sorted(
        item["example_id"] for item in parsed
    )
    assert all(item["schema_version"] == "sentinel.dataset.v1" for item in parsed)
