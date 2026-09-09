import json
from pathlib import Path

import pytest

from koschei_sentinel.web4_research_snapshot import (
    Web4ResearchSnapshotReceipt,
    build_web4_research_snapshot_receipt,
    verify_web4_research_snapshot_receipt,
    write_web4_research_snapshot_receipt,
)

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ID = "ietf.draft.nemethi.aid-agent-identity-discovery-00"
_CAPTURED_AT = "2026-09-07T10:57:00+03:00"


def _registry(root: Path = _ROOT) -> Path:
    return root / "configs/corpus/web4-v1.sources.proposed.jsonl"


def _fixture(root: Path = _ROOT) -> Path:
    return root / "fixtures/web4/research-snapshot/fixture-source.txt"


def _build(snapshot: Path | None = None, registry: Path | None = None) -> Web4ResearchSnapshotReceipt:
    return build_web4_research_snapshot_receipt(
        source_id=_SOURCE_ID,
        snapshot_path=snapshot or _fixture(),
        captured_at=_CAPTURED_AT,
        source_registry_path=registry or _registry(),
    )


def _copy_registry(tmp_path: Path) -> Path:
    destination = tmp_path / "web4-v1.sources.proposed.jsonl"
    destination.write_bytes(_registry().read_bytes())
    return destination


def _mutate_source(registry: Path, field: str, value: object) -> None:
    rows = [
        json.loads(line)
        for line in registry.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source = next(row for row in rows if row["source_id"] == _SOURCE_ID)
    source[field] = value
    registry.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_research_snapshot_receipt_is_byte_bound_and_non_authorizing(tmp_path: Path) -> None:
    receipt = _build()
    receipt_path = tmp_path / "receipt.json"
    write_web4_research_snapshot_receipt(receipt, receipt_path)

    verified = verify_web4_research_snapshot_receipt(
        receipt_path=receipt_path,
        snapshot_path=_fixture(),
        source_registry_path=_registry(),
    )

    assert verified == receipt
    assert receipt.source_match_verified is False
    assert receipt.provenance_review_status == "REVIEW_REQUIRED"
    assert receipt.registry_license_status == "REVIEW_REQUIRED"
    assert receipt.network_fetch_performed is False
    assert receipt.corpus_materialized is False
    assert receipt.training_authorization is False
    assert receipt.evaluation_authorization is False
    assert receipt.promotion_eligible is False
    assert receipt.production_activation_allowed is False
    assert receipt.snapshot_size_bytes == _fixture().stat().st_size


def test_unknown_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown Web4 research snapshot source_id"):
        build_web4_research_snapshot_receipt(
            source_id="unknown.web4.source",
            snapshot_path=_fixture(),
            captured_at=_CAPTURED_AT,
            source_registry_path=_registry(),
        )


def test_symlink_snapshot_is_rejected(tmp_path: Path) -> None:
    link = tmp_path / "snapshot-link.txt"
    link.symlink_to(_fixture())

    with pytest.raises(ValueError, match="regular non-symlink file"):
        _build(snapshot=link)


def test_source_training_authorization_drift_is_rejected(tmp_path: Path) -> None:
    registry = _copy_registry(tmp_path)
    _mutate_source(registry, "training_authorization", True)

    with pytest.raises(ValueError, match="training authorization is not closed"):
        _build(registry=registry)


def test_source_license_review_bypass_is_rejected(tmp_path: Path) -> None:
    registry = _copy_registry(tmp_path)
    _mutate_source(registry, "license_status", "APPROVED")

    with pytest.raises(ValueError, match="license review state is not REVIEW_REQUIRED"):
        _build(registry=registry)


def test_snapshot_tamper_is_detected_by_verifier(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.txt"
    snapshot.write_bytes(_fixture().read_bytes())
    receipt = _build(snapshot=snapshot)
    receipt_path = tmp_path / "receipt.json"
    write_web4_research_snapshot_receipt(receipt, receipt_path)
    snapshot.write_text("tampered after capture\n", encoding="utf-8")

    with pytest.raises(ValueError, match="differs from current source/snapshot binding"):
        verify_web4_research_snapshot_receipt(
            receipt_path=receipt_path,
            snapshot_path=snapshot,
            source_registry_path=_registry(),
        )


def test_receipt_self_hash_tamper_is_rejected(tmp_path: Path) -> None:
    receipt = _build()
    payload = receipt.model_dump(mode="json")
    payload["snapshot_size_bytes"] += 1
    receipt_path = tmp_path / "tampered-receipt.json"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="self-hash does not verify"):
        verify_web4_research_snapshot_receipt(
            receipt_path=receipt_path,
            snapshot_path=_fixture(),
            source_registry_path=_registry(),
        )
