from __future__ import annotations

import json
from pathlib import Path

import pytest

from koschei_sentinel.pq_network_intelligence import (
    build_pq_research_snapshot_receipt,
    materialize_pq_network_research_record,
    verify_pq_network_materialization,
    verify_pq_research_snapshot_receipt,
    write_pq_research_snapshot_receipt,
)

_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY = _ROOT / "configs/corpus/pq-network-watch.v1.json"
_SNAPSHOT = _ROOT / "fixtures/pq/research-snapshot/synthetic-source.txt"
_CLAIM = _ROOT / "fixtures/pq/research-snapshot/synthetic-claim.json"
_SOURCE_ID = "ethereum-protocol-priorities-2026-02-18"
_CAPTURED_AT = "2026-09-09T02:05:00+03:00"


def _capture(snapshot: Path = _SNAPSHOT):
    return build_pq_research_snapshot_receipt(
        source_id=_SOURCE_ID,
        snapshot_path=snapshot,
        captured_at=_CAPTURED_AT,
        watch_registry_path=_REGISTRY,
    )


def _write_snapshot_receipt(tmp_path: Path, snapshot: Path = _SNAPSHOT) -> Path:
    receipt_path = tmp_path / "snapshot-receipt.json"
    write_pq_research_snapshot_receipt(_capture(snapshot), receipt_path)
    return receipt_path


def _write_claim_mutation(tmp_path: Path, **updates: object) -> Path:
    payload = json.loads(_CLAIM.read_text(encoding="utf-8"))
    payload.update(updates)
    path = tmp_path / "claim.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_snapshot_receipt_is_byte_bound_and_non_authorizing(tmp_path: Path) -> None:
    receipt = _capture()
    receipt_path = tmp_path / "snapshot-receipt.json"
    write_pq_research_snapshot_receipt(receipt, receipt_path)

    verified = verify_pq_research_snapshot_receipt(
        receipt_path=receipt_path,
        snapshot_path=_SNAPSHOT,
        watch_registry_path=_REGISTRY,
    )

    assert verified == receipt
    assert receipt.source_match_verified is False
    assert receipt.provenance_review_status == "REVIEW_REQUIRED"
    assert receipt.network_fetch_performed is False
    assert receipt.training_authorization is False
    assert receipt.evaluation_authorization is False
    assert receipt.gold_eligible is False
    assert receipt.snapshot_size_bytes == _SNAPSHOT.stat().st_size


def test_unresolved_canonical_locator_is_rejected() -> None:
    with pytest.raises(ValueError, match="canonical locator is unresolved"):
        build_pq_research_snapshot_receipt(
            source_id="ethereum-current-emerging-priorities-2026-09-07",
            snapshot_path=_SNAPSHOT,
            captured_at=_CAPTURED_AT,
            watch_registry_path=_REGISTRY,
        )


def test_symlink_snapshot_is_rejected(tmp_path: Path) -> None:
    link = tmp_path / "snapshot-link.txt"
    link.symlink_to(_SNAPSHOT)

    with pytest.raises(ValueError, match="regular non-symlink file"):
        _capture(link)


def test_snapshot_tamper_is_detected(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.txt"
    snapshot.write_bytes(_SNAPSHOT.read_bytes())
    receipt_path = _write_snapshot_receipt(tmp_path, snapshot)
    snapshot.write_text("tampered after capture\n", encoding="utf-8")

    with pytest.raises(ValueError, match="differs from current source/snapshot binding"):
        verify_pq_research_snapshot_receipt(
            receipt_path=receipt_path,
            snapshot_path=snapshot,
            watch_registry_path=_REGISTRY,
        )


def test_snapshot_receipt_self_hash_tamper_is_rejected(tmp_path: Path) -> None:
    receipt = _capture()
    payload = receipt.model_dump(mode="json")
    payload["snapshot_size_bytes"] += 1
    receipt_path = tmp_path / "tampered-receipt.json"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="self-hash does not verify"):
        verify_pq_research_snapshot_receipt(
            receipt_path=receipt_path,
            snapshot_path=_SNAPSHOT,
            watch_registry_path=_REGISTRY,
        )


def test_materialization_binds_claim_snapshot_and_record_without_authorizing_training(
    tmp_path: Path,
) -> None:
    snapshot_receipt = _write_snapshot_receipt(tmp_path)
    record_path = tmp_path / "record.json"
    materialization_path = tmp_path / "materialization-receipt.json"

    record, receipt = materialize_pq_network_research_record(
        claim_path=_CLAIM,
        snapshot_receipt_path=snapshot_receipt,
        snapshot_path=_SNAPSHOT,
        watch_registry_path=_REGISTRY,
        output_path=record_path,
        materialization_receipt_path=materialization_path,
    )
    verified = verify_pq_network_materialization(
        claim_path=_CLAIM,
        snapshot_receipt_path=snapshot_receipt,
        snapshot_path=_SNAPSHOT,
        watch_registry_path=_REGISTRY,
        record_path=record_path,
        materialization_receipt_path=materialization_path,
    )

    assert verified == receipt
    assert record.schema_version == "sentinel.pq-network-intelligence.v1"
    assert record.training_authorization is False
    assert receipt.training_authorization is False
    assert receipt.evidence_verified is False
    assert receipt.provenance_review_status == "REVIEW_REQUIRED"
    assert len(record.evidence) == 1
    assert record.evidence[0].verified is False
    assert record.evidence[0].snapshot_sha256 == receipt.snapshot_sha256


def test_claim_source_mismatch_is_rejected(tmp_path: Path) -> None:
    snapshot_receipt = _write_snapshot_receipt(tmp_path)
    claim = _write_claim_mutation(tmp_path, source_id="different-source-id")

    with pytest.raises(ValueError, match="claim source_id differs"):
        materialize_pq_network_research_record(
            claim_path=claim,
            snapshot_receipt_path=snapshot_receipt,
            snapshot_path=_SNAPSHOT,
            watch_registry_path=_REGISTRY,
            output_path=tmp_path / "record.json",
            materialization_receipt_path=tmp_path / "materialization.json",
        )


def test_claim_network_outside_watch_set_is_rejected(tmp_path: Path) -> None:
    snapshot_receipt = _write_snapshot_receipt(tmp_path)
    claim = _write_claim_mutation(tmp_path, network="OutsideWatchNetwork")

    with pytest.raises(ValueError, match="network is not in the watch set"):
        materialize_pq_network_research_record(
            claim_path=claim,
            snapshot_receipt_path=snapshot_receipt,
            snapshot_path=_SNAPSHOT,
            watch_registry_path=_REGISTRY,
            output_path=tmp_path / "record.json",
            materialization_receipt_path=tmp_path / "materialization.json",
        )


def test_materialized_record_tamper_is_detected(tmp_path: Path) -> None:
    snapshot_receipt = _write_snapshot_receipt(tmp_path)
    record_path = tmp_path / "record.json"
    materialization_path = tmp_path / "materialization-receipt.json"
    materialize_pq_network_research_record(
        claim_path=_CLAIM,
        snapshot_receipt_path=snapshot_receipt,
        snapshot_path=_SNAPSHOT,
        watch_registry_path=_REGISTRY,
        output_path=record_path,
        materialization_receipt_path=materialization_path,
    )
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["uncertainty"] = "tampered after materialization"
    record_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="record differs from current claim/evidence binding"):
        verify_pq_network_materialization(
            claim_path=_CLAIM,
            snapshot_receipt_path=snapshot_receipt,
            snapshot_path=_SNAPSHOT,
            watch_registry_path=_REGISTRY,
            record_path=record_path,
            materialization_receipt_path=materialization_path,
        )


def test_materialization_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    snapshot_receipt = _write_snapshot_receipt(tmp_path)
    record_path = tmp_path / "record.json"
    record_path.write_text("already exists\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output already exists"):
        materialize_pq_network_research_record(
            claim_path=_CLAIM,
            snapshot_receipt_path=snapshot_receipt,
            snapshot_path=_SNAPSHOT,
            watch_registry_path=_REGISTRY,
            output_path=record_path,
            materialization_receipt_path=tmp_path / "materialization.json",
        )
