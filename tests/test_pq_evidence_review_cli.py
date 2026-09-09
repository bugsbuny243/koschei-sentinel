from __future__ import annotations

from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.pq_evidence_review_cli import main
from koschei_sentinel.pq_network_intelligence import (
    build_pq_research_snapshot_receipt,
    materialize_pq_network_research_record,
    write_pq_research_snapshot_receipt,
)

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _ROOT / "fixtures/pq/evidence-review"
_WATCH = _FIXTURE / "synthetic-watch.json"
_SNAPSHOT = _FIXTURE / "synthetic-source.txt"
_CLAIM = _FIXTURE / "synthetic-claim.json"
_REVIEW = _FIXTURE / "synthetic-review.json"


def _write_private(path: Path, key: Ed25519PrivateKey) -> None:
    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )


def _write_public(path: Path, key: Ed25519PrivateKey) -> None:
    path.write_bytes(
        key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )


def test_pq_evidence_review_cli_trust_sign_verify(tmp_path: Path, capsys) -> None:
    snapshot_receipt = build_pq_research_snapshot_receipt(
        source_id="fixture.pq.evidence-review",
        snapshot_path=_SNAPSHOT,
        captured_at="2026-09-09T05:56:30+03:00",
        watch_registry_path=_WATCH,
    )
    snapshot_receipt_path = tmp_path / "snapshot-receipt.json"
    write_pq_research_snapshot_receipt(snapshot_receipt, snapshot_receipt_path)
    record_path = tmp_path / "record.json"
    materialization_path = tmp_path / "materialization-receipt.json"
    materialize_pq_network_research_record(
        claim_path=_CLAIM,
        snapshot_receipt_path=snapshot_receipt_path,
        snapshot_path=_SNAPSHOT,
        watch_registry_path=_WATCH,
        output_path=record_path,
        materialization_receipt_path=materialization_path,
    )

    owner = Ed25519PrivateKey.generate()
    reviewer = Ed25519PrivateKey.generate()
    owner_private = tmp_path / "owner-private.pem"
    owner_public = tmp_path / "owner-public.pem"
    reviewer_private = tmp_path / "reviewer-private.pem"
    reviewer_public = tmp_path / "reviewer-public.pem"
    _write_private(owner_private, owner)
    _write_public(owner_public, owner)
    _write_private(reviewer_private, reviewer)
    _write_public(reviewer_public, reviewer)

    policy_path = tmp_path / "trust-policy.json"
    assert (
        main(
            [
                "trust-create",
                "--policy-id",
                "fixture-pq-review-policy",
                "--reviewer-public-key",
                str(reviewer_public),
                "--owner-private-key",
                str(owner_private),
                "--output",
                str(policy_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    proof_path = tmp_path / "review-proof.json"
    assert (
        main(
            [
                "sign",
                "--review-input",
                str(_REVIEW),
                "--claim",
                str(_CLAIM),
                "--snapshot-receipt",
                str(snapshot_receipt_path),
                "--snapshot",
                str(_SNAPSHOT),
                "--watch",
                str(_WATCH),
                "--record",
                str(record_path),
                "--materialization-receipt",
                str(materialization_path),
                "--reviewer-private-key",
                str(reviewer_private),
                "--trust-policy",
                str(policy_path),
                "--owner-public-key",
                str(owner_public),
                "--output",
                str(proof_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "verify",
                "--proof",
                str(proof_path),
                "--claim",
                str(_CLAIM),
                "--snapshot-receipt",
                str(snapshot_receipt_path),
                "--snapshot",
                str(_SNAPSHOT),
                "--watch",
                str(_WATCH),
                "--record",
                str(record_path),
                "--materialization-receipt",
                str(materialization_path),
                "--reviewer-public-key",
                str(reviewer_public),
                "--trust-policy",
                str(policy_path),
                "--owner-public-key",
                str(owner_public),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert '"evidence_verified": true' in output
    assert '"training_authorization": false' in output
