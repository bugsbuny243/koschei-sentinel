from __future__ import annotations

from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.pq_evidence_review import (
    build_pq_evidence_review_proof,
    build_pq_evidence_reviewer_trust_policy,
    write_pq_evidence_review_proof,
    write_pq_evidence_reviewer_trust_policy,
)
from koschei_sentinel.pq_network_intelligence import (
    build_pq_research_snapshot_receipt,
    materialize_pq_network_research_record,
    write_pq_research_snapshot_receipt,
)
from koschei_sentinel.pq_reviewed_evidence_admission_cli import main

_ROOT = Path(__file__).resolve().parents[1]
_REVIEW_FIXTURE = _ROOT / "fixtures/pq/evidence-review"
_WATCH = _REVIEW_FIXTURE / "synthetic-watch.json"
_SNAPSHOT = _REVIEW_FIXTURE / "synthetic-source.txt"
_CLAIM = _REVIEW_FIXTURE / "synthetic-claim.json"
_REVIEW = _REVIEW_FIXTURE / "synthetic-review.json"
_REQUEST = _ROOT / "fixtures/pq/reviewed-catalog-admission/synthetic-request.json"
_POLICY = _ROOT / "configs/corpus/pq-reviewed-evidence-admission.v1.json"


def _write_public(path: Path, key: Ed25519PrivateKey) -> None:
    path.write_bytes(
        key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )


def test_pq_reviewed_catalog_cli_admit_and_verify(tmp_path: Path, capsys) -> None:
    snapshot_receipt = build_pq_research_snapshot_receipt(
        source_id="fixture.pq.evidence-review",
        snapshot_path=_SNAPSHOT,
        captured_at="2026-09-09T06:00:00+03:00",
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
    owner_public = tmp_path / "owner-public.pem"
    reviewer_public = tmp_path / "reviewer-public.pem"
    _write_public(owner_public, owner)
    _write_public(reviewer_public, reviewer)

    trust = build_pq_evidence_reviewer_trust_policy(
        reviewer_public_key=reviewer.public_key(),
        owner_private_key=owner,
        policy_id="fixture-pq-review-policy",
    )
    trust_path = tmp_path / "reviewer-trust.json"
    write_pq_evidence_reviewer_trust_policy(trust, trust_path)

    proof = build_pq_evidence_review_proof(
        review_input_path=_REVIEW,
        claim_path=_CLAIM,
        snapshot_receipt_path=snapshot_receipt_path,
        snapshot_path=_SNAPSHOT,
        watch_registry_path=_WATCH,
        record_path=record_path,
        materialization_receipt_path=materialization_path,
        reviewer_private_key=reviewer,
        trust_policy=trust,
        owner_public_key=owner.public_key(),
    )
    proof_path = tmp_path / "review-proof.json"
    write_pq_evidence_review_proof(proof, proof_path)

    entry_path = tmp_path / "catalog-entry.json"
    common = [
        "--request",
        str(_REQUEST),
        "--policy",
        str(_POLICY),
        "--review-proof",
        str(proof_path),
        "--trust-policy",
        str(trust_path),
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
        "--owner-public-key",
        str(owner_public),
    ]

    assert main(["admit", *common, "--output", str(entry_path)]) == 0
    admitted_output = capsys.readouterr().out
    assert '"reviewed_catalog_admitted": true' in admitted_output
    assert '"training_authorization": false' in admitted_output
    assert '"dataset_admission_allowed": false' in admitted_output

    assert main(["verify", *common, "--entry", str(entry_path)]) == 0
    verified_output = capsys.readouterr().out
    assert '"reviewed_catalog_admitted": true' in verified_output
    assert '"split_assignment": null' in verified_output
