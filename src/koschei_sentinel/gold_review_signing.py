from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import Field

from koschei_sentinel.defense_reflex_corpus_v3 import DefenseReflexTrainingExampleV3
from koschei_sentinel.defense_reflex_gold_queue import GoldReviewSplit
from koschei_sentinel.defense_reflex_gold_release import (
    GoldHoldoutEvaluationCase,
    write_gold_defense_release,
)
from koschei_sentinel.defense_reflex_gold_review import GoldReviewedPacket
from koschei_sentinel.gold_model_visible_context import verify_gold_model_visible_context
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import canonical_json

_DIGEST = r"^[a-f0-9]{64}$"
_SIGNATURE_CONTEXT = b"koschei-sentinel-gold-human-review-v1\0"


class GoldReviewSignatureProof(StrictModel):
    schema_version: Literal["sentinel.gold-review-signature-proof.v1"] = (
        "sentinel.gold-review-signature-proof.v1"
    )
    reviewer_id: str = Field(min_length=3, max_length=256)
    reviewer_key_fingerprint: str = Field(pattern=_DIGEST)
    packet_sha256: str = Field(pattern=_DIGEST)
    scenario_id: str = Field(min_length=3, max_length=256)
    split: GoldReviewSplit
    review_sha256: str = Field(pattern=_DIGEST)
    binding_sha256: str = Field(pattern=_DIGEST)
    signature_algorithm: Literal["ed25519"] = "ed25519"
    signature_base64: str = Field(min_length=80, max_length=128)
    signature_verified: Literal[True] = True
    proof_sha256: str = Field(pattern=_DIGEST)


class GoldReviewSignatureAudit(StrictModel):
    schema_version: Literal["sentinel.gold-review-signature-audit.v1"] = (
        "sentinel.gold-review-signature-audit.v1"
    )
    reviewer_key_fingerprint: str = Field(pattern=_DIGEST)
    expected_review_count: int = Field(ge=0)
    proof_count: int = Field(ge=0)
    proof_set_sha256: str = Field(pattern=_DIGEST)
    exact_review_set_verified: bool
    signatures_verified: bool
    valid: bool
    violations: list[str]
    audit_sha256: str = Field(pattern=_DIGEST)


def _sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _digest(payload: dict[str, object], field_name: str | None = None) -> str:
    unsigned = dict(payload)
    if field_name is not None:
        unsigned.pop(field_name, None)
    return _sha256_text(canonical_json(unsigned))


def reviewer_public_key_fingerprint(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def load_reviewer_private_key(path: str | Path) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("Gold reviewer private key must be unencrypted Ed25519 PEM")
    return key


def load_reviewer_public_key(path: str | Path) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(Path(path).read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("Gold reviewer public key must be Ed25519")
    return key


def _binding_payload(
    reviewed: GoldReviewedPacket,
    reviewer_key_fingerprint: str,
) -> dict[str, object]:
    return {
        "reviewer_id": reviewed.reviewer_id,
        "reviewer_key_fingerprint": reviewer_key_fingerprint,
        "packet_sha256": reviewed.packet_sha256,
        "scenario_id": reviewed.scenario_id,
        "split": reviewed.split.value,
        "review_sha256": reviewed.review_sha256,
    }


def _signature_message(binding_sha256: str) -> bytes:
    return _SIGNATURE_CONTEXT + binding_sha256.encode("ascii")


def sign_gold_reviewed_packet(
    reviewed: GoldReviewedPacket,
    reviewer_private_key: Ed25519PrivateKey,
) -> GoldReviewSignatureProof:
    public_key = reviewer_private_key.public_key()
    fingerprint = reviewer_public_key_fingerprint(public_key)
    binding = _binding_payload(reviewed, fingerprint)
    binding_sha = _digest(binding)
    signature = reviewer_private_key.sign(_signature_message(binding_sha))
    public_key.verify(signature, _signature_message(binding_sha))
    payload: dict[str, object] = {
        "schema_version": "sentinel.gold-review-signature-proof.v1",
        **binding,
        "binding_sha256": binding_sha,
        "signature_algorithm": "ed25519",
        "signature_base64": base64.b64encode(signature).decode("ascii"),
        "signature_verified": True,
    }
    payload["proof_sha256"] = _digest(payload)
    return GoldReviewSignatureProof.model_validate(payload)


def verify_gold_review_signature_proof(
    proof: GoldReviewSignatureProof,
    reviewer_public_key: Ed25519PublicKey,
) -> None:
    payload = proof.model_dump(mode="json")
    if _digest(payload, "proof_sha256") != proof.proof_sha256:
        raise ValueError("Gold review signature proof self-hash does not verify")
    fingerprint = reviewer_public_key_fingerprint(reviewer_public_key)
    if proof.reviewer_key_fingerprint != fingerprint:
        raise ValueError("Gold review signature proof uses an untrusted reviewer key")
    binding = {
        "reviewer_id": proof.reviewer_id,
        "reviewer_key_fingerprint": proof.reviewer_key_fingerprint,
        "packet_sha256": proof.packet_sha256,
        "scenario_id": proof.scenario_id,
        "split": proof.split.value,
        "review_sha256": proof.review_sha256,
    }
    if _digest(binding) != proof.binding_sha256:
        raise ValueError("Gold review signature binding digest does not verify")
    try:
        signature = base64.b64decode(proof.signature_base64, validate=True)
        reviewer_public_key.verify(signature, _signature_message(proof.binding_sha256))
    except (InvalidSignature, ValueError) as exc:
        raise ValueError("Gold reviewer Ed25519 signature verification failed") from exc


def verify_gold_review_signature(
    reviewed: GoldReviewedPacket,
    proof: GoldReviewSignatureProof,
    reviewer_public_key: Ed25519PublicKey,
) -> None:
    verify_gold_review_signature_proof(proof, reviewer_public_key)
    expected = _binding_payload(reviewed, proof.reviewer_key_fingerprint)
    observed = {
        "reviewer_id": proof.reviewer_id,
        "reviewer_key_fingerprint": proof.reviewer_key_fingerprint,
        "packet_sha256": proof.packet_sha256,
        "scenario_id": proof.scenario_id,
        "split": proof.split.value,
        "review_sha256": proof.review_sha256,
    }
    if observed != expected:
        raise ValueError("Gold review signature proof does not bind the reviewed packet")


def _serialize_proofs(proofs: list[GoldReviewSignatureProof]) -> str:
    ordered = sorted(
        proofs,
        key=lambda row: (row.split.value, row.scenario_id, row.review_sha256),
    )
    return "".join(
        json.dumps(
            row.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in ordered
    )


def _proof_set_sha256(proofs: list[GoldReviewSignatureProof]) -> str:
    bindings = [
        {
            "split": proof.split.value,
            "scenario_id": proof.scenario_id,
            "reviewer_id": proof.reviewer_id,
            "review_sha256": proof.review_sha256,
            "proof_sha256": proof.proof_sha256,
        }
        for proof in sorted(
            proofs,
            key=lambda row: (row.split.value, row.scenario_id, row.review_sha256),
        )
    ]
    return _sha256_text(canonical_json(bindings))


def write_signed_gold_defense_release(
    rows: list[tuple[object, object, GoldReviewedPacket]],
    proofs: list[GoldReviewSignatureProof],
    output_dir: str | Path,
    reviewer_public_key: Ed25519PublicKey,
):
    if len(rows) != len(proofs):
        raise ValueError("Gold signed release requires one signature proof per reviewed packet")
    proof_by_review = {proof.review_sha256: proof for proof in proofs}
    if len(proof_by_review) != len(proofs):
        raise ValueError("Gold signed release signature proof review digests must be unique")
    for scenario, packet, reviewed in rows:
        proof = proof_by_review.get(reviewed.review_sha256)
        if proof is None:
            raise ValueError("Gold signed release is missing a reviewed-packet signature proof")
        verify_gold_model_visible_context(
            scenario=scenario,
            context=packet.model_visible_context,
            context_sha256=packet.model_visible_context_sha256,
        )
        verify_gold_review_signature(reviewed, proof, reviewer_public_key)

    manifest = write_gold_defense_release(rows, output_dir)
    signature_path = Path(output_dir) / "review-signatures.jsonl"
    signature_path.write_text(_serialize_proofs(proofs), encoding="utf-8")
    return manifest


def _expected_release_reviews(
    release_dir: Path,
) -> set[tuple[str, str, str, str, str]]:
    expected: set[tuple[str, str, str, str, str]] = set()
    for split in (GoldReviewSplit.TRAIN, GoldReviewSplit.VALIDATION):
        path = release_dir / split.value.lower() / "examples.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = DefenseReflexTrainingExampleV3.model_validate_json(line)
            reviewer_id = row.provenance.get("reviewer_id")
            packet_sha256 = row.provenance.get("gold_packet_sha256")
            review_sha256 = row.provenance.get("gold_review_sha256")
            if reviewer_id is None:
                raise ValueError("Gold training example lacks reviewer_id provenance")
            if packet_sha256 is None:
                raise ValueError("Gold training example lacks gold_packet_sha256 provenance")
            if review_sha256 is None:
                raise ValueError("Gold training example lacks gold_review_sha256 provenance")
            expected.add(
                (
                    split.value,
                    row.scenario_id,
                    reviewer_id,
                    packet_sha256,
                    review_sha256,
                )
            )

    holdout_path = release_dir / "holdout" / "cases.jsonl"
    for line in holdout_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = GoldHoldoutEvaluationCase.model_validate_json(line)
        expected.add(
            (
                GoldReviewSplit.HOLDOUT.value,
                row.scenario_id,
                row.reviewer_id,
                row.packet_sha256,
                row.review_sha256,
            )
        )
    return expected


def audit_gold_release_review_signatures(
    release_dir: str | Path,
    reviewer_public_key: Ed25519PublicKey,
) -> GoldReviewSignatureAudit:
    root = Path(release_dir)
    fingerprint = reviewer_public_key_fingerprint(reviewer_public_key)
    violations: list[str] = []
    proofs: list[GoldReviewSignatureProof] = []
    expected: set[tuple[str, str, str, str, str]] = set()
    signatures_verified = False
    exact_set_verified = False

    try:
        expected = _expected_release_reviews(root)
        signature_path = root / "review-signatures.jsonl"
        lines = signature_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                proof = GoldReviewSignatureProof.model_validate_json(line)
                verify_gold_review_signature_proof(proof, reviewer_public_key)
            except ValueError as exc:
                raise ValueError(
                    f"invalid Gold review signature proof at line {line_number}: {exc}"
                ) from exc
            proofs.append(proof)
        observed = {
            (
                proof.split.value,
                proof.scenario_id,
                proof.reviewer_id,
                proof.packet_sha256,
                proof.review_sha256,
            )
            for proof in proofs
        }
        if len(observed) != len(proofs):
            violations.append("Gold review signature proofs contain duplicate review bindings")
        exact_set_verified = observed == expected and len(proofs) == len(expected)
        if not exact_set_verified:
            violations.append("Gold review signature proof set differs from release review set")
        signatures_verified = not violations
    except (OSError, TypeError, ValueError) as exc:
        violations.append(str(exc))

    valid = signatures_verified and exact_set_verified and not violations
    payload: dict[str, object] = {
        "schema_version": "sentinel.gold-review-signature-audit.v1",
        "reviewer_key_fingerprint": fingerprint,
        "expected_review_count": len(expected),
        "proof_count": len(proofs),
        "proof_set_sha256": _proof_set_sha256(proofs),
        "exact_review_set_verified": exact_set_verified,
        "signatures_verified": signatures_verified,
        "valid": valid,
        "violations": violations,
    }
    payload["audit_sha256"] = _digest(payload)
    return GoldReviewSignatureAudit.model_validate(payload)
