import copy
import hashlib

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.defense_reflex_gold_queue import _packet_digest
from koschei_sentinel.defense_reflex_gold_review import _review_digest
from koschei_sentinel.gold_model_visible_context import gold_model_visible_context_sha256
from koschei_sentinel.gold_review_signing import (
    GoldReviewSignatureProof,
    audit_gold_release_review_signatures,
    sign_gold_reviewed_packet,
    verify_gold_review_signature,
    verify_gold_review_signature_proof,
    write_signed_gold_defense_release,
)
from koschei_sentinel.training import canonical_json
from tests.test_defense_reflex_gold_release import _release_rows


def _sha(payload) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def test_gold_review_signature_verifies_with_trusted_reviewer_key() -> None:
    _policy, rows = _release_rows()
    reviewed = rows[0][2]
    private_key = Ed25519PrivateKey.generate()
    proof = sign_gold_reviewed_packet(reviewed, private_key)

    verify_gold_review_signature(reviewed, proof, private_key.public_key())

    assert proof.reviewer_id == reviewed.reviewer_id
    assert proof.review_sha256 == reviewed.review_sha256
    assert proof.signature_verified is True


def test_gold_review_signature_rejects_untrusted_reviewer_key() -> None:
    _policy, rows = _release_rows()
    reviewed = rows[0][2]
    proof = sign_gold_reviewed_packet(reviewed, Ed25519PrivateKey.generate())
    wrong_key = Ed25519PrivateKey.generate().public_key()

    with pytest.raises(ValueError, match="untrusted reviewer key"):
        verify_gold_review_signature_proof(proof, wrong_key)


def test_gold_review_signature_rejects_rehashed_review_tamper() -> None:
    _policy, rows = _release_rows()
    reviewed = rows[0][2]
    private_key = Ed25519PrivateKey.generate()
    proof = sign_gold_reviewed_packet(reviewed, private_key)

    payload = proof.model_dump(mode="json")
    payload["review_sha256"] = "0" * 64
    binding = {
        "reviewer_id": payload["reviewer_id"],
        "reviewer_key_fingerprint": payload["reviewer_key_fingerprint"],
        "packet_sha256": payload["packet_sha256"],
        "scenario_id": payload["scenario_id"],
        "split": payload["split"],
        "review_sha256": payload["review_sha256"],
    }
    payload["binding_sha256"] = _sha(binding)
    unsigned = dict(payload)
    unsigned.pop("proof_sha256", None)
    payload["proof_sha256"] = _sha(unsigned)
    tampered = GoldReviewSignatureProof.model_validate(payload)

    with pytest.raises(ValueError, match="Ed25519 signature verification failed"):
        verify_gold_review_signature_proof(tampered, private_key.public_key())


def test_signed_release_rejects_rehashed_context_poison_even_with_valid_signature(
    tmp_path,
) -> None:
    _policy, rows = _release_rows()
    scenario, packet, reviewed = rows[0]
    poisoned_context = copy.deepcopy(packet.model_visible_context)
    poisoned_context["graph_snapshots"][0]["truth"] = "MALICIOUS"
    poisoned_packet = packet.model_copy(
        update={
            "model_visible_context": poisoned_context,
            "model_visible_context_sha256": gold_model_visible_context_sha256(
                poisoned_context
            ),
        }
    )
    poisoned_packet = poisoned_packet.model_copy(
        update={
            "packet_sha256": _packet_digest(poisoned_packet.model_dump(mode="json"))
        }
    )

    reviewed_payload = reviewed.model_dump(mode="json")
    reviewed_payload["packet_sha256"] = poisoned_packet.packet_sha256
    reviewed_payload["review_sha256"] = _review_digest(reviewed_payload)
    poisoned_reviewed = reviewed.__class__.model_validate(reviewed_payload)

    private_key = Ed25519PrivateKey.generate()
    proof = sign_gold_reviewed_packet(poisoned_reviewed, private_key)
    verify_gold_review_signature(poisoned_reviewed, proof, private_key.public_key())

    with pytest.raises(ValueError, match="forbidden answer-key/review fields"):
        write_signed_gold_defense_release(
            [(scenario, poisoned_packet, poisoned_reviewed)],
            [proof],
            tmp_path / "poisoned-release",
            private_key.public_key(),
        )


def test_signed_gold_release_audit_requires_exact_signed_review_set(tmp_path) -> None:
    _policy, rows = _release_rows()
    private_key = Ed25519PrivateKey.generate()
    proofs = [
        sign_gold_reviewed_packet(reviewed, private_key)
        for _scenario, _packet, reviewed in rows
    ]
    release = tmp_path / "gold-release"
    write_signed_gold_defense_release(
        rows,
        proofs,
        release,
        private_key.public_key(),
    )

    valid = audit_gold_release_review_signatures(release, private_key.public_key())
    assert valid.valid is True
    assert valid.exact_review_set_verified is True
    assert valid.proof_count == len(rows)
    assert len(valid.proof_set_sha256) == 64

    lines = (release / "review-signatures.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    (release / "review-signatures.jsonl").write_text(
        "\n".join(lines[:-1]) + "\n",
        encoding="utf-8",
    )
    missing = audit_gold_release_review_signatures(release, private_key.public_key())

    assert missing.valid is False
    assert missing.exact_review_set_verified is False
    assert missing.proof_set_sha256 != valid.proof_set_sha256
    assert any("proof set differs" in row for row in missing.violations)
