import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.gold_holdout_pack_admission import (
    admit_owner_trusted_signed_gold_holdout_pack,
    admit_signed_gold_holdout_pack,
    verify_admitted_gold_holdout_pack,
)
from koschei_sentinel.gold_holdout_pack_signing import sign_gold_holdout_inference_pack
from koschei_sentinel.gold_reviewer_trust import (
    build_gold_reviewer_trust_policy,
    write_gold_reviewer_trust_policy,
)
from tests.test_gold_holdout_inference_runner import _pack


_REVIEW_AUDIT_SHA = "d" * 64


def _write_public_key(path, private_key: Ed25519PrivateKey) -> None:
    path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )


def _signed_pack(tmp_path):
    pack = _pack(tmp_path)
    private_key = Ed25519PrivateKey.generate()
    proof = sign_gold_holdout_inference_pack(
        pack / "manifest.json",
        private_key,
        review_signature_audit_sha256=_REVIEW_AUDIT_SHA,
    )
    signature_path = tmp_path / "pack-signature.json"
    signature_path.write_text(
        json.dumps(proof.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    public_key_path = tmp_path / "reviewer-public.pem"
    _write_public_key(public_key_path, private_key)
    return pack, signature_path, public_key_path, proof


def _owner_trusted_signed_pack(tmp_path):
    pack, signature_path, reviewer_public_key_path, proof = _signed_pack(tmp_path)
    reviewer_public_key = serialization.load_pem_public_key(
        reviewer_public_key_path.read_bytes()
    )
    owner = Ed25519PrivateKey.generate()
    policy = build_gold_reviewer_trust_policy(
        reviewer_public_key,
        owner,
        policy_id="gold-reviewer-v1",
    )
    policy_path = tmp_path / "reviewer-trust.json"
    write_gold_reviewer_trust_policy(policy, policy_path)
    owner_public_key_path = tmp_path / "owner-public.pem"
    _write_public_key(owner_public_key_path, owner)
    return (
        pack,
        signature_path,
        reviewer_public_key_path,
        policy_path,
        owner_public_key_path,
        proof,
    )


def test_signed_pack_admission_accepts_exact_pack_proof_and_trust_root(tmp_path) -> None:
    pack, signature_path, public_key_path, proof = _signed_pack(tmp_path)

    admission = admit_signed_gold_holdout_pack(
        inference_pack=pack,
        signature_path=signature_path,
        reviewer_public_key_path=public_key_path,
    )

    assert admission.proof == proof
    verify_admitted_gold_holdout_pack(admission, pack)


def test_signed_pack_admission_rejects_wrong_trust_root(tmp_path) -> None:
    pack, signature_path, _public_key_path, _proof = _signed_pack(tmp_path)
    wrong_key_path = tmp_path / "wrong-reviewer-public.pem"
    _write_public_key(wrong_key_path, Ed25519PrivateKey.generate())

    with pytest.raises(ValueError, match="untrusted reviewer key"):
        admit_signed_gold_holdout_pack(
            inference_pack=pack,
            signature_path=signature_path,
            reviewer_public_key_path=wrong_key_path,
        )


def test_admission_keeps_original_trust_root_after_key_and_proof_paths_are_swapped(
    tmp_path,
) -> None:
    pack, signature_path, public_key_path, proof = _signed_pack(tmp_path)
    admission = admit_signed_gold_holdout_pack(
        inference_pack=pack,
        signature_path=signature_path,
        reviewer_public_key_path=public_key_path,
    )

    _write_public_key(public_key_path, Ed25519PrivateKey.generate())
    signature_path.write_text("{}\n", encoding="utf-8")

    verify_admitted_gold_holdout_pack(admission, pack)
    assert admission.proof == proof


def test_owner_trusted_admission_accepts_owner_signed_reviewer_policy(tmp_path) -> None:
    (
        pack,
        signature_path,
        reviewer_public_key_path,
        policy_path,
        owner_public_key_path,
        proof,
    ) = _owner_trusted_signed_pack(tmp_path)

    admission = admit_owner_trusted_signed_gold_holdout_pack(
        inference_pack=pack,
        signature_path=signature_path,
        reviewer_public_key_path=reviewer_public_key_path,
        reviewer_trust_policy_path=policy_path,
        owner_public_key_path=owner_public_key_path,
    )

    assert admission.proof == proof
    verify_admitted_gold_holdout_pack(admission, pack)


def test_owner_trusted_admission_rejects_attacker_owner_root(tmp_path) -> None:
    (
        pack,
        signature_path,
        reviewer_public_key_path,
        policy_path,
        _owner_public_key_path,
        _proof,
    ) = _owner_trusted_signed_pack(tmp_path)
    attacker_owner_public_key_path = tmp_path / "attacker-owner-public.pem"
    _write_public_key(attacker_owner_public_key_path, Ed25519PrivateKey.generate())

    with pytest.raises(ValueError, match="owner public key does not match"):
        admit_owner_trusted_signed_gold_holdout_pack(
            inference_pack=pack,
            signature_path=signature_path,
            reviewer_public_key_path=reviewer_public_key_path,
            reviewer_trust_policy_path=policy_path,
            owner_public_key_path=attacker_owner_public_key_path,
        )
