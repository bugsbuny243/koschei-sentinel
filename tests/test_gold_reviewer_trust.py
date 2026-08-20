import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.gold_reviewer_trust import (
    build_gold_reviewer_trust_policy,
    load_trusted_reviewer_private_key,
    load_trusted_reviewer_public_key,
    verify_gold_reviewer_trust_policy,
)


def _write_public_key(path, private_key: Ed25519PrivateKey) -> None:
    path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )


def _write_private_key(path, private_key: Ed25519PrivateKey) -> None:
    path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )


def _write_policy(path, policy) -> None:
    path.write_text(
        json.dumps(policy.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_owner_signed_policy_accepts_only_pinned_reviewer_key() -> None:
    owner = Ed25519PrivateKey.generate()
    reviewer = Ed25519PrivateKey.generate()
    policy = build_gold_reviewer_trust_policy(
        reviewer.public_key(),
        owner,
        policy_id="gold-reviewer-v1",
    )

    verified = verify_gold_reviewer_trust_policy(
        policy,
        reviewer.public_key(),
        owner.public_key(),
    )

    assert verified == policy

    attacker_reviewer = Ed25519PrivateKey.generate()
    with pytest.raises(ValueError, match="does not match owner-signed Gold trust policy"):
        verify_gold_reviewer_trust_policy(
            policy,
            attacker_reviewer.public_key(),
            owner.public_key(),
        )


def test_attacker_self_consistent_policy_fails_against_real_owner_root() -> None:
    real_owner = Ed25519PrivateKey.generate()
    attacker_owner = Ed25519PrivateKey.generate()
    attacker_reviewer = Ed25519PrivateKey.generate()
    attacker_policy = build_gold_reviewer_trust_policy(
        attacker_reviewer.public_key(),
        attacker_owner,
        policy_id="gold-reviewer-v1",
    )

    with pytest.raises(ValueError, match="owner public key does not match"):
        verify_gold_reviewer_trust_policy(
            attacker_policy,
            attacker_reviewer.public_key(),
            real_owner.public_key(),
        )


def test_trusted_key_loaders_reject_reviewer_file_swap(tmp_path) -> None:
    owner = Ed25519PrivateKey.generate()
    reviewer = Ed25519PrivateKey.generate()
    attacker = Ed25519PrivateKey.generate()
    policy = build_gold_reviewer_trust_policy(
        reviewer.public_key(),
        owner,
        policy_id="gold-reviewer-v1",
    )

    owner_public = tmp_path / "owner-public.pem"
    reviewer_public = tmp_path / "reviewer-public.pem"
    reviewer_private = tmp_path / "reviewer-private.pem"
    policy_path = tmp_path / "reviewer-trust.json"
    _write_public_key(owner_public, owner)
    _write_public_key(reviewer_public, reviewer)
    _write_private_key(reviewer_private, reviewer)
    _write_policy(policy_path, policy)

    loaded_public = load_trusted_reviewer_public_key(
        reviewer_public_key_path=reviewer_public,
        trust_policy_path=policy_path,
        owner_public_key_path=owner_public,
    )
    loaded_private = load_trusted_reviewer_private_key(
        reviewer_private_key_path=reviewer_private,
        trust_policy_path=policy_path,
        owner_public_key_path=owner_public,
    )
    assert loaded_public.public_bytes_raw() == reviewer.public_key().public_bytes_raw()
    assert loaded_private.public_key().public_bytes_raw() == reviewer.public_key().public_bytes_raw()

    _write_public_key(reviewer_public, attacker)
    _write_private_key(reviewer_private, attacker)
    with pytest.raises(ValueError, match="does not match owner-signed Gold trust policy"):
        load_trusted_reviewer_public_key(
            reviewer_public_key_path=reviewer_public,
            trust_policy_path=policy_path,
            owner_public_key_path=owner_public,
        )
    with pytest.raises(ValueError, match="does not match owner-signed Gold trust policy"):
        load_trusted_reviewer_private_key(
            reviewer_private_key_path=reviewer_private,
            trust_policy_path=policy_path,
            owner_public_key_path=owner_public,
        )
