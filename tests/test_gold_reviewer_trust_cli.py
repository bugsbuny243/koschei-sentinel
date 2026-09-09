from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.gold_reviewer_trust_cli import main


def _write_private_key(path, private_key: Ed25519PrivateKey) -> None:
    path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )


def _write_public_key(path, private_key: Ed25519PrivateKey) -> None:
    path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )


def test_gold_reviewer_trust_cli_issue_then_verify(tmp_path) -> None:
    owner = Ed25519PrivateKey.generate()
    reviewer = Ed25519PrivateKey.generate()
    owner_private = tmp_path / "owner-private.pem"
    owner_public = tmp_path / "owner-public.pem"
    reviewer_public = tmp_path / "reviewer-public.pem"
    policy = tmp_path / "reviewer-trust.json"
    _write_private_key(owner_private, owner)
    _write_public_key(owner_public, owner)
    _write_public_key(reviewer_public, reviewer)

    issue_status = main(
        [
            "issue",
            "--policy-id",
            "gold-reviewer-v1",
            "--reviewer-public-key",
            str(reviewer_public),
            "--owner-private-key",
            str(owner_private),
            "--output",
            str(policy),
        ]
    )
    verify_status = main(
        [
            "verify",
            "--policy",
            str(policy),
            "--reviewer-public-key",
            str(reviewer_public),
            "--owner-public-key",
            str(owner_public),
        ]
    )

    assert issue_status == 0
    assert verify_status == 0
    assert policy.is_file()


def test_gold_reviewer_trust_cli_rejects_wrong_owner_root(tmp_path) -> None:
    owner = Ed25519PrivateKey.generate()
    reviewer = Ed25519PrivateKey.generate()
    attacker_owner = Ed25519PrivateKey.generate()
    owner_private = tmp_path / "owner-private.pem"
    reviewer_public = tmp_path / "reviewer-public.pem"
    attacker_owner_public = tmp_path / "attacker-owner-public.pem"
    policy = tmp_path / "reviewer-trust.json"
    _write_private_key(owner_private, owner)
    _write_public_key(reviewer_public, reviewer)
    _write_public_key(attacker_owner_public, attacker_owner)

    assert (
        main(
            [
                "issue",
                "--policy-id",
                "gold-reviewer-v1",
                "--reviewer-public-key",
                str(reviewer_public),
                "--owner-private-key",
                str(owner_private),
                "--output",
                str(policy),
            ]
        )
        == 0
    )
    status = main(
        [
            "verify",
            "--policy",
            str(policy),
            "--reviewer-public-key",
            str(reviewer_public),
            "--owner-public-key",
            str(attacker_owner_public),
        ]
    )

    assert status == 2
