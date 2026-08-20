import pytest

from koschei_sentinel.defense_reflex_gold_release_audit_cli import (
    build_parser as build_audit_parser,
)
from koschei_sentinel.defense_reflex_gold_release_cli import (
    build_parser as build_release_parser,
)
from koschei_sentinel.defense_reflex_gold_review_cli import (
    build_parser as build_review_parser,
)


def _trust_args() -> list[str]:
    return [
        "--reviewer-trust-policy",
        "reviewer-trust.json",
        "--owner-public-key",
        "owner-public.pem",
    ]


def test_gold_review_cli_requires_signing_key_and_owner_pinned_trust() -> None:
    with pytest.raises(SystemExit) as exc:
        build_review_parser().parse_args(
            [
                "--packet",
                "packet.json",
                "--scenario",
                "scenario.json",
                "--review-spec",
                "review.json",
                "--output",
                "reviewed.json",
            ]
        )
    assert exc.value.code == 2


def test_gold_review_cli_accepts_explicit_owner_pinned_signing_contract() -> None:
    args = build_review_parser().parse_args(
        [
            "--packet",
            "packet.json",
            "--scenario",
            "scenario.json",
            "--review-spec",
            "review.json",
            "--reviewer-private-key",
            "reviewer-private.pem",
            *_trust_args(),
            "--output",
            "reviewed.json",
            "--signature-output",
            "review-signature.json",
        ]
    )

    assert args.reviewer_private_key == "reviewer-private.pem"
    assert args.reviewer_trust_policy == "reviewer-trust.json"
    assert args.owner_public_key == "owner-public.pem"
    assert args.signature_output == "review-signature.json"


def test_gold_release_cli_requires_signature_proofs_and_owner_pinned_public_key() -> None:
    with pytest.raises(SystemExit) as exc:
        build_release_parser().parse_args(
            [
                "--scenario",
                "scenario.json",
                "--packet",
                "packet.json",
                "--reviewed",
                "reviewed.json",
                "--output-dir",
                "gold-release",
            ]
        )
    assert exc.value.code == 2


def test_gold_release_cli_accepts_owner_pinned_signed_release_contract() -> None:
    args = build_release_parser().parse_args(
        [
            "--scenario",
            "scenario.json",
            "--packet",
            "packet.json",
            "--reviewed",
            "reviewed.json",
            "--review-signature",
            "review-signature.json",
            "--reviewer-public-key",
            "reviewer-public.pem",
            *_trust_args(),
            "--output-dir",
            "gold-release",
        ]
    )

    assert args.review_signature == ["review-signature.json"]
    assert args.reviewer_public_key == "reviewer-public.pem"
    assert args.reviewer_trust_policy == "reviewer-trust.json"
    assert args.owner_public_key == "owner-public.pem"


def test_gold_audit_cli_requires_owner_pinned_reviewer_public_key() -> None:
    with pytest.raises(SystemExit) as exc:
        build_audit_parser().parse_args(["--release-dir", "gold-release"])
    assert exc.value.code == 2


def test_gold_audit_cli_accepts_owner_pinned_signed_audit_contract() -> None:
    args = build_audit_parser().parse_args(
        [
            "--release-dir",
            "gold-release",
            "--reviewer-public-key",
            "reviewer-public.pem",
            *_trust_args(),
            "--output",
            "gold-audit.json",
        ]
    )

    assert args.reviewer_public_key == "reviewer-public.pem"
    assert args.reviewer_trust_policy == "reviewer-trust.json"
    assert args.owner_public_key == "owner-public.pem"
