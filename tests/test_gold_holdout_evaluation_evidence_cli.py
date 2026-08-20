import pytest

from koschei_sentinel.gold_holdout_evaluation_evidence_cli import build_parser


def _base_args():
    return [
        "--release-dir",
        "release",
        "--inference-pack",
        "pack",
        "--inference-pack-signature",
        "pack-signature.json",
        "--inference-output",
        "output",
        "--candidate-export",
        "candidate-export",
        "--reviewer-public-key",
        "reviewer-public.pem",
        "--reviewer-trust-policy",
        "reviewer-trust.json",
        "--owner-public-key",
        "owner-public.pem",
        "--policy",
        "configs/training/gold-holdout-evaluation-policy.v1.json",
        "--output",
        "evidence.json",
    ]


def _without(argv: list[str], option: str) -> list[str]:
    result = list(argv)
    index = result.index(option)
    del result[index : index + 2]
    return result


def test_gold_holdout_evidence_cli_requires_explicit_policy() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(_without(_base_args(), "--policy"))
    assert exc.value.code == 2


def test_gold_holdout_evidence_cli_requires_candidate_export() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(_without(_base_args(), "--candidate-export"))
    assert exc.value.code == 2


def test_gold_holdout_evidence_cli_requires_reviewer_public_key() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(_without(_base_args(), "--reviewer-public-key"))
    assert exc.value.code == 2


def test_gold_holdout_evidence_cli_requires_reviewer_trust_policy() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(_without(_base_args(), "--reviewer-trust-policy"))
    assert exc.value.code == 2


def test_gold_holdout_evidence_cli_requires_owner_public_key() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(_without(_base_args(), "--owner-public-key"))
    assert exc.value.code == 2


def test_gold_holdout_evidence_cli_requires_pack_signature() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(_without(_base_args(), "--inference-pack-signature"))
    assert exc.value.code == 2


def test_gold_holdout_evidence_cli_accepts_owner_pinned_trust_contract() -> None:
    args = build_parser().parse_args(_base_args())

    assert args.candidate_export == "candidate-export"
    assert args.inference_pack_signature == "pack-signature.json"
    assert args.reviewer_public_key == "reviewer-public.pem"
    assert args.reviewer_trust_policy == "reviewer-trust.json"
    assert args.owner_public_key == "owner-public.pem"
    assert args.policy == "configs/training/gold-holdout-evaluation-policy.v1.json"
