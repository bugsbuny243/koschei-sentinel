import pytest

from koschei_sentinel.gold_holdout_evaluation_evidence_cli import build_parser


def test_gold_holdout_evidence_cli_requires_explicit_policy() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(
            [
                "--release-dir",
                "release",
                "--inference-pack",
                "pack",
                "--inference-output",
                "output",
                "--output",
                "evidence.json",
            ]
        )

    assert exc.value.code == 2


def test_gold_holdout_evidence_cli_accepts_explicit_policy() -> None:
    args = build_parser().parse_args(
        [
            "--release-dir",
            "release",
            "--inference-pack",
            "pack",
            "--inference-output",
            "output",
            "--policy",
            "configs/training/gold-holdout-evaluation-policy.v1.json",
            "--output",
            "evidence.json",
        ]
    )

    assert args.policy == "configs/training/gold-holdout-evaluation-policy.v1.json"
