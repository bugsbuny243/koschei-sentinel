import pytest

from koschei_sentinel.gold_holdout_evaluation_cli import build_parser


def test_gold_evaluate_output_requires_candidate_export() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(
            [
                "evaluate-output",
                "--release-dir",
                "release",
                "--inference-pack",
                "pack",
                "--inference-output",
                "output",
                "--policy",
                "configs/training/gold-holdout-evaluation-policy.v1.json",
            ]
        )

    assert exc.value.code == 2


def test_gold_evaluate_output_requires_explicit_policy() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(
            [
                "evaluate-output",
                "--release-dir",
                "release",
                "--inference-pack",
                "pack",
                "--inference-output",
                "output",
                "--candidate-export",
                "candidate-export",
            ]
        )

    assert exc.value.code == 2


def test_gold_evaluate_output_accepts_verified_inputs() -> None:
    args = build_parser().parse_args(
        [
            "evaluate-output",
            "--release-dir",
            "release",
            "--inference-pack",
            "pack",
            "--inference-output",
            "output",
            "--candidate-export",
            "candidate-export",
            "--policy",
            "configs/training/gold-holdout-evaluation-policy.v1.json",
        ]
    )

    assert args.candidate_export == "candidate-export"
    assert args.policy == "configs/training/gold-holdout-evaluation-policy.v1.json"
