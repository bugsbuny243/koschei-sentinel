import pytest

from koschei_sentinel.gold_holdout_inference_runner_cli import build_parser


def test_gold_holdout_inference_cli_requires_generation_policy() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(
            [
                "--inference-pack",
                "pack",
                "--candidate-export",
                "candidate-export",
                "--model-ref",
                "sentinel:test",
            ]
        )

    assert exc.value.code == 2


def test_gold_holdout_inference_cli_requires_candidate_export() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(
            [
                "--inference-pack",
                "pack",
                "--model-ref",
                "sentinel:test",
                "--generation-policy",
                "configs/training/gold-holdout-generation-policy.v1.json",
            ]
        )

    assert exc.value.code == 2


def test_gold_holdout_inference_cli_accepts_explicit_policies_and_export() -> None:
    args = build_parser().parse_args(
        [
            "--inference-pack",
            "pack",
            "--candidate-export",
            "candidate-export",
            "--model-ref",
            "sentinel:test",
            "--generation-policy",
            "configs/training/gold-holdout-generation-policy.v1.json",
        ]
    )

    assert args.candidate_export == "candidate-export"
    assert args.generation_policy == (
        "configs/training/gold-holdout-generation-policy.v1.json"
    )
