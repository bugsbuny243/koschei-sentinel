import pytest

from koschei_sentinel.gold_holdout_inference_runner_cli import build_parser


def test_gold_holdout_inference_cli_requires_generation_policy() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(
            [
                "--inference-pack",
                "pack",
                "--run-dir",
                "run",
                "--training-config",
                "training.json",
                "--model-ref",
                "sentinel:test",
            ]
        )

    assert exc.value.code == 2


def test_gold_holdout_inference_cli_accepts_generation_policy() -> None:
    args = build_parser().parse_args(
        [
            "--inference-pack",
            "pack",
            "--run-dir",
            "run",
            "--training-config",
            "training.json",
            "--model-ref",
            "sentinel:test",
            "--generation-policy",
            "configs/training/gold-holdout-generation-policy.v1.json",
        ]
    )

    assert args.generation_policy == (
        "configs/training/gold-holdout-generation-policy.v1.json"
    )
