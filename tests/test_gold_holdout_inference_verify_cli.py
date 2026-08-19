import pytest

from koschei_sentinel.gold_holdout_inference_verify_cli import build_parser


def test_gold_holdout_verify_cli_requires_candidate_export() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(
            [
                "--output-dir",
                "output",
                "--inference-pack",
                "pack",
            ]
        )

    assert exc.value.code == 2


def test_gold_holdout_verify_cli_accepts_candidate_export() -> None:
    args = build_parser().parse_args(
        [
            "--output-dir",
            "output",
            "--inference-pack",
            "pack",
            "--candidate-export",
            "candidate-export",
        ]
    )

    assert args.candidate_export == "candidate-export"
