from types import SimpleNamespace

import pytest

import koschei_sentinel.gold_holdout_inference_runner_cli as cli_module
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


def test_raw_candidate_failure_blocks_holdout_plan(monkeypatch, capsys) -> None:
    planned = False
    monkeypatch.setattr(cli_module, "_load_policy", lambda *_args: object())
    monkeypatch.setattr(
        cli_module,
        "verify_cyber_sft_export",
        lambda path: SimpleNamespace(
            valid=False,
            violations=["candidate export directory must not be a symlink"],
        ),
    )

    def forbidden_plan(**_kwargs):
        nonlocal planned
        planned = True
        raise AssertionError("HOLDOUT plan must not run after raw candidate failure")

    monkeypatch.setattr(cli_module, "build_gold_holdout_inference_plan", forbidden_plan)

    status = cli_module.main(
        [
            "--inference-pack",
            "pack",
            "--candidate-export",
            "candidate-export-link",
            "--model-ref",
            "sentinel:test",
            "--generation-policy",
            "policy.json",
        ]
    )

    assert status == 2
    assert planned is False
    assert "must not be a symlink" in capsys.readouterr().out
