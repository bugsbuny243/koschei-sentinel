from types import SimpleNamespace

import pytest

import koschei_sentinel.gold_holdout_inference_verify_cli as cli_module
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


def test_raw_candidate_failure_blocks_offline_verifier(monkeypatch, capsys) -> None:
    verified = False
    monkeypatch.setattr(
        cli_module,
        "verify_cyber_sft_export",
        lambda path: SimpleNamespace(
            valid=False,
            violations=["candidate export directory must not be a symlink"],
        ),
    )

    def forbidden_verify(*_args, **_kwargs):
        nonlocal verified
        verified = True
        raise AssertionError("offline verifier must not run after raw candidate failure")

    monkeypatch.setattr(
        cli_module,
        "verify_gold_holdout_inference_output",
        forbidden_verify,
    )

    status = cli_module.main(
        [
            "--output-dir",
            "output",
            "--inference-pack",
            "pack",
            "--candidate-export",
            "candidate-export-link",
        ]
    )

    assert status == 2
    assert verified is False
    assert "must not be a symlink" in capsys.readouterr().out
