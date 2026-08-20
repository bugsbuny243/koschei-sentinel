from types import SimpleNamespace

import pytest

import koschei_sentinel.gold_holdout_inference_verify_cli as cli_module
from koschei_sentinel.gold_holdout_inference_verify_cli import build_parser


def _base_args():
    return [
        "--output-dir",
        "output",
        "--inference-pack",
        "pack",
        "--inference-pack-signature",
        "pack-signature.json",
        "--reviewer-public-key",
        "reviewer-public.pem",
        "--candidate-export",
        "candidate-export",
    ]


def test_gold_holdout_verify_cli_requires_candidate_export() -> None:
    argv = _base_args()
    index = argv.index("--candidate-export")
    del argv[index : index + 2]

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(argv)

    assert exc.value.code == 2


def test_gold_holdout_verify_cli_requires_signed_pack_identity() -> None:
    argv = _base_args()
    index = argv.index("--inference-pack-signature")
    del argv[index : index + 2]

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(argv)

    assert exc.value.code == 2


def test_gold_holdout_verify_cli_accepts_candidate_export() -> None:
    args = build_parser().parse_args(_base_args())

    assert args.candidate_export == "candidate-export"
    assert args.inference_pack_signature == "pack-signature.json"
    assert args.reviewer_public_key == "reviewer-public.pem"


def test_signed_pack_failure_blocks_candidate_and_offline_verifier(
    monkeypatch,
    capsys,
) -> None:
    candidate_checked = False
    verified = False

    def fail_signed_pack(**_kwargs):
        raise ValueError("Gold HOLDOUT pack signature proof does not bind this inference manifest")

    def forbidden_candidate(_path):
        nonlocal candidate_checked
        candidate_checked = True
        raise AssertionError("candidate verification must not run after signed-pack failure")

    def forbidden_verify(*_args, **_kwargs):
        nonlocal verified
        verified = True
        raise AssertionError("offline verifier must not run after signed-pack failure")

    monkeypatch.setattr(cli_module, "_verify_signed_pack", fail_signed_pack)
    monkeypatch.setattr(cli_module, "verify_cyber_sft_export", forbidden_candidate)
    monkeypatch.setattr(cli_module, "verify_gold_holdout_inference_output", forbidden_verify)

    status = cli_module.main(_base_args())

    assert status == 2
    assert candidate_checked is False
    assert verified is False
    assert "does not bind this inference manifest" in capsys.readouterr().out


def test_raw_candidate_failure_blocks_offline_verifier(monkeypatch, capsys) -> None:
    verified = False
    monkeypatch.setattr(cli_module, "_verify_signed_pack", lambda **_kwargs: None)
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

    argv = _base_args()
    argv[argv.index("candidate-export")] = "candidate-export-link"
    status = cli_module.main(argv)

    assert status == 2
    assert verified is False
    assert "must not be a symlink" in capsys.readouterr().out
