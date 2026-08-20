from pathlib import Path
from types import SimpleNamespace

import pytest

import koschei_sentinel.gold_holdout_inference_runner_cli as cli_module
from koschei_sentinel.gold_holdout_inference_runner_cli import build_parser


def _base_args():
    return [
        "--inference-pack",
        "pack",
        "--inference-pack-signature",
        "pack-signature.json",
        "--reviewer-public-key",
        "reviewer-public.pem",
        "--candidate-export",
        "candidate-export",
        "--model-ref",
        "sentinel:test",
        "--generation-policy",
        "configs/training/gold-holdout-generation-policy.v1.json",
    ]


def test_gold_holdout_inference_cli_requires_generation_policy() -> None:
    argv = _base_args()
    index = argv.index("--generation-policy")
    del argv[index : index + 2]

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(argv)

    assert exc.value.code == 2


def test_gold_holdout_inference_cli_requires_candidate_export() -> None:
    argv = _base_args()
    index = argv.index("--candidate-export")
    del argv[index : index + 2]

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(argv)

    assert exc.value.code == 2


def test_gold_holdout_inference_cli_requires_signed_pack_identity() -> None:
    argv = _base_args()
    index = argv.index("--inference-pack-signature")
    del argv[index : index + 2]

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(argv)

    assert exc.value.code == 2


def test_gold_holdout_inference_cli_accepts_explicit_policies_and_export() -> None:
    args = build_parser().parse_args(_base_args())

    assert args.candidate_export == "candidate-export"
    assert args.inference_pack_signature == "pack-signature.json"
    assert args.reviewer_public_key == "reviewer-public.pem"
    assert args.generation_policy == (
        "configs/training/gold-holdout-generation-policy.v1.json"
    )


def test_signed_pack_failure_blocks_candidate_and_plan(monkeypatch, capsys) -> None:
    candidate_checked = False
    planned = False
    monkeypatch.setattr(cli_module, "_load_policy", lambda *_args: object())

    def fail_signed_pack(**_kwargs):
        raise ValueError("Gold HOLDOUT pack signature proof does not bind this inference manifest")

    def forbidden_candidate(_path):
        nonlocal candidate_checked
        candidate_checked = True
        raise AssertionError("candidate verification must not run after pack signature failure")

    def forbidden_plan(**_kwargs):
        nonlocal planned
        planned = True
        raise AssertionError("HOLDOUT plan must not run after pack signature failure")

    monkeypatch.setattr(cli_module, "_verify_signed_pack", fail_signed_pack)
    monkeypatch.setattr(cli_module, "verify_cyber_sft_export", forbidden_candidate)
    monkeypatch.setattr(cli_module, "build_gold_holdout_inference_plan", forbidden_plan)

    status = cli_module.main(_base_args())

    assert status == 2
    assert candidate_checked is False
    assert planned is False
    assert "does not bind this inference manifest" in capsys.readouterr().out


def test_raw_candidate_failure_blocks_holdout_plan(monkeypatch, capsys) -> None:
    planned = False
    monkeypatch.setattr(cli_module, "_load_policy", lambda *_args: object())
    monkeypatch.setattr(cli_module, "_verify_signed_pack", lambda **_kwargs: None)
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

    argv = _base_args()
    argv[argv.index("candidate-export")] = "candidate-export-link"
    status = cli_module.main(argv)

    assert status == 2
    assert planned is False
    assert "must not be a symlink" in capsys.readouterr().out


def test_atomic_execute_publishes_only_after_offline_verification(
    monkeypatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "inference-output"
    receipt = SimpleNamespace(run_id="fixture")

    def fake_execute(**kwargs):
        staging = Path(kwargs["output_dir"])
        staging.mkdir()
        (staging / "receipt.json").write_text("fixture\n", encoding="utf-8")
        return receipt

    monkeypatch.setattr(cli_module, "execute_gold_holdout_inference", fake_execute)
    monkeypatch.setattr(
        cli_module,
        "verify_gold_holdout_inference_output",
        lambda *_args, **_kwargs: SimpleNamespace(valid=True, violations=[]),
    )

    result = cli_module._execute_atomic(
        inference_pack="pack",
        candidate_export="candidate-export",
        model_ref="sentinel:test",
        output_dir=str(destination),
        generation_policy=object(),
    )

    assert result is receipt
    assert (destination / "receipt.json").read_text(encoding="utf-8") == "fixture\n"
    assert not list(tmp_path.glob(".inference-output.staging-*"))


def test_atomic_execute_failure_never_publishes_unverified_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "inference-output"

    def fake_execute(**kwargs):
        staging = Path(kwargs["output_dir"])
        staging.mkdir()
        (staging / "partial.json").write_text("partial\n", encoding="utf-8")
        return SimpleNamespace(run_id="fixture")

    monkeypatch.setattr(cli_module, "execute_gold_holdout_inference", fake_execute)
    monkeypatch.setattr(
        cli_module,
        "verify_gold_holdout_inference_output",
        lambda *_args, **_kwargs: SimpleNamespace(
            valid=False,
            violations=["sealed output mismatch"],
        ),
    )

    with pytest.raises(ValueError, match="sealed output mismatch"):
        cli_module._execute_atomic(
            inference_pack="pack",
            candidate_export="candidate-export",
            model_ref="sentinel:test",
            output_dir=str(destination),
            generation_policy=object(),
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".inference-output.staging-*"))


def test_atomic_execute_rejects_existing_destination_before_gpu_work(
    monkeypatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "inference-output"
    destination.mkdir()
    executed = False

    def forbidden_execute(**_kwargs):
        nonlocal executed
        executed = True
        raise AssertionError("GPU inference must not run for an existing destination")

    monkeypatch.setattr(cli_module, "execute_gold_holdout_inference", forbidden_execute)

    with pytest.raises(FileExistsError, match="output already exists"):
        cli_module._execute_atomic(
            inference_pack="pack",
            candidate_export="candidate-export",
            model_ref="sentinel:test",
            output_dir=str(destination),
            generation_policy=object(),
        )

    assert executed is False
