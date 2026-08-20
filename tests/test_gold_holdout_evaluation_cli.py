from pathlib import Path
from types import SimpleNamespace

import pytest

import koschei_sentinel.gold_holdout_evaluation_cli as cli_module
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


def test_atomic_export_publishes_only_after_sealed_verification(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = SimpleNamespace(case_count=1)
    destination = tmp_path / "holdout-pack"

    def fake_export(_release_dir, staging):
        staging = Path(staging)
        staging.mkdir()
        (staging / "inputs.jsonl").write_text("fixture\n", encoding="utf-8")
        (staging / "manifest.json").write_text("{}\n", encoding="utf-8")
        return manifest

    monkeypatch.setattr(cli_module, "export_gold_holdout_inference_pack", fake_export)
    monkeypatch.setattr(
        cli_module,
        "_load_inference_pack",
        lambda staging: ([object()], manifest, b"{}\n"),
    )

    result = cli_module._export_inputs_atomic("release", str(destination))

    assert result is manifest
    assert (destination / "inputs.jsonl").read_text(encoding="utf-8") == "fixture\n"
    assert (destination / "manifest.json").is_file()
    assert not list(tmp_path.glob(".holdout-pack.staging-*"))


def test_atomic_export_failure_never_publishes_partial_pack(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = SimpleNamespace(case_count=1)
    destination = tmp_path / "holdout-pack"

    def fake_export(_release_dir, staging):
        staging = Path(staging)
        staging.mkdir()
        (staging / "inputs.jsonl").write_text("partial\n", encoding="utf-8")
        (staging / "manifest.json").write_text("{}\n", encoding="utf-8")
        return manifest

    def fail_sealed_verify(_staging):
        raise ValueError("sealed pack verification failed")

    monkeypatch.setattr(cli_module, "export_gold_holdout_inference_pack", fake_export)
    monkeypatch.setattr(cli_module, "_load_inference_pack", fail_sealed_verify)

    with pytest.raises(ValueError, match="sealed pack verification failed"):
        cli_module._export_inputs_atomic("release", str(destination))

    assert not destination.exists()
    assert not list(tmp_path.glob(".holdout-pack.staging-*"))


def test_atomic_export_rejects_existing_destination_before_build(
    monkeypatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "holdout-pack"
    destination.mkdir()
    called = False

    def forbidden_export(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("exporter must not run for an existing destination")

    monkeypatch.setattr(
        cli_module,
        "export_gold_holdout_inference_pack",
        forbidden_export,
    )

    with pytest.raises(FileExistsError, match="output already exists"):
        cli_module._export_inputs_atomic("release", str(destination))

    assert called is False


def test_raw_candidate_failure_blocks_verified_evaluation(monkeypatch) -> None:
    verified = False
    args = SimpleNamespace(
        candidate_export="candidate-export-link",
        inference_output="output",
        inference_pack="pack",
        release_dir="release",
        policy="policy.json",
    )
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
        raise AssertionError("inference verifier must not run after raw candidate failure")

    monkeypatch.setattr(
        cli_module,
        "verify_gold_holdout_inference_output",
        forbidden_verify,
    )

    with pytest.raises(ValueError, match="must not be a symlink"):
        cli_module._evaluate_verified_output(args)

    assert verified is False
