from pathlib import Path
from types import SimpleNamespace

import pytest

import koschei_sentinel.gold_holdout_evaluation_cli as evaluation_cli
import koschei_sentinel.gold_holdout_inference_runner_cli as runner_cli
import koschei_sentinel.gold_holdout_inference_verify_cli as verify_cli
import koschei_sentinel.gold_holdout_pack_admission as admission_module


def test_snapshot_admission_copies_then_reverifies(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "pack"
    source.mkdir()
    (source / "inputs.jsonl").write_text("fixture\n", encoding="utf-8")
    (source / "manifest.json").write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "snapshot"
    admission = object()
    observed: list[Path] = []

    def fake_verify(given_admission, pack) -> None:
        assert given_admission is admission
        observed.append(Path(pack))

    monkeypatch.setattr(
        admission_module,
        "verify_admitted_gold_holdout_pack",
        fake_verify,
    )

    snapshot = admission_module.snapshot_admitted_gold_holdout_pack(
        admission,
        source,
        destination,
    )

    assert snapshot == destination
    assert observed == [destination]
    assert (snapshot / "inputs.jsonl").read_text(encoding="utf-8") == "fixture\n"
    assert (snapshot / "manifest.json").read_text(encoding="utf-8") == "{}\n"


def test_snapshot_admission_failure_removes_untrusted_copy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "pack"
    source.mkdir()
    (source / "inputs.jsonl").write_text("fixture\n", encoding="utf-8")
    (source / "manifest.json").write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "snapshot"

    monkeypatch.setattr(
        admission_module,
        "verify_admitted_gold_holdout_pack",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("snapshot rejected")),
    )

    with pytest.raises(ValueError, match="snapshot rejected"):
        admission_module.snapshot_admitted_gold_holdout_pack(
            object(),
            source,
            destination,
        )

    assert not destination.exists()


def test_inference_cli_plan_and_execute_consume_same_snapshots(monkeypatch) -> None:
    admission = object()
    policy = object()
    observed_plan = None
    observed_execute = None

    monkeypatch.setattr(runner_cli, "_load_policy", lambda *_args: policy)
    monkeypatch.setattr(runner_cli, "_verify_signed_pack", lambda **_kwargs: admission)
    monkeypatch.setattr(runner_cli, "_assert_raw_candidate_export", lambda *_args: None)
    monkeypatch.setattr(
        runner_cli,
        "_snapshot_signed_pack",
        lambda **_kwargs: Path("revalidated-pack"),
    )
    monkeypatch.setattr(
        runner_cli,
        "snapshot_verified_cyber_sft_export",
        lambda candidate_export, destination: Path("revalidated-candidate"),
    )

    def fake_plan(**kwargs):
        nonlocal observed_plan
        observed_plan = (
            Path(kwargs["inference_pack_dir"]),
            Path(kwargs["candidate_export_dir"]),
        )
        return SimpleNamespace(
            model_dump=lambda **_kwargs: {"plan_sha256": "a" * 64},
        )

    def fake_execute(**kwargs):
        nonlocal observed_execute
        observed_execute = (
            Path(kwargs["inference_pack"]),
            Path(kwargs["candidate_export"]),
        )
        return SimpleNamespace(
            model_dump=lambda **_kwargs: {"receipt_sha256": "b" * 64},
        )

    monkeypatch.setattr(runner_cli, "build_gold_holdout_inference_plan", fake_plan)
    monkeypatch.setattr(runner_cli, "_execute_atomic", fake_execute)

    status = runner_cli.main(
        [
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
            "generation-policy.json",
            "--execute",
            "--output-dir",
            "output",
        ]
    )

    expected = (Path("revalidated-pack"), Path("revalidated-candidate"))
    assert status == 0
    assert observed_plan == expected
    assert observed_execute == expected


def test_offline_verify_cli_consumes_revalidated_snapshots(monkeypatch) -> None:
    admission = object()
    observed_pack = None
    observed_candidate = None

    monkeypatch.setattr(verify_cli, "_verify_signed_pack", lambda **_kwargs: admission)
    monkeypatch.setattr(verify_cli, "_assert_raw_candidate_export", lambda *_args: None)

    def fake_pack_snapshot(given_admission, inference_pack, destination):
        assert given_admission is admission
        assert inference_pack == "pack"
        assert Path(destination).name == "pack"
        return Path("revalidated-pack")

    def fake_candidate_snapshot(candidate_export, destination):
        assert candidate_export == "candidate-export"
        assert Path(destination).name == "candidate-export"
        return Path("revalidated-candidate")

    def fake_verify(output_dir, inference_pack, candidate_export):
        nonlocal observed_pack, observed_candidate
        assert output_dir == "output"
        observed_pack = Path(inference_pack)
        observed_candidate = Path(candidate_export)
        return SimpleNamespace(
            valid=True,
            model_dump=lambda **_kwargs: {"valid": True},
        )

    monkeypatch.setattr(
        verify_cli,
        "snapshot_admitted_gold_holdout_pack",
        fake_pack_snapshot,
    )
    monkeypatch.setattr(
        verify_cli,
        "snapshot_verified_cyber_sft_export",
        fake_candidate_snapshot,
    )
    monkeypatch.setattr(verify_cli, "verify_gold_holdout_inference_output", fake_verify)

    status = verify_cli.main(
        [
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
    )

    assert status == 0
    assert observed_pack == Path("revalidated-pack")
    assert observed_candidate == Path("revalidated-candidate")


def test_evaluate_output_passes_snapshots_to_inference_verifier(monkeypatch) -> None:
    admission = object()
    observed_pack = None
    observed_candidate = None
    args = SimpleNamespace(
        candidate_export="candidate-export",
        inference_output="output",
        inference_pack="pack",
        inference_pack_signature="pack-signature.json",
        reviewer_public_key="reviewer-public.pem",
        release_dir="release",
        policy="policy.json",
    )

    monkeypatch.setattr(evaluation_cli, "_verify_signed_pack", lambda **_kwargs: admission)
    monkeypatch.setattr(evaluation_cli, "_assert_raw_candidate_export", lambda *_args: None)
    monkeypatch.setattr(
        evaluation_cli,
        "snapshot_admitted_gold_holdout_pack",
        lambda given_admission, inference_pack, destination: Path("revalidated-pack"),
    )
    monkeypatch.setattr(
        evaluation_cli,
        "snapshot_verified_cyber_sft_export",
        lambda candidate_export, destination: Path("revalidated-candidate"),
    )

    def stop_after_snapshot(output_dir, inference_pack, candidate_export):
        nonlocal observed_pack, observed_candidate
        assert output_dir == "output"
        observed_pack = Path(inference_pack)
        observed_candidate = Path(candidate_export)
        raise ValueError("stop after snapshot verifier call")

    monkeypatch.setattr(
        evaluation_cli,
        "verify_gold_holdout_inference_output",
        stop_after_snapshot,
    )

    with pytest.raises(ValueError, match="stop after snapshot verifier call"):
        evaluation_cli._evaluate_verified_output(args)

    assert observed_pack == Path("revalidated-pack")
    assert observed_candidate == Path("revalidated-candidate")
