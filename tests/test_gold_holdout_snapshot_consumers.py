from pathlib import Path
from types import SimpleNamespace

import pytest

import koschei_sentinel.gold_holdout_evaluation_cli as evaluation_cli
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


def test_offline_verify_cli_consumes_revalidated_snapshot(monkeypatch) -> None:
    admission = object()
    observed_pack = None

    monkeypatch.setattr(verify_cli, "_verify_signed_pack", lambda **_kwargs: admission)
    monkeypatch.setattr(verify_cli, "_assert_raw_candidate_export", lambda *_args: None)

    def fake_snapshot(given_admission, inference_pack, destination):
        assert given_admission is admission
        assert inference_pack == "pack"
        assert Path(destination).name == "pack"
        return Path("revalidated-snapshot")

    def fake_verify(output_dir, inference_pack, candidate_export):
        nonlocal observed_pack
        assert output_dir == "output"
        assert candidate_export == "candidate-export"
        observed_pack = Path(inference_pack)
        return SimpleNamespace(
            valid=True,
            model_dump=lambda **_kwargs: {"valid": True},
        )

    monkeypatch.setattr(verify_cli, "snapshot_admitted_gold_holdout_pack", fake_snapshot)
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
    assert observed_pack == Path("revalidated-snapshot")


def test_evaluate_output_passes_snapshot_to_inference_verifier(monkeypatch) -> None:
    admission = object()
    observed_pack = None
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
        lambda given_admission, inference_pack, destination: Path("revalidated-snapshot"),
    )

    def stop_after_snapshot(output_dir, inference_pack, candidate_export):
        nonlocal observed_pack
        assert output_dir == "output"
        assert candidate_export == "candidate-export"
        observed_pack = Path(inference_pack)
        raise ValueError("stop after snapshot verifier call")

    monkeypatch.setattr(
        evaluation_cli,
        "verify_gold_holdout_inference_output",
        stop_after_snapshot,
    )

    with pytest.raises(ValueError, match="stop after snapshot verifier call"):
        evaluation_cli._evaluate_verified_output(args)

    assert observed_pack == Path("revalidated-snapshot")
