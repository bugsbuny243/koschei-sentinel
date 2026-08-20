import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import koschei_sentinel.gold_holdout_inference_runner_cli as cli_module
from koschei_sentinel.gold_holdout_inference_runner_cli import build_parser
from tests.test_gold_holdout_pack_admission import _owner_trusted_signed_pack


def _base_args():
    return [
        "--inference-pack",
        "pack",
        "--inference-pack-signature",
        "pack-signature.json",
        "--reviewer-public-key",
        "reviewer-public.pem",
        "--reviewer-trust-policy",
        "reviewer-trust.json",
        "--owner-public-key",
        "owner-public.pem",
        "--candidate-export",
        "candidate-export",
        "--model-ref",
        "sentinel:test",
        "--generation-policy",
        "configs/training/gold-holdout-generation-policy.v1.json",
    ]


def _without(argv: list[str], option: str) -> list[str]:
    result = list(argv)
    index = result.index(option)
    del result[index : index + 2]
    return result


def test_gold_holdout_inference_cli_requires_generation_policy() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(_without(_base_args(), "--generation-policy"))
    assert exc.value.code == 2


def test_gold_holdout_inference_cli_requires_candidate_export() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(_without(_base_args(), "--candidate-export"))
    assert exc.value.code == 2


def test_gold_holdout_inference_cli_requires_signed_pack_identity() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(_without(_base_args(), "--inference-pack-signature"))
    assert exc.value.code == 2


@pytest.mark.parametrize("missing", ["--reviewer-trust-policy", "--owner-public-key"])
def test_gold_holdout_inference_cli_requires_owner_pinned_trust(missing: str) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(_without(_base_args(), missing))
    assert exc.value.code == 2


def test_gold_holdout_inference_cli_accepts_explicit_policies_and_export() -> None:
    args = build_parser().parse_args(_base_args())

    assert args.candidate_export == "candidate-export"
    assert args.inference_pack_signature == "pack-signature.json"
    assert args.reviewer_public_key == "reviewer-public.pem"
    assert args.reviewer_trust_policy == "reviewer-trust.json"
    assert args.owner_public_key == "owner-public.pem"
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


def test_signed_pack_snapshot_is_reverified_after_copy(tmp_path: Path) -> None:
    (
        pack,
        signature_path,
        public_key_path,
        policy_path,
        owner_public_key_path,
        _proof,
    ) = _owner_trusted_signed_pack(tmp_path)
    admission = cli_module._verify_signed_pack(
        inference_pack=str(pack),
        signature_path=str(signature_path),
        reviewer_public_key_path=str(public_key_path),
        reviewer_trust_policy_path=str(policy_path),
        owner_public_key_path=str(owner_public_key_path),
    )
    snapshot = tmp_path / "snapshot" / "pack"
    snapshot.parent.mkdir()

    result = cli_module._snapshot_signed_pack(
        inference_pack=str(pack),
        admission=admission,
        destination=snapshot,
    )

    assert result == snapshot
    assert (snapshot / "inputs.jsonl").read_bytes() == (pack / "inputs.jsonl").read_bytes()
    assert (snapshot / "manifest.json").read_bytes() == (pack / "manifest.json").read_bytes()


def test_signed_pack_snapshot_rejects_manifest_bytes_changed_after_admission(
    tmp_path: Path,
) -> None:
    (
        pack,
        signature_path,
        public_key_path,
        policy_path,
        owner_public_key_path,
        _proof,
    ) = _owner_trusted_signed_pack(tmp_path)
    admission = cli_module._verify_signed_pack(
        inference_pack=str(pack),
        signature_path=str(signature_path),
        reviewer_public_key_path=str(public_key_path),
        reviewer_trust_policy_path=str(policy_path),
        owner_public_key_path=str(owner_public_key_path),
    )

    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    snapshot = tmp_path / "snapshot" / "pack"
    snapshot.parent.mkdir()

    with pytest.raises(ValueError, match="does not bind this inference manifest"):
        cli_module._snapshot_signed_pack(
            inference_pack=str(pack),
            admission=admission,
            destination=snapshot,
        )

    assert not snapshot.exists()


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
