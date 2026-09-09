from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import koschei_sentinel.cyber_sft_export as export_builder
from koschei_sentinel.cyber_sft_export_verify import CyberSFTExportVerification
from koschei_sentinel.cyber_sft_run_attestation import CyberSFTRunAttestation
from koschei_sentinel.cyber_sft_training import CyberSFTConfig, CyberSFTPlan


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _valid_report() -> CyberSFTExportVerification:
    return CyberSFTExportVerification(
        run_id="export-builder-test",
        valid=True,
        attestation_sha256_verified=True,
        config_sha256_verified=True,
        plan_sha256_verified=True,
        training_source_sha256_verified=True,
        model_preflight_sha256_verified=True,
        verification_sha256_verified=True,
        model_runtime_sha256_verified=True,
        resume_runtime_sha256_verified=True,
        fresh_run_verification_valid=True,
        receipt_binding_verified=True,
        adapter_digest_verified=True,
        corpus_examples_sha256_verified=True,
        corpus_manifest_sha256_verified=True,
        profile_binding_verified=True,
        repository_commit_binding_verified=True,
        violations=[],
    )


def _source_fixture(tmp_path: Path, monkeypatch) -> tuple[Path, CyberSFTRunAttestation]:
    root = tmp_path / "source"
    root.mkdir()
    examples_raw = b'{"example_id":"one"}\n'
    manifest_raw = b'{"examples_sha256":"fixture"}\n'
    examples_sha = _sha(examples_raw)
    manifest_sha = _sha(manifest_raw)

    config = CyberSFTConfig(
        run_id="export-builder-test",
        stage="DEFENSE_REFLEX",
        base_model="Qwen/Qwen3.5-9B-Base",
        base_revision="a" * 40,
        corpus_dir="build/corpus",
        validation_ratio=0.10,
        output_dir="build/run",
        minimum_cuda_memory_gb=0.0,
        quantization={"bits": 4, "compute_dtype": "float16"},
    )
    _write_json(root / "config.json", config.model_dump(mode="json"))

    plan = CyberSFTPlan(
        run_id=config.run_id,
        stage="DEFENSE_REFLEX",
        execution_profile="DENSE_SINGLE_GPU_QLORA",
        executable_with_current_trainer=True,
        corpus_promotion_eligible=False,
        base_model=config.base_model,
        base_revision=config.base_revision,
        corpus_examples_sha256=examples_sha,
        corpus_manifest_sha256=manifest_sha,
        explicit_validation=False,
        example_count=1,
        training_examples=1,
        validation_examples=0,
        effective_batch_size=config.effective_batch_size,
        estimated_optimizer_steps=1,
        input_adapter_dir=None,
        output_dir=config.output_dir,
        warnings=[],
    )
    _write_json(root / "plan.json", plan.model_dump(mode="json"))
    _write_json(root / "training-source.json", {"fixture": True})
    _write_json(root / "model-preflight.json", {"fixture": True})
    _write_json(root / "verification.json", {"fixture": True})

    (root / "build/corpus").mkdir(parents=True)
    (root / "build/corpus/examples.jsonl").write_bytes(examples_raw)
    (root / "build/corpus/manifest.json").write_bytes(manifest_raw)

    run = root / "build/run"
    run.mkdir(parents=True)
    _write_json(
        run / "adapter-manifest.json",
        {
            "schema_version": "sentinel.cyber-sft-adapter-manifest.v1",
            "run_id": config.run_id,
            "stage": "DEFENSE_REFLEX",
            "execution_profile": "DENSE_SINGLE_GPU_QLORA",
            "base_model": config.base_model,
            "base_revision": config.base_revision,
            "corpus_examples_sha256": examples_sha,
            "corpus_manifest_sha256": manifest_sha,
            "corpus_promotion_eligible": False,
            "input_adapter_dir": None,
            "adapter_digest": "d" * 64,
            "adapter_files": [],
            "trainable_target_module_count": 1,
            "trainable_target_modules_sha256": "1" * 64,
            "training_examples": 1,
            "validation_examples": 0,
            "gradient_checkpointing": True,
            "optimizer": "paged_adamw_8bit",
            "output_dir": config.output_dir,
        },
    )
    for name in (
        "training-receipt.json",
        "model-runtime.json",
        "resume-runtime.json",
    ):
        _write_json(run / name, {"fixture": name})

    attestation = CyberSFTRunAttestation(
        run_id=config.run_id,
        selected_profile="normal",
        repository_commit="f" * 40,
        base_model=config.base_model,
        base_revision=config.base_revision,
        resolved_model_revision=config.base_revision,
        config_sha256="2" * 64,
        plan_sha256="3" * 64,
        training_source_sha256="4" * 64,
        model_preflight_sha256="5" * 64,
        verification_sha256="6" * 64,
        model_runtime_sha256="7" * 64,
        resume_runtime_sha256="8" * 64,
        receipt_sha256="9" * 64,
        adapter_digest="d" * 64,
        corpus_examples_sha256=examples_sha,
        corpus_manifest_sha256=manifest_sha,
        global_step=1,
        resumed=False,
        resume_checkpoint=None,
        smoke_only=True,
        promotion_eligible=False,
        attestation_sha256="0" * 64,
    )
    _write_json(root / "attestation.json", attestation.model_dump(mode="json"))

    monkeypatch.setattr(
        export_builder,
        "build_cyber_sft_run_attestation",
        lambda **_kwargs: attestation,
    )
    monkeypatch.setattr(
        export_builder,
        "resolve_planned_cyber_sft_corpora",
        lambda *_args, **_kwargs: ([object()], [], examples_sha, manifest_sha, False),
    )
    monkeypatch.setattr(
        export_builder,
        "verify_cyber_sft_export",
        lambda *_args, **_kwargs: _valid_report(),
    )
    return root, attestation


def _build(root: Path, destination: Path):
    return export_builder.build_cyber_sft_export(
        config_path="config.json",
        plan_path="plan.json",
        training_source_path="training-source.json",
        model_preflight_path="model-preflight.json",
        verification_path="verification.json",
        attestation_path="attestation.json",
        output_dir=destination,
        root=root,
    )


def test_exporter_copies_exact_portable_bytes_atomically(tmp_path: Path, monkeypatch) -> None:
    root, attestation = _source_fixture(tmp_path, monkeypatch)
    destination = tmp_path / "candidate-export"

    report = _build(root, destination)

    assert report.valid is True
    assert (destination / "corpus-examples.jsonl").read_bytes() == (
        root / "build/corpus/examples.jsonl"
    ).read_bytes()
    assert (destination / "corpus-manifest.json").read_bytes() == (
        root / "build/corpus/manifest.json"
    ).read_bytes()
    assert (destination / "run/adapter-manifest.json").read_bytes() == (
        root / "build/run/adapter-manifest.json"
    ).read_bytes()
    assert (destination / "selected-profile.txt").read_text(encoding="utf-8") == "normal\n"
    assert (destination / "repository-commit.txt").read_text(encoding="utf-8") == (
        attestation.repository_commit + "\n"
    )
    assert not list(tmp_path.glob(".candidate-export.staging-*"))


def test_exporter_rejects_existing_destination(tmp_path: Path, monkeypatch) -> None:
    root, _ = _source_fixture(tmp_path, monkeypatch)
    destination = tmp_path / "candidate-export"
    destination.mkdir()

    with pytest.raises(ValueError, match="destination already exists"):
        _build(root, destination)


def test_exporter_rejects_symlinked_run_artifact(tmp_path: Path, monkeypatch) -> None:
    root, _ = _source_fixture(tmp_path, monkeypatch)
    runtime = root / "build/run/model-runtime.json"
    target = root / "runtime-target.json"
    target.write_bytes(runtime.read_bytes())
    runtime.unlink()
    runtime.symlink_to(target)

    with pytest.raises(ValueError, match="must not traverse a symlink"):
        _build(root, tmp_path / "candidate-export")


def test_exporter_rejects_stale_attestation_rebuild(tmp_path: Path, monkeypatch) -> None:
    root, attestation = _source_fixture(tmp_path, monkeypatch)
    stale = attestation.model_copy(update={"selected_profile": "micro"})
    monkeypatch.setattr(
        export_builder,
        "build_cyber_sft_run_attestation",
        lambda **_kwargs: stale,
    )

    with pytest.raises(ValueError, match="differs from a fresh source rebuild"):
        _build(root, tmp_path / "candidate-export")


def test_exporter_removes_staging_on_final_verification_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root, _ = _source_fixture(tmp_path, monkeypatch)
    invalid = _valid_report().model_copy(
        update={"valid": False, "violations": ["fixture failure"]}
    )
    monkeypatch.setattr(
        export_builder,
        "verify_cyber_sft_export",
        lambda *_args, **_kwargs: invalid,
    )
    destination = tmp_path / "candidate-export"

    with pytest.raises(ValueError, match="fixture failure"):
        _build(root, destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".candidate-export.staging-*"))
