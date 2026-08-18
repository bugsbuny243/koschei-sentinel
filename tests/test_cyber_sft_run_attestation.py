from __future__ import annotations

import json
from pathlib import Path

import pytest

import koschei_sentinel.cyber_sft_run_attestation as attestation_module
from koschei_sentinel.cyber_sft_artifact_verify import CyberSFTArtifactVerification
from koschei_sentinel.cyber_sft_run_attestation import (
    _config_sha256,
    build_cyber_sft_run_attestation,
)
from koschei_sentinel.cyber_sft_training import CyberSFTConfig


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch) -> dict[str, object]:
    config = CyberSFTConfig(
        run_id="attestation-test",
        stage="DEFENSE_REFLEX",
        base_model="Qwen/Qwen3.5-9B-Base",
        base_revision="a" * 40,
        corpus_dir="build/corpus",
        output_dir="build/run",
        minimum_cuda_memory_gb=0.0,
        quantization={"bits": 4, "compute_dtype": "float16"},
    )
    config_path = tmp_path / "config.json"
    _write_json(config_path, config.model_dump(mode="json"))

    run = tmp_path / "build" / "run"
    run.mkdir(parents=True)
    manifest = {
        "schema_version": "sentinel.cyber-sft-adapter-manifest.v1",
        "run_id": config.run_id,
        "stage": "DEFENSE_REFLEX",
        "execution_profile": "DENSE_SINGLE_GPU_QLORA",
        "base_model": config.base_model,
        "base_revision": config.base_revision,
        "corpus_examples_sha256": "b" * 64,
        "corpus_manifest_sha256": "c" * 64,
        "corpus_promotion_eligible": False,
        "input_adapter_dir": None,
        "adapter_digest": "d" * 64,
        "adapter_files": ["adapter/adapter_model.safetensors"],
        "trainable_target_module_count": 1,
        "trainable_target_modules_sha256": "1" * 64,
        "training_examples": 9,
        "validation_examples": 1,
        "gradient_checkpointing": True,
        "optimizer": "paged_adamw_8bit",
        "output_dir": config.output_dir,
    }
    receipt = {
        "schema_version": "sentinel.cyber-sft-training-receipt.v1",
        "run_id": config.run_id,
        "stage": "DEFENSE_REFLEX",
        "base_model": config.base_model,
        "base_revision": config.base_revision,
        "corpus_examples_sha256": "b" * 64,
        "corpus_manifest_sha256": "c" * 64,
        "corpus_promotion_eligible": False,
        "adapter_digest": "d" * 64,
        "optimizer": "paged_adamw_8bit",
        "global_step": 3,
        "train_metrics": {},
        "eval_metrics": {},
        "cuda_device_index": 0,
        "cuda_device_name": "fake-gpu",
        "cuda_total_memory_gb": 16.0,
        "max_cuda_memory_allocated_gb": 8.0,
        "max_cuda_memory_reserved_gb": 9.0,
        "runtime_versions": {"torch": "test"},
        "receipt_sha256": "e" * 64,
    }
    _write_json(run / "adapter-manifest.json", manifest)
    _write_json(run / "training-receipt.json", receipt)
    _write_json(
        run / "model-runtime.json",
        {
            "expected_model_class": "Qwen3_5ForCausalLM",
            "loader": "AutoModelForCausalLM",
            "model_class": "Qwen3_5ForCausalLM",
            "text_only": True,
        },
    )
    _write_json(
        run / "resume-runtime.json",
        {
            "schema_version": "sentinel.cyber-sft-resume-runtime.v1",
            "resumed": True,
            "resume_checkpoint": "checkpoint-2",
            "checkpoint_every_optimizer_steps": 2,
            "checkpoint_retention": 2,
            "resume_binding": {
                "schema_version": "sentinel.cyber-sft-resume-binding.v1",
                "run_id": config.run_id,
                "base_model": config.base_model,
                "base_revision": config.base_revision,
                "corpus_examples_sha256": "b" * 64,
                "corpus_manifest_sha256": "c" * 64,
                "config_sha256": _config_sha256(config),
            },
        },
    )

    plan_path = tmp_path / "plan.json"
    _write_json(
        plan_path,
        {
            "schema_version": "sentinel.cyber-sft-plan.v1",
            "run_id": config.run_id,
            "stage": "DEFENSE_REFLEX",
            "execution_profile": "DENSE_SINGLE_GPU_QLORA",
            "executable_with_current_trainer": True,
            "corpus_promotion_eligible": False,
            "base_model": config.base_model,
            "base_revision": config.base_revision,
            "corpus_examples_sha256": "b" * 64,
            "corpus_manifest_sha256": "c" * 64,
            "example_count": 10,
            "training_examples": 9,
            "validation_examples": 1,
            "effective_batch_size": config.effective_batch_size,
            "estimated_optimizer_steps": 1,
            "input_adapter_dir": None,
            "output_dir": config.output_dir,
            "warnings": [],
        },
    )

    preflight_path = tmp_path / "model-preflight.json"
    _write_json(
        preflight_path,
        {
            "schema_version": "sentinel.cyber-model-access-preflight.v1",
            "base_model": config.base_model,
            "requested_revision": config.base_revision,
            "resolved_revision": config.base_revision,
            "public_ungated": True,
            "model_type": "qwen3_5",
            "causal_lm_class": "Qwen3_5ForCausalLM",
            "safetensors_files": 2,
            "tokenizer_files_present": True,
            "ready": True,
            "blockers": [],
            "warnings": [],
        },
    )

    verification_payload = {
        "schema_version": "sentinel.cyber-sft-artifact-verification.v1",
        "run_id": config.run_id,
        "valid": True,
        "smoke_only": True,
        "adapter_digest_verified": True,
        "receipt_digest_verified": True,
        "receipt_bindings_verified": True,
        "model_runtime_verified": True,
        "global_step": 3,
        "violations": [],
    }
    verification_path = tmp_path / "verification.json"
    _write_json(verification_path, verification_payload)
    verification = CyberSFTArtifactVerification.model_validate(verification_payload)
    monkeypatch.setattr(attestation_module, "verify_cyber_sft_run", lambda *_args, **_kwargs: verification)

    return {
        "config": config,
        "config_path": config_path,
        "plan_path": plan_path,
        "run": run,
        "preflight_path": preflight_path,
        "verification_path": verification_path,
    }


def _build_kwargs(fixture: dict[str, object], tmp_path: Path) -> dict[str, object]:
    return {
        "config_path": fixture["config_path"],
        "plan_path": fixture["plan_path"],
        "run_dir": "build/run",
        "model_preflight_path": fixture["preflight_path"],
        "verification_path": fixture["verification_path"],
        "selected_profile": "normal",
        "repository_commit": "f" * 40,
        "root": tmp_path,
    }


def test_attestation_is_deterministic_and_binds_resume(monkeypatch, tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    kwargs = _build_kwargs(fixture, tmp_path)

    first = build_cyber_sft_run_attestation(**kwargs)
    second = build_cyber_sft_run_attestation(**kwargs)

    assert first.attestation_sha256 == second.attestation_sha256
    assert first.plan_sha256
    assert first.resumed is True
    assert first.resume_checkpoint == "checkpoint-2"
    assert first.global_step == 3
    assert first.smoke_only is True


def test_attestation_rejects_resolved_revision_drift(monkeypatch, tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    path = fixture["preflight_path"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["resolved_revision"] = "9" * 40
    _write_json(path, payload)

    with pytest.raises(ValueError, match="exact revision"):
        build_cyber_sft_run_attestation(**_build_kwargs(fixture, tmp_path))


def test_attestation_rejects_resume_binding_drift(monkeypatch, tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    path = fixture["run"] / "resume-runtime.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["resume_binding"]["config_sha256"] = "0" * 64
    _write_json(path, payload)

    with pytest.raises(ValueError, match="resume-runtime.json is not bound"):
        build_cyber_sft_run_attestation(**_build_kwargs(fixture, tmp_path))


def test_attestation_rejects_plan_drift(monkeypatch, tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    path = fixture["plan_path"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["base_revision"] = "9" * 40
    _write_json(path, payload)

    with pytest.raises(ValueError, match="plan binding mismatch: base_revision"):
        build_cyber_sft_run_attestation(**_build_kwargs(fixture, tmp_path))
