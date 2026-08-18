from __future__ import annotations

import hashlib
import json
from pathlib import Path

import koschei_sentinel.cyber_sft_export_verify as export_module
from koschei_sentinel.cyber_sft_artifact_verify import CyberSFTArtifactVerification
from koschei_sentinel.cyber_sft_export_verify import verify_cyber_sft_export
from koschei_sentinel.cyber_sft_run_attestation import (
    _attestation_digest,
    _config_sha256,
)
from koschei_sentinel.cyber_sft_training import CyberSFTConfig


def _write_json(path: Path, payload: dict[str, object]) -> bytes:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _build_export(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "export"
    root.mkdir()

    config = CyberSFTConfig(
        run_id="portable-export-test",
        stage="DEFENSE_REFLEX",
        base_model="Qwen/Qwen3.5-9B-Base",
        base_revision="a" * 40,
        corpus_dir="build/corpus",
        output_dir="build/run",
        minimum_cuda_memory_gb=0.0,
        quantization={"bits": 4, "compute_dtype": "float16"},
    )
    _write_json(
        root / "training-config.json",
        config.model_dump(mode="json"),
    )
    config_sha = _config_sha256(config)

    examples_raw = b'{"example_id":"portable"}\n'
    (root / "corpus-examples.jsonl").write_bytes(examples_raw)
    examples_sha = _sha(examples_raw)
    corpus_manifest_raw = _write_json(
        root / "corpus-manifest.json",
        {
            "schema_version": "sentinel.defense-reflex-corpus-manifest.v3",
            "examples_sha256": examples_sha,
        },
    )
    corpus_manifest_sha = _sha(corpus_manifest_raw)

    plan_raw = _write_json(
        root / "training-plan.json",
        {
            "schema_version": "sentinel.cyber-sft-plan.v1",
            "run_id": config.run_id,
            "stage": "DEFENSE_REFLEX",
            "execution_profile": "DENSE_SINGLE_GPU_QLORA",
            "executable_with_current_trainer": True,
            "corpus_promotion_eligible": False,
            "base_model": config.base_model,
            "base_revision": config.base_revision,
            "corpus_examples_sha256": examples_sha,
            "corpus_manifest_sha256": corpus_manifest_sha,
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

    preflight_raw = _write_json(
        root / "model-preflight.json",
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
    verification_raw = _write_json(root / "verification.json", verification_payload)
    verification = CyberSFTArtifactVerification.model_validate(verification_payload)
    monkeypatch.setattr(
        export_module,
        "verify_cyber_sft_run",
        lambda *_args, **_kwargs: verification,
    )

    adapter_digest = "d" * 64
    receipt_sha = "e" * 64
    run = root / "run"
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
            "corpus_manifest_sha256": corpus_manifest_sha,
            "corpus_promotion_eligible": False,
            "input_adapter_dir": None,
            "adapter_digest": adapter_digest,
            "adapter_files": [],
            "trainable_target_module_count": 1,
            "trainable_target_modules_sha256": "1" * 64,
            "training_examples": 9,
            "validation_examples": 1,
            "gradient_checkpointing": True,
            "optimizer": "paged_adamw_8bit",
            "output_dir": config.output_dir,
        },
    )
    _write_json(
        run / "training-receipt.json",
        {
            "schema_version": "sentinel.cyber-sft-training-receipt.v1",
            "run_id": config.run_id,
            "stage": "DEFENSE_REFLEX",
            "base_model": config.base_model,
            "base_revision": config.base_revision,
            "corpus_examples_sha256": examples_sha,
            "corpus_manifest_sha256": corpus_manifest_sha,
            "corpus_promotion_eligible": False,
            "adapter_digest": adapter_digest,
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
            "receipt_sha256": receipt_sha,
        },
    )
    model_runtime_raw = _write_json(
        run / "model-runtime.json",
        {
            "expected_model_class": "Qwen3_5ForCausalLM",
            "loader": "AutoModelForCausalLM",
            "model_class": "Qwen3_5ForCausalLM",
            "text_only": True,
            "requested_compute_dtype": "float16",
            "observed_floating_dtypes_before_kbit_prepare": [
                "float16",
                "float32",
            ],
        },
    )
    resume_runtime_raw = _write_json(
        run / "resume-runtime.json",
        {
            "schema_version": "sentinel.cyber-sft-resume-runtime.v1",
            "resumed": False,
            "resume_checkpoint": None,
            "checkpoint_every_optimizer_steps": 2,
            "checkpoint_retention": 2,
            "resume_binding": {
                "schema_version": "sentinel.cyber-sft-resume-binding.v1",
                "run_id": config.run_id,
                "base_model": config.base_model,
                "base_revision": config.base_revision,
                "corpus_examples_sha256": examples_sha,
                "corpus_manifest_sha256": corpus_manifest_sha,
                "config_sha256": config_sha,
            },
        },
    )

    (root / "selected-profile.txt").write_text("normal\n", encoding="utf-8")
    (root / "repository-commit.txt").write_text(
        "f" * 40 + "\n",
        encoding="utf-8",
    )

    attestation_payload: dict[str, object] = {
        "schema_version": "sentinel.cyber-sft-run-attestation.v1",
        "run_id": config.run_id,
        "selected_profile": "normal",
        "repository_commit": "f" * 40,
        "base_model": config.base_model,
        "base_revision": config.base_revision,
        "resolved_model_revision": config.base_revision,
        "config_sha256": config_sha,
        "plan_sha256": _sha(plan_raw),
        "model_preflight_sha256": _sha(preflight_raw),
        "verification_sha256": _sha(verification_raw),
        "model_runtime_sha256": _sha(model_runtime_raw),
        "resume_runtime_sha256": _sha(resume_runtime_raw),
        "receipt_sha256": receipt_sha,
        "adapter_digest": adapter_digest,
        "corpus_examples_sha256": examples_sha,
        "corpus_manifest_sha256": corpus_manifest_sha,
        "global_step": 3,
        "resumed": False,
        "resume_checkpoint": None,
        "smoke_only": True,
        "promotion_eligible": False,
    }
    attestation_payload["attestation_sha256"] = _attestation_digest(
        attestation_payload
    )
    _write_json(root / "run-attestation.json", attestation_payload)
    return root


def test_portable_export_verifies_when_all_bindings_match(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = _build_export(tmp_path, monkeypatch)

    report = verify_cyber_sft_export(root)

    assert report.valid is True
    assert report.attestation_sha256_verified is True
    assert report.plan_sha256_verified is True
    assert report.corpus_examples_sha256_verified is True
    assert report.corpus_manifest_sha256_verified is True
    assert report.violations == []


def test_portable_export_rejects_corpus_example_tampering(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = _build_export(tmp_path, monkeypatch)
    with (root / "corpus-examples.jsonl").open("ab") as handle:
        handle.write(b'{"tampered":true}\n')

    report = verify_cyber_sft_export(root)

    assert report.valid is False
    assert report.corpus_examples_sha256_verified is False
    assert any("corpus examples" in row for row in report.violations)


def test_portable_export_rejects_plan_tampering(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = _build_export(tmp_path, monkeypatch)
    path = root / "training-plan.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["warnings"] = ["tampered"]
    _write_json(path, payload)

    report = verify_cyber_sft_export(root)

    assert report.valid is False
    assert report.plan_sha256_verified is False
    assert any("execution plan SHA-256" in row for row in report.violations)


def test_portable_export_rejects_dtype_semantic_leak(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = _build_export(tmp_path, monkeypatch)
    runtime_path = root / "run" / "model-runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["observed_floating_dtypes_before_kbit_prepare"].append("bfloat16")
    runtime_raw = _write_json(runtime_path, runtime)

    attestation_path = root / "run-attestation.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["model_runtime_sha256"] = _sha(runtime_raw)
    attestation["attestation_sha256"] = _attestation_digest(attestation)
    _write_json(attestation_path, attestation)

    report = verify_cyber_sft_export(root)

    assert report.valid is False
    assert report.model_runtime_sha256_verified is True
    assert any("runtime semantic bindings" in row for row in report.violations)
