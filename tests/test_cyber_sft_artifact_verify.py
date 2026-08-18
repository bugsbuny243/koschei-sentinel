import hashlib
import json

from koschei_sentinel.cyber_sft_artifact_verify import verify_cyber_sft_run
from koschei_sentinel.cyber_sft_trainer import (
    CyberSFTAdapterManifest,
    CyberSFTTrainingReceipt,
)
from koschei_sentinel.cyber_sft_training import CyberExecutionProfile
from koschei_sentinel.training import canonical_json


def _directory_digest(root, files):
    digest = hashlib.sha256()
    for relative in files:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_run(tmp_path, *, global_step=8):
    run = tmp_path / "build" / "run"
    adapter = run / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"sentinel-adapter")
    files = ["adapter/adapter_model.safetensors"]
    adapter_digest = _directory_digest(run, files)

    manifest = CyberSFTAdapterManifest(
        run_id="cyber-sft-test",
        stage="DEFENSE_REFLEX",
        execution_profile=CyberExecutionProfile.DENSE_SINGLE_GPU_QLORA,
        base_model="Qwen/Qwen3.5-9B-Base",
        base_revision="a" * 40,
        corpus_examples_sha256="b" * 64,
        corpus_manifest_sha256="c" * 64,
        corpus_promotion_eligible=False,
        input_adapter_dir=None,
        adapter_digest=adapter_digest,
        adapter_files=files,
        trainable_target_module_count=4,
        trainable_target_modules_sha256="d" * 64,
        training_examples=29,
        validation_examples=3,
        gradient_checkpointing=True,
        optimizer="paged_adamw_8bit",
        output_dir="build/run",
    )
    (run / "adapter-manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run / "model-runtime.json").write_text(
        json.dumps(
            {
                "loader": "AutoModelForCausalLM",
                "model_class": "Qwen3_5ForCausalLM",
                "expected_model_class": "Qwen3_5ForCausalLM",
                "text_only": True,
                "requested_compute_dtype": "float16",
                "observed_floating_dtypes_before_kbit_prepare": [
                    "float16",
                    "float32",
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    payload = {
        "schema_version": "sentinel.cyber-sft-training-receipt.v1",
        "run_id": manifest.run_id,
        "stage": manifest.stage,
        "base_model": manifest.base_model,
        "base_revision": manifest.base_revision,
        "corpus_examples_sha256": manifest.corpus_examples_sha256,
        "corpus_manifest_sha256": manifest.corpus_manifest_sha256,
        "corpus_promotion_eligible": manifest.corpus_promotion_eligible,
        "adapter_digest": manifest.adapter_digest,
        "optimizer": manifest.optimizer,
        "global_step": global_step,
        "train_metrics": {"train_loss": 1.25},
        "eval_metrics": {"eval_loss": 1.10},
        "cuda_device_index": 0,
        "cuda_device_name": "Fake GPU",
        "cuda_total_memory_gb": 15.0,
        "max_cuda_memory_allocated_gb": 9.0,
        "max_cuda_memory_reserved_gb": 10.0,
        "runtime_versions": {
            "torch": "test",
            "transformers": "test",
            "peft": "test",
            "bitsandbytes": "test",
            "accelerate": "test",
            "datasets": "test",
        },
    }
    receipt_sha = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    receipt = CyberSFTTrainingReceipt(**payload, receipt_sha256=receipt_sha)
    (run / "training-receipt.json").write_text(
        json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run


def test_completed_smoke_run_verifies(tmp_path) -> None:
    _write_run(tmp_path)
    report = verify_cyber_sft_run("build/run", root=tmp_path)

    assert report.valid is True
    assert report.smoke_only is True
    assert report.adapter_digest_verified is True
    assert report.receipt_digest_verified is True
    assert report.receipt_bindings_verified is True
    assert report.model_runtime_verified is True
    assert report.global_step == 8
    assert report.violations == []


def test_adapter_tamper_is_detected(tmp_path) -> None:
    run = _write_run(tmp_path)
    (run / "adapter" / "adapter_model.safetensors").write_bytes(b"tampered")

    report = verify_cyber_sft_run("build/run", root=tmp_path)
    assert report.valid is False
    assert report.adapter_digest_verified is False


def test_receipt_tamper_is_detected(tmp_path) -> None:
    run = _write_run(tmp_path)
    receipt_path = run / "training-receipt.json"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["cuda_device_name"] = "tampered-gpu"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    report = verify_cyber_sft_run("build/run", root=tmp_path)
    assert report.valid is False
    assert report.receipt_digest_verified is False


def test_wrong_model_runtime_is_detected(tmp_path) -> None:
    run = _write_run(tmp_path)
    runtime_path = run / "model-runtime.json"
    payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    payload["text_only"] = False
    runtime_path.write_text(json.dumps(payload), encoding="utf-8")

    report = verify_cyber_sft_run("build/run", root=tmp_path)
    assert report.valid is False
    assert report.model_runtime_verified is False
    assert any("model runtime mismatch" in row for row in report.violations)


def test_bfloat16_leak_is_detected_for_float16_run(tmp_path) -> None:
    run = _write_run(tmp_path)
    runtime_path = run / "model-runtime.json"
    payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    payload["observed_floating_dtypes_before_kbit_prepare"] = [
        "float16",
        "bfloat16",
        "float32",
    ]
    runtime_path.write_text(json.dumps(payload), encoding="utf-8")

    report = verify_cyber_sft_run("build/run", root=tmp_path)
    assert report.valid is False
    assert report.model_runtime_verified is False
    assert any("competing low-precision dtype" in row for row in report.violations)


def test_zero_step_run_is_not_real_training(tmp_path) -> None:
    _write_run(tmp_path, global_step=0)
    report = verify_cyber_sft_run("build/run", root=tmp_path)

    assert report.valid is False
    assert report.receipt_digest_verified is True
    assert report.model_runtime_verified is True
    assert report.global_step == 0
    assert any("zero optimizer steps" in row for row in report.violations)
