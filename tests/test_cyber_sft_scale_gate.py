from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import koschei_sentinel.cyber_sft_scale_gate as scale_module
from koschei_sentinel.cyber_sft_scale_gate import (
    CyberSFTScaleDisposition,
    evaluate_9b_scale_gate,
)

_MICRO_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
_TARGET_SUFFIXES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _micro_config_payload(
    *,
    bits: int = 4,
    dtype: str = "float16",
    max_sequence_length: int = 2048,
    suffixes: list[str] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "sentinel.cyber-sft-config.v1",
        "run_id": "micro-test",
        "stage": "DEFENSE_REFLEX",
        "execution_profile": "DENSE_SINGLE_GPU_QLORA",
        "base_model": "Qwen/Qwen3.5-0.8B-Base",
        "base_revision": _MICRO_REVISION,
        "corpus_dir": "build/corpus",
        "output_dir": "build/run",
        "input_adapter_dir": None,
        "max_sequence_length": max_sequence_length,
        "epochs": 1.0,
        "learning_rate": 0.0001,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "validation_ratio": 0.1,
        "warmup_ratio": 0.03,
        "logging_steps": 1,
        "seed": 1701,
        "minimum_cuda_memory_gb": 6.0,
        "gradient_checkpointing": True,
        "enable_router_aux_loss": False,
        "quantization": {
            "bits": bits,
            "quant_type": "nf4",
            "double_quant": True,
            "compute_dtype": dtype,
        },
        "lora": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_suffixes": suffixes or list(_TARGET_SUFFIXES),
        },
    }


def _micro_export(
    tmp_path: Path,
    *,
    total_memory_gb: float,
    fast_warning: bool,
    bits: int = 4,
    dtype: str = "float16",
    max_sequence_length: int = 2048,
    suffixes: list[str] | None = None,
) -> Path:
    root = tmp_path / "micro"
    run = root / "run"
    run.mkdir(parents=True)
    _write_json(
        root / "training-config.json",
        _micro_config_payload(
            bits=bits,
            dtype=dtype,
            max_sequence_length=max_sequence_length,
            suffixes=suffixes,
        ),
    )

    _write_json(
        root / "run-attestation.json",
        {
            "schema_version": "sentinel.cyber-sft-run-attestation.v1",
            "run_id": "micro-test",
            "selected_profile": "micro",
            "repository_commit": "f" * 40,
            "base_model": "Qwen/Qwen3.5-0.8B-Base",
            "base_revision": _MICRO_REVISION,
            "resolved_model_revision": _MICRO_REVISION,
            "config_sha256": "1" * 64,
            "plan_sha256": "2" * 64,
            "model_preflight_sha256": "3" * 64,
            "verification_sha256": "4" * 64,
            "model_runtime_sha256": "5" * 64,
            "resume_runtime_sha256": "6" * 64,
            "receipt_sha256": "7" * 64,
            "adapter_digest": "8" * 64,
            "corpus_examples_sha256": "9" * 64,
            "corpus_manifest_sha256": "a" * 64,
            "global_step": 3,
            "resumed": False,
            "resume_checkpoint": None,
            "smoke_only": True,
            "promotion_eligible": False,
            "attestation_sha256": "b" * 64,
        },
    )
    _write_json(
        run / "training-receipt.json",
        {
            "schema_version": "sentinel.cyber-sft-training-receipt.v1",
            "run_id": "micro-test",
            "stage": "DEFENSE_REFLEX",
            "base_model": "Qwen/Qwen3.5-0.8B-Base",
            "base_revision": _MICRO_REVISION,
            "corpus_examples_sha256": "9" * 64,
            "corpus_manifest_sha256": "a" * 64,
            "corpus_promotion_eligible": False,
            "adapter_digest": "8" * 64,
            "optimizer": "paged_adamw_8bit",
            "global_step": 3,
            "train_metrics": {},
            "eval_metrics": {},
            "cuda_device_index": 0,
            "cuda_device_name": "test-gpu",
            "cuda_total_memory_gb": total_memory_gb,
            "max_cuda_memory_allocated_gb": 4.0,
            "max_cuda_memory_reserved_gb": 5.0,
            "runtime_versions": {"torch": "test"},
            "receipt_sha256": "7" * 64,
        },
    )
    if fast_warning:
        _write_json(
            run / "runtime-warnings.json",
            {
                "warnings": [
                    "fla is unavailable; Qwen3.5 DeltaNet may use slower kernels"
                ]
            },
        )
    return root


def _target_config() -> str:
    return "configs/training/cyber-sft.qwen3.5-9b.smoke.json"


def _mark_export_valid(monkeypatch) -> None:
    monkeypatch.setattr(
        scale_module,
        "verify_cyber_sft_export",
        lambda _root: SimpleNamespace(valid=True),
    )


def test_valid_micro_on_sufficient_gpu_permits_9b(monkeypatch, tmp_path: Path) -> None:
    _mark_export_valid(monkeypatch)
    root = _micro_export(tmp_path, total_memory_gb=16.0, fast_warning=False)

    report = evaluate_9b_scale_gate(
        micro_export_dir=root,
        target_config_path=_target_config(),
    )

    assert report.allowed_to_attempt_9b is True
    assert report.disposition is CyberSFTScaleDisposition.PROCEED_9B
    assert report.quantization_bits_match is True
    assert report.compute_dtype_match is True
    assert report.lora_target_coverage is True
    assert report.micro_max_sequence_length == report.target_max_sequence_length == 2048
    assert report.blockers == []
    assert report.warnings == []


def test_missing_fast_kernels_is_caution_not_block(monkeypatch, tmp_path: Path) -> None:
    _mark_export_valid(monkeypatch)
    root = _micro_export(tmp_path, total_memory_gb=16.0, fast_warning=True)

    report = evaluate_9b_scale_gate(
        micro_export_dir=root,
        target_config_path=_target_config(),
    )

    assert report.allowed_to_attempt_9b is True
    assert report.disposition is CyberSFTScaleDisposition.PROCEED_9B_CAUTION
    assert report.fast_kernel_warnings
    assert report.warnings


def test_gpu_below_9b_minimum_blocks_scale_up(monkeypatch, tmp_path: Path) -> None:
    _mark_export_valid(monkeypatch)
    root = _micro_export(tmp_path, total_memory_gb=12.0, fast_warning=False)

    report = evaluate_9b_scale_gate(
        micro_export_dir=root,
        target_config_path=_target_config(),
    )

    assert report.allowed_to_attempt_9b is False
    assert report.disposition is CyberSFTScaleDisposition.BLOCK_9B
    assert any("below the 9B config minimum" in row for row in report.blockers)


def test_quantization_recipe_mismatch_blocks_scale_up(monkeypatch, tmp_path: Path) -> None:
    _mark_export_valid(monkeypatch)
    root = _micro_export(
        tmp_path,
        total_memory_gb=16.0,
        fast_warning=False,
        bits=8,
    )

    report = evaluate_9b_scale_gate(
        micro_export_dir=root,
        target_config_path=_target_config(),
    )

    assert report.allowed_to_attempt_9b is False
    assert report.quantization_bits_match is False
    assert any("quantization bits" in row for row in report.blockers)


def test_missing_lora_target_class_blocks_scale_up(monkeypatch, tmp_path: Path) -> None:
    _mark_export_valid(monkeypatch)
    root = _micro_export(
        tmp_path,
        total_memory_gb=16.0,
        fast_warning=False,
        suffixes=[row for row in _TARGET_SUFFIXES if row != "in_proj_qkv"],
    )

    report = evaluate_9b_scale_gate(
        micro_export_dir=root,
        target_config_path=_target_config(),
    )

    assert report.allowed_to_attempt_9b is False
    assert report.lora_target_coverage is False
    assert report.missing_target_suffixes == ["in_proj_qkv"]


def test_shorter_micro_context_is_caution(monkeypatch, tmp_path: Path) -> None:
    _mark_export_valid(monkeypatch)
    root = _micro_export(
        tmp_path,
        total_memory_gb=16.0,
        fast_warning=False,
        max_sequence_length=1024,
    )

    report = evaluate_9b_scale_gate(
        micro_export_dir=root,
        target_config_path=_target_config(),
    )

    assert report.allowed_to_attempt_9b is True
    assert report.disposition is CyberSFTScaleDisposition.PROCEED_9B_CAUTION
    assert report.micro_max_sequence_length == 1024
    assert report.target_max_sequence_length == 2048
    assert any("context length" in row for row in report.warnings)
