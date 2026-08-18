from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import koschei_sentinel.cyber_sft_scale_gate as scale_module
from koschei_sentinel.cyber_sft_scale_gate import (
    CyberSFTScaleDisposition,
    evaluate_9b_scale_gate,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _micro_export(tmp_path: Path, *, total_memory_gb: float, fast_warning: bool) -> Path:
    root = tmp_path / "micro"
    run = root / "run"
    run.mkdir(parents=True)

    _write_json(
        root / "run-attestation.json",
        {
            "schema_version": "sentinel.cyber-sft-run-attestation.v1",
            "run_id": "micro-test",
            "selected_profile": "micro",
            "repository_commit": "f" * 40,
            "base_model": "Qwen/Qwen3.5-0.8B-Base",
            "base_revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
            "resolved_model_revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
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
            "base_revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
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


def test_valid_micro_on_sufficient_gpu_permits_9b(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(scale_module, "verify_cyber_sft_export", lambda _root: SimpleNamespace(valid=True))
    root = _micro_export(tmp_path, total_memory_gb=16.0, fast_warning=False)

    report = evaluate_9b_scale_gate(
        micro_export_dir=root,
        target_config_path=_target_config(),
    )

    assert report.allowed_to_attempt_9b is True
    assert report.disposition is CyberSFTScaleDisposition.PROCEED_9B
    assert report.blockers == []
    assert report.warnings == []


def test_missing_fast_kernels_is_caution_not_block(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(scale_module, "verify_cyber_sft_export", lambda _root: SimpleNamespace(valid=True))
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
    monkeypatch.setattr(scale_module, "verify_cyber_sft_export", lambda _root: SimpleNamespace(valid=True))
    root = _micro_export(tmp_path, total_memory_gb=12.0, fast_warning=False)

    report = evaluate_9b_scale_gate(
        micro_export_dir=root,
        target_config_path=_target_config(),
    )

    assert report.allowed_to_attempt_9b is False
    assert report.disposition is CyberSFTScaleDisposition.BLOCK_9B
    assert any("below the 9B config minimum" in row for row in report.blockers)
