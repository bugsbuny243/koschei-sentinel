from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import koschei_sentinel.cyber_sft_run_reuse as reuse_module
from koschei_sentinel.cyber_sft_run_reuse import evaluate_completed_run_reuse
from koschei_sentinel.cyber_sft_training import CyberSFTConfig
from koschei_sentinel.cyber_sft_training_source import (
    build_training_source_binding,
    verify_training_source_binding,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, CyberSFTConfig]:
    config = CyberSFTConfig(
        run_id="source-binding-test",
        stage="DEFENSE_REFLEX",
        base_model="Qwen/Qwen3.5-0.8B-Base",
        base_revision="a" * 40,
        corpus_dir="build/corpus",
        output_dir="build/run",
        minimum_cuda_memory_gb=0.0,
    )
    config_path = tmp_path / "config.json"
    _write_json(config_path, config.model_dump(mode="json"))
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
    source_path = tmp_path / "training-source.json"
    source = build_training_source_binding(
        config_path=config_path,
        plan_path=plan_path,
        repository_commit="f" * 40,
    )
    _write_json(source_path, source.model_dump(mode="json"))
    return config_path, plan_path, source_path, config


def test_training_source_binding_is_deterministic(tmp_path: Path) -> None:
    config_path, plan_path, _source_path, _config = _fixture(tmp_path)

    first = build_training_source_binding(
        config_path=config_path,
        plan_path=plan_path,
        repository_commit="f" * 40,
    )
    second = build_training_source_binding(
        config_path=config_path,
        plan_path=plan_path,
        repository_commit="f" * 40,
    )

    assert first == second
    assert first.source_binding_sha256


def test_training_source_binding_rejects_repository_commit_drift(tmp_path: Path) -> None:
    config_path, plan_path, source_path, _config = _fixture(tmp_path)
    source = build_training_source_binding(
        config_path=config_path,
        plan_path=plan_path,
        repository_commit="f" * 40,
    )

    with pytest.raises(ValueError, match="differs from current repo/config/plan"):
        verify_training_source_binding(
            source,
            config_path=config_path,
            plan_path=plan_path,
            repository_commit="0" * 40,
        )

    assert source_path.is_file()


def _write_run_manifest(tmp_path: Path, config: CyberSFTConfig) -> None:
    _write_json(
        tmp_path / "build" / "run" / "adapter-manifest.json",
        {
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
            "adapter_files": [],
            "trainable_target_module_count": 1,
            "trainable_target_modules_sha256": "e" * 64,
            "training_examples": 9,
            "validation_examples": 1,
            "gradient_checkpointing": True,
            "optimizer": "paged_adamw_8bit",
            "output_dir": config.output_dir,
        },
    )


def _mark_run_valid(monkeypatch) -> None:
    monkeypatch.setattr(
        reuse_module,
        "verify_cyber_sft_run",
        lambda *_args, **_kwargs: SimpleNamespace(valid=True, violations=[]),
    )


def test_completed_run_is_reusable_only_under_same_source_identity(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path, plan_path, source_path, config = _fixture(tmp_path)
    _write_run_manifest(tmp_path, config)
    _mark_run_valid(monkeypatch)

    report = evaluate_completed_run_reuse(
        run_dir="build/run",
        source_binding_path=source_path,
        config_path=config_path,
        plan_path=plan_path,
        repository_commit="f" * 40,
        root=tmp_path,
    )

    assert report.reusable is True
    assert report.run_verification_valid is True
    assert report.source_binding_valid is True
    assert report.run_source_identity_valid is True
    assert report.violations == []


def test_completed_run_reuse_rejects_new_repository_commit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path, plan_path, source_path, config = _fixture(tmp_path)
    _write_run_manifest(tmp_path, config)
    _mark_run_valid(monkeypatch)

    report = evaluate_completed_run_reuse(
        run_dir="build/run",
        source_binding_path=source_path,
        config_path=config_path,
        plan_path=plan_path,
        repository_commit="0" * 40,
        root=tmp_path,
    )

    assert report.reusable is False
    assert report.source_binding_valid is False
    assert any("training source binding is invalid" in row for row in report.violations)
