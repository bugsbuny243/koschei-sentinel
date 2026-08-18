from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import koschei_sentinel.cyber_sft_run_reuse as reuse_module
from koschei_sentinel.cyber_sft_run_reuse import evaluate_completed_run_reuse
from koschei_sentinel.cyber_sft_training import CyberSFTConfig
from koschei_sentinel.cyber_sft_training_source import (
    CyberSFTTrainingSourceBinding,
    build_training_source_binding,
    verify_training_source_binding,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _plan_payload(
    config: CyberSFTConfig,
    *,
    explicit_validation: bool = False,
    validation_examples_sha: str | None = None,
    validation_manifest_sha: str | None = None,
) -> dict[str, object]:
    return {
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
        "validation_corpus_examples_sha256": validation_examples_sha,
        "validation_corpus_manifest_sha256": validation_manifest_sha,
        "explicit_validation": explicit_validation,
        "example_count": 10,
        "training_examples": 9 if not explicit_validation else 8,
        "validation_examples": 1 if not explicit_validation else 2,
        "effective_batch_size": config.effective_batch_size,
        "estimated_optimizer_steps": 1,
        "input_adapter_dir": None,
        "output_dir": config.output_dir,
        "warnings": [],
    }


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
    _write_json(plan_path, _plan_payload(config))
    source_path = tmp_path / "training-source.json"
    source = build_training_source_binding(
        config_path=config_path,
        plan_path=plan_path,
        repository_commit="f" * 40,
    )
    _write_json(source_path, source.model_dump(mode="json"))
    return config_path, plan_path, source_path, config


def _explicit_fixture(tmp_path: Path) -> tuple[Path, Path, Path, CyberSFTConfig]:
    config = CyberSFTConfig(
        run_id="source-binding-gold-test",
        stage="DEFENSE_REFLEX",
        base_model="Qwen/Qwen3.5-9B-Base",
        base_revision="a" * 40,
        corpus_dir="build/gold/train",
        validation_corpus_dir="build/gold/validation",
        validation_ratio=0.0,
        output_dir="build/gold/run",
        minimum_cuda_memory_gb=0.0,
    )
    config_path = tmp_path / "gold-config.json"
    _write_json(config_path, config.model_dump(mode="json"))
    plan_path = tmp_path / "gold-plan.json"
    _write_json(
        plan_path,
        _plan_payload(
            config,
            explicit_validation=True,
            validation_examples_sha="d" * 64,
            validation_manifest_sha="e" * 64,
        ),
    )
    source_path = tmp_path / "gold-training-source.json"
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
    assert first.explicit_validation is False
    assert first.validation_corpus_examples_sha256 is None
    assert first.validation_corpus_manifest_sha256 is None


def test_explicit_validation_source_binds_validation_corpus_hashes(tmp_path: Path) -> None:
    config_path, plan_path, _source_path, _config = _explicit_fixture(tmp_path)

    source = build_training_source_binding(
        config_path=config_path,
        plan_path=plan_path,
        repository_commit="f" * 40,
    )

    assert source.explicit_validation is True
    assert source.validation_corpus_examples_sha256 == "d" * 64
    assert source.validation_corpus_manifest_sha256 == "e" * 64


def test_explicit_validation_source_rejects_validation_plan_drift(tmp_path: Path) -> None:
    config_path, plan_path, source_path, _config = _explicit_fixture(tmp_path)
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    source = build_training_source_binding(
        config_path=config_path,
        plan_path=plan_path,
        repository_commit="f" * 40,
    )
    assert source_payload["source_binding_sha256"] == source.source_binding_sha256

    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_payload["validation_corpus_examples_sha256"] = "9" * 64
    _write_json(plan_path, plan_payload)

    with pytest.raises(ValueError, match="differs from current repo/config/plan"):
        verify_training_source_binding(
            source,
            config_path=config_path,
            plan_path=plan_path,
            repository_commit="f" * 40,
        )


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


def _expected_resume_binding(source: CyberSFTTrainingSourceBinding) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "sentinel.cyber-sft-resume-binding.v1",
        "run_id": source.run_id,
        "base_model": source.base_model,
        "base_revision": source.base_revision,
        "corpus_examples_sha256": source.corpus_examples_sha256,
        "corpus_manifest_sha256": source.corpus_manifest_sha256,
        "config_sha256": source.config_sha256,
    }
    if source.explicit_validation:
        payload.update(
            {
                "explicit_validation": True,
                "validation_corpus_examples_sha256": source.validation_corpus_examples_sha256,
                "validation_corpus_manifest_sha256": source.validation_corpus_manifest_sha256,
            }
        )
    return payload


def _write_run_manifest(
    tmp_path: Path,
    config: CyberSFTConfig,
    source: CyberSFTTrainingSourceBinding,
) -> None:
    run = tmp_path / config.output_dir
    _write_json(
        run / "adapter-manifest.json",
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
    _write_json(
        run / "resume-runtime.json",
        {
            "schema_version": "sentinel.cyber-sft-resume-runtime.v1",
            "resumed": False,
            "resume_checkpoint": None,
            "checkpoint_every_optimizer_steps": 2,
            "checkpoint_retention": 2,
            "resume_binding": _expected_resume_binding(source),
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
    source = CyberSFTTrainingSourceBinding.model_validate_json(source_path.read_bytes())
    _write_run_manifest(tmp_path, config, source)
    _mark_run_valid(monkeypatch)

    report = evaluate_completed_run_reuse(
        run_dir=config.output_dir,
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
    source = CyberSFTTrainingSourceBinding.model_validate_json(source_path.read_bytes())
    _write_run_manifest(tmp_path, config, source)
    _mark_run_valid(monkeypatch)

    report = evaluate_completed_run_reuse(
        run_dir=config.output_dir,
        source_binding_path=source_path,
        config_path=config_path,
        plan_path=plan_path,
        repository_commit="0" * 40,
        root=tmp_path,
    )

    assert report.reusable is False
    assert report.source_binding_valid is False
    assert any("training source binding is invalid" in row for row in report.violations)


def test_completed_gold_run_reuse_rejects_validation_identity_drift(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path, plan_path, source_path, config = _explicit_fixture(tmp_path)
    source = CyberSFTTrainingSourceBinding.model_validate_json(source_path.read_bytes())
    _write_run_manifest(tmp_path, config, source)
    _mark_run_valid(monkeypatch)

    run = tmp_path / config.output_dir
    resume_path = run / "resume-runtime.json"
    resume = json.loads(resume_path.read_text(encoding="utf-8"))
    resume["resume_binding"]["validation_corpus_examples_sha256"] = "9" * 64
    _write_json(resume_path, resume)

    report = evaluate_completed_run_reuse(
        run_dir=config.output_dir,
        source_binding_path=source_path,
        config_path=config_path,
        plan_path=plan_path,
        repository_commit="f" * 40,
        root=tmp_path,
    )

    assert report.reusable is False
    assert report.source_binding_valid is True
    assert report.run_source_identity_valid is False
    assert any("resume binding differs" in row for row in report.violations)
