from __future__ import annotations

from pathlib import Path

import pytest

import koschei_sentinel.cyber_sft_trainer as trainer_module
from koschei_sentinel.cyber_sft_training import CyberSFTConfig, CyberSFTPlan


def _config() -> CyberSFTConfig:
    return CyberSFTConfig(
        run_id="explicit-validation-test",
        stage="DEFENSE_REFLEX",
        base_model="Qwen/Qwen3.5-9B-Base",
        base_revision="a" * 40,
        corpus_dir="build/train",
        validation_corpus_dir="build/validation",
        validation_ratio=0.0,
        output_dir="build/run",
        minimum_cuda_memory_gb=0.0,
        quantization={"bits": 4, "compute_dtype": "float16"},
    )


def _plan(*, validation_examples_sha: str = "c" * 64) -> CyberSFTPlan:
    config = _config()
    return CyberSFTPlan(
        run_id=config.run_id,
        stage=config.stage,
        execution_profile=config.execution_profile,
        executable_with_current_trainer=True,
        corpus_promotion_eligible=True,
        base_model=config.base_model,
        base_revision=config.base_revision,
        corpus_examples_sha256="1" * 64,
        corpus_manifest_sha256="2" * 64,
        validation_corpus_examples_sha256=validation_examples_sha,
        validation_corpus_manifest_sha256="d" * 64,
        explicit_validation=True,
        example_count=3,
        training_examples=2,
        validation_examples=1,
        effective_batch_size=config.effective_batch_size,
        estimated_optimizer_steps=1,
        input_adapter_dir=None,
        output_dir=config.output_dir,
        warnings=[],
    )


def test_executor_resolves_preassigned_validation_without_resplitting(monkeypatch) -> None:
    config = _config()
    plan = _plan()
    train_rows = [object(), object()]
    validation_rows = [object()]
    disjoint_calls: list[tuple[list[object], list[object]]] = []

    monkeypatch.setattr(
        trainer_module,
        "load_cyber_sft_examples",
        lambda *_args, **_kwargs: (train_rows, "1" * 64, "2" * 64, True),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_cyber_sft_validation_examples",
        lambda *_args, **_kwargs: (validation_rows, "c" * 64, "d" * 64, True),
    )
    monkeypatch.setattr(
        trainer_module,
        "_assert_explicit_split_disjoint",
        lambda train, validation: disjoint_calls.append((train, validation)),
    )

    def forbidden_split(*_args, **_kwargs):
        raise AssertionError("explicit validation must not be re-split")

    monkeypatch.setattr(trainer_module, "split_cyber_sft_examples", forbidden_split)

    resolved = trainer_module._resolve_planned_corpora(config, plan, root=Path("."))

    assert resolved[0] is train_rows
    assert resolved[1] is validation_rows
    assert resolved[2:] == ("1" * 64, "2" * 64, True)
    assert disjoint_calls == [(train_rows, validation_rows)]


def test_executor_rejects_explicit_validation_digest_drift(monkeypatch) -> None:
    config = _config()
    plan = _plan(validation_examples_sha="e" * 64)

    monkeypatch.setattr(
        trainer_module,
        "load_cyber_sft_examples",
        lambda *_args, **_kwargs: ([object(), object()], "1" * 64, "2" * 64, True),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_cyber_sft_validation_examples",
        lambda *_args, **_kwargs: ([object()], "c" * 64, "d" * 64, True),
    )
    monkeypatch.setattr(
        trainer_module,
        "_assert_explicit_split_disjoint",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ValueError, match="VALIDATION corpus changed"):
        trainer_module._resolve_planned_corpora(config, plan)


def test_executor_rejects_plan_count_drift(monkeypatch) -> None:
    config = _config()
    plan = _plan().model_copy(update={"validation_examples": 2, "example_count": 4})

    monkeypatch.setattr(
        trainer_module,
        "load_cyber_sft_examples",
        lambda *_args, **_kwargs: ([object(), object()], "1" * 64, "2" * 64, True),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_cyber_sft_validation_examples",
        lambda *_args, **_kwargs: ([object()], "c" * 64, "d" * 64, True),
    )
    monkeypatch.setattr(
        trainer_module,
        "_assert_explicit_split_disjoint",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ValueError, match="VALIDATION example count differs"):
        trainer_module._resolve_planned_corpora(config, plan)
