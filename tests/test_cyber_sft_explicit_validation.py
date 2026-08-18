import json
from pathlib import Path

import pytest

from koschei_sentinel.cyber_seed_curriculum import build_seed_curriculum
from koschei_sentinel.cyber_sft_text_trainer import (
    _assigned_training_rows,
    _resume_binding,
)
from koschei_sentinel.cyber_sft_training import (
    CyberSFTConfig,
    load_cyber_sft_examples,
    plan_cyber_sft,
)
from koschei_sentinel.defense_reflex_corpus_v3 import (
    DefenseLessonKind,
    DefenseReviewMethod,
    build_defense_reflex_v3_example,
    build_defense_reflex_v3_manifest,
    create_reviewed_defense_lesson,
    serialize_defense_reflex_v3,
)


def _human_example(seed_index: int, reviewer_suffix: str = "a"):
    _family, scenario, seed_lesson = build_seed_curriculum()[seed_index]
    lesson = create_reviewed_defense_lesson(
        lesson_kind=DefenseLessonKind.GOLD,
        scenario=scenario,
        reviewer_id=f"human-reviewer:explicit-{reviewer_suffix}",
        review_method=DefenseReviewMethod.HUMAN,
        interpretation=seed_lesson.interpretation,
        expected_steps=seed_lesson.expected_steps,
        review_evidence_ids=[f"human-review:explicit:{seed_index}:{reviewer_suffix}"],
        training_authorization=True,
        promotion_eligible=True,
    )
    return build_defense_reflex_v3_example(scenario, lesson)


def _write_corpus(root: Path, relative: str, examples) -> None:
    destination = root / relative
    destination.mkdir(parents=True, exist_ok=True)
    payload = serialize_defense_reflex_v3(examples)
    manifest = build_defense_reflex_v3_manifest(examples)
    assert manifest.promotion_eligible is True
    (destination / "examples.jsonl").write_text(payload, encoding="utf-8")
    (destination / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _config() -> CyberSFTConfig:
    return CyberSFTConfig(
        run_id="gold-explicit-validation-test",
        stage="DEFENSE_REFLEX",
        base_model="Qwen/Qwen3.5-9B-Base",
        base_revision="a" * 40,
        corpus_dir="build/gold/train",
        validation_corpus_dir="build/gold/validation",
        validation_ratio=0.0,
        output_dir="build/gold/run",
        minimum_cuda_memory_gb=0.0,
        quantization={"bits": 4, "compute_dtype": "float16"},
    )


def _write_distinct_splits(tmp_path: Path):
    train_example = _human_example(0, "train")
    validation_example = _human_example(1, "validation")
    _write_corpus(tmp_path, "build/gold/train", [train_example])
    _write_corpus(tmp_path, "build/gold/validation", [validation_example])
    return train_example, validation_example


def test_explicit_validation_plan_preserves_preassigned_splits(tmp_path: Path) -> None:
    train_example, validation_example = _write_distinct_splits(tmp_path)
    config = _config()

    plan = plan_cyber_sft(config, root=tmp_path)

    assert plan.explicit_validation is True
    assert plan.training_examples == 1
    assert plan.validation_examples == 1
    assert plan.example_count == 2
    assert plan.corpus_promotion_eligible is True
    assert plan.validation_corpus_examples_sha256
    assert plan.validation_corpus_manifest_sha256

    rows, _examples_sha, _manifest_sha, promotion = load_cyber_sft_examples(
        config,
        root=tmp_path,
    )
    training_rows, validation_rows = _assigned_training_rows(
        config,
        plan,
        rows,
        promotion,
        root=tmp_path,
    )
    assert [row.example_id for row in training_rows] == [train_example.example_id]
    assert [row.example_id for row in validation_rows] == [validation_example.example_id]


def test_explicit_validation_requires_zero_validation_ratio() -> None:
    with pytest.raises(ValueError, match="validation_ratio=0.0"):
        CyberSFTConfig(
            run_id="gold-invalid-resplit-test",
            stage="DEFENSE_REFLEX",
            base_model="Qwen/Qwen3.5-9B-Base",
            base_revision="a" * 40,
            corpus_dir="build/gold/train",
            validation_corpus_dir="build/gold/validation",
            validation_ratio=0.1,
            output_dir="build/gold/run",
            minimum_cuda_memory_gb=0.0,
        )


def test_explicit_validation_rejects_scenario_overlap(tmp_path: Path) -> None:
    train_example = _human_example(0, "train")
    validation_example = _human_example(0, "validation")
    assert train_example.example_id != validation_example.example_id
    assert train_example.scenario_id == validation_example.scenario_id
    _write_corpus(tmp_path, "build/gold/train", [train_example])
    _write_corpus(tmp_path, "build/gold/validation", [validation_example])

    with pytest.raises(ValueError, match="scenario IDs overlap"):
        plan_cyber_sft(_config(), root=tmp_path)


def test_validation_corpus_drift_after_planning_is_rejected(tmp_path: Path) -> None:
    _write_distinct_splits(tmp_path)
    config = _config()
    plan = plan_cyber_sft(config, root=tmp_path)
    rows, _examples_sha, _manifest_sha, promotion = load_cyber_sft_examples(
        config,
        root=tmp_path,
    )

    replacement = _human_example(2, "replacement")
    validation_dir = tmp_path / "build/gold/validation"
    for child in validation_dir.iterdir():
        child.unlink()
    _write_corpus(tmp_path, "build/gold/validation", [replacement])

    with pytest.raises(ValueError, match="validation examples changed after plan creation"):
        _assigned_training_rows(
            config,
            plan,
            rows,
            promotion,
            root=tmp_path,
        )


def test_explicit_validation_resume_binding_contains_validation_hashes(tmp_path: Path) -> None:
    _write_distinct_splits(tmp_path)
    config = _config()
    plan = plan_cyber_sft(config, root=tmp_path)

    binding = _resume_binding(config, plan)

    assert binding["explicit_validation"] is True
    assert (
        binding["validation_corpus_examples_sha256"]
        == plan.validation_corpus_examples_sha256
    )
    assert (
        binding["validation_corpus_manifest_sha256"]
        == plan.validation_corpus_manifest_sha256
    )
