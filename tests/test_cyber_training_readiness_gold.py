from pathlib import Path

from koschei_sentinel.cyber_sft_training import CyberSFTConfig, load_cyber_sft_config
from koschei_sentinel.cyber_training_readiness import (
    CyberTrainingUseClass,
    audit_cyber_training_readiness,
)
from koschei_sentinel.defense_reflex_gold_release import write_gold_defense_release
from tests.test_defense_reflex_gold_release import _release_rows


def _gold_release(tmp_path: Path) -> Path:
    _policy, rows = _release_rows()
    output = tmp_path / "build" / "gold-defense-release"
    write_gold_defense_release(rows, output)
    return output


def _gold_config() -> CyberSFTConfig:
    return CyberSFTConfig(
        run_id="gold-readiness-test",
        stage="DEFENSE_REFLEX",
        base_model="Qwen/Qwen3.5-9B-Base",
        base_revision="a" * 40,
        corpus_dir="build/gold-defense-release/train",
        validation_corpus_dir="build/gold-defense-release/validation",
        validation_ratio=0.0,
        output_dir="build/cyber-training/runs/gold-readiness-test",
        minimum_cuda_memory_gb=14.0,
    )


def test_promotion_eligible_readiness_requires_and_accepts_audited_gold_release(
    tmp_path,
) -> None:
    release = _gold_release(tmp_path)

    report = audit_cyber_training_readiness(_gold_config(), root=tmp_path)

    assert report.static_plan_ready is True
    assert report.use_class is CyberTrainingUseClass.PROMOTION_ELIGIBLE
    assert report.gold_release_audit_checked is True
    assert report.gold_release_audit_valid is True
    assert report.gold_release_audit_sha256
    assert report.gold_release_dir == str(release.resolve())
    assert report.ready_to_execute is False
    assert not any("Gold Defense release audit failed" in row for row in report.blockers)


def test_invalid_gold_release_blocks_before_tokenization(tmp_path, monkeypatch) -> None:
    release = _gold_release(tmp_path)
    (release / "holdout" / "examples.jsonl").write_text("{}\n", encoding="utf-8")

    import koschei_sentinel.cyber_training_readiness as readiness

    original_find_spec = readiness.importlib.util.find_spec

    def fake_find_spec(name):
        if name == "transformers":
            return object()
        return original_find_spec(name)

    called = False

    def forbidden_tokenization(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("tokenization must not run after a failed Gold release audit")

    monkeypatch.setattr(readiness.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(readiness, "_tokenization_preflight", forbidden_tokenization)

    report = audit_cyber_training_readiness(
        _gold_config(),
        root=tmp_path,
        check_tokenization=True,
    )

    assert called is False
    assert report.gold_release_audit_checked is True
    assert report.gold_release_audit_valid is False
    assert report.tokenization_ready is False
    assert report.ready_to_execute is False
    assert any("Gold Defense release audit failed" in row for row in report.blockers)
    assert any("tokenization preflight skipped" in row for row in report.blockers)


def test_gold_9b_example_config_uses_explicit_validation_release() -> None:
    config = load_cyber_sft_config(
        "configs/training/cyber-sft.qwen3.5-9b.gold.example.json"
    )

    assert config.corpus_dir == "build/gold-defense-release/train"
    assert config.validation_corpus_dir == "build/gold-defense-release/validation"
    assert config.validation_ratio == 0.0
    assert config.base_model == "Qwen/Qwen3.5-9B-Base"
