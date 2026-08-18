from types import SimpleNamespace

import pytest

import koschei_sentinel.cyber_sft_cli as cli
from koschei_sentinel.cyber_sft_training import CyberSFTConfig


def _config() -> CyberSFTConfig:
    return CyberSFTConfig(
        run_id="gold-cli-gate-test",
        stage="DEFENSE_REFLEX",
        base_model="Qwen/Qwen3.5-9B-Base",
        base_revision="a" * 40,
        corpus_dir="build/gold/train",
        validation_corpus_dir="build/gold/validation",
        validation_ratio=0.0,
        output_dir="build/run",
        minimum_cuda_memory_gb=0.0,
    )


def test_promotion_eligible_execution_rejects_invalid_gold_audit(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "audit_cyber_training_readiness",
        lambda *_args, **_kwargs: SimpleNamespace(
            gold_release_audit_checked=True,
            gold_release_audit_valid=False,
            blockers=["Gold Defense release audit failed: HOLDOUT tampered"],
        ),
    )

    with pytest.raises(RuntimeError, match="requires a valid Gold release audit"):
        cli._assert_gold_execution_gate(
            _config(),
            SimpleNamespace(corpus_promotion_eligible=True),
        )


def test_smoke_execution_does_not_invoke_gold_audit(monkeypatch) -> None:
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("smoke-only execution must not invoke Gold audit")

    monkeypatch.setattr(cli, "audit_cyber_training_readiness", forbidden)

    cli._assert_gold_execution_gate(
        _config(),
        SimpleNamespace(corpus_promotion_eligible=False),
    )

    assert called is False
