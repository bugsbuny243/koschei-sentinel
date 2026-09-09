from __future__ import annotations

import koschei_sentinel.cyber_sft_cli as cli_module
from koschei_sentinel.cyber_sft_training import CyberSFTConfig, CyberSFTPlan


def _config() -> CyberSFTConfig:
    return CyberSFTConfig(
        run_id="execution-cli-preflight-test",
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


def _plan() -> CyberSFTPlan:
    config = _config()
    return CyberSFTPlan(
        run_id=config.run_id,
        stage=config.stage,
        execution_profile=config.execution_profile,
        executable_with_current_trainer=True,
        corpus_promotion_eligible=False,
        base_model=config.base_model,
        base_revision=config.base_revision,
        corpus_examples_sha256="1" * 64,
        corpus_manifest_sha256="2" * 64,
        validation_corpus_examples_sha256="3" * 64,
        validation_corpus_manifest_sha256="4" * 64,
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


def test_execute_preflights_sources_before_gpu_executor(monkeypatch, capsys) -> None:
    config = _config()
    plan = _plan()
    calls: list[str] = []

    monkeypatch.setattr(cli_module, "load_cyber_sft_config", lambda *_args: config)
    monkeypatch.setattr(cli_module, "plan_cyber_sft", lambda *_args: plan)
    monkeypatch.setattr(
        cli_module,
        "resolve_planned_cyber_sft_corpora",
        lambda *_args, **_kwargs: calls.append("source-preflight"),
    )
    monkeypatch.setattr(
        cli_module,
        "_assert_gold_execution_gate",
        lambda *_args, **_kwargs: calls.append("gold-gate"),
    )

    def execute(*_args, **_kwargs):
        calls.append("execute")
        return plan

    monkeypatch.setattr(cli_module, "execute_cyber_sft_text", execute)

    status = cli_module.main(["--config", "fixture.json", "--execute"])

    assert status == 0
    assert calls == ["source-preflight", "gold-gate", "execute"]
    assert '"run_id": "execution-cli-preflight-test"' in capsys.readouterr().out


def test_source_preflight_failure_blocks_gpu_executor(monkeypatch, capsys) -> None:
    config = _config()
    plan = _plan()
    executed = False

    monkeypatch.setattr(cli_module, "load_cyber_sft_config", lambda *_args: config)
    monkeypatch.setattr(cli_module, "plan_cyber_sft", lambda *_args: plan)

    def fail_preflight(*_args, **_kwargs):
        raise ValueError("VALIDATION corpus changed after plan creation")

    monkeypatch.setattr(cli_module, "resolve_planned_cyber_sft_corpora", fail_preflight)
    monkeypatch.setattr(cli_module, "_assert_gold_execution_gate", lambda *_args: None)

    def forbidden_execute(*_args, **_kwargs):
        nonlocal executed
        executed = True
        raise AssertionError("GPU executor must not run after source preflight failure")

    monkeypatch.setattr(cli_module, "execute_cyber_sft_text", forbidden_execute)

    status = cli_module.main(["--config", "fixture.json", "--execute"])

    assert status == 2
    assert executed is False
    assert "VALIDATION corpus changed after plan creation" in capsys.readouterr().out
