from koschei_sentinel.cyber_seed_curriculum import write_seed_curriculum
from koschei_sentinel.cyber_sft_training import (
    CyberExecutionProfile,
    CyberSFTConfig,
)
from koschei_sentinel.cyber_training_readiness import (
    CyberTrainingUseClass,
    audit_cyber_training_readiness,
)


def _dense_config() -> CyberSFTConfig:
    return CyberSFTConfig(
        run_id="readiness-dense-test",
        stage="DEFENSE_REFLEX",
        execution_profile=CyberExecutionProfile.DENSE_SINGLE_GPU_QLORA,
        base_model="Qwen/Qwen3.5-9B-Base",
        base_revision="a" * 40,
        corpus_dir="build/cyber-training/defense-reflex-v3",
        output_dir="build/cyber-training/runs/readiness-dense-test",
        minimum_cuda_memory_gb=14.0,
    )


def test_static_readiness_does_not_claim_gpu_execution_ready(tmp_path) -> None:
    write_seed_curriculum(tmp_path / "build" / "cyber-training")
    report = audit_cyber_training_readiness(
        _dense_config(),
        root=tmp_path,
        check_runtime=False,
    )

    assert report.static_plan_ready is True
    assert report.runtime_checked is False
    assert report.tokenization_checked is False
    assert report.ready_to_execute is False
    assert report.use_class is CyberTrainingUseClass.SMOKE_ONLY
    assert report.plan is not None
    assert report.plan.example_count == 32


def test_tokenization_preflight_can_pass_without_claiming_cuda_ready(tmp_path, monkeypatch) -> None:
    write_seed_curriculum(tmp_path / "build" / "cyber-training")

    import koschei_sentinel.cyber_training_readiness as readiness

    original_find_spec = readiness.importlib.util.find_spec

    def fake_find_spec(name):
        if name == "transformers":
            return object()
        return original_find_spec(name)

    monkeypatch.setattr(readiness.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(
        readiness,
        "_tokenization_preflight",
        lambda config, root: (True, 1337, []),
    )

    report = audit_cyber_training_readiness(
        _dense_config(),
        root=tmp_path,
        check_runtime=False,
        check_tokenization=True,
    )

    assert report.tokenization_checked is True
    assert report.tokenization_ready is True
    assert report.max_observed_sequence_tokens == 1337
    assert report.overlength_example_ids == []
    assert report.ready_to_execute is False


def test_tokenization_preflight_blocks_overlength_examples(tmp_path, monkeypatch) -> None:
    write_seed_curriculum(tmp_path / "build" / "cyber-training")

    import koschei_sentinel.cyber_training_readiness as readiness

    original_find_spec = readiness.importlib.util.find_spec

    def fake_find_spec(name):
        if name == "transformers":
            return object()
        return original_find_spec(name)

    monkeypatch.setattr(readiness.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(
        readiness,
        "_tokenization_preflight",
        lambda config, root: (False, 2300, ["defense-reflex-v3:test:oversize"]),
    )

    report = audit_cyber_training_readiness(
        _dense_config(),
        root=tmp_path,
        check_runtime=False,
        check_tokenization=True,
    )

    assert report.tokenization_ready is False
    assert report.max_observed_sequence_tokens == 2300
    assert report.overlength_example_ids == ["defense-reflex-v3:test:oversize"]
    assert any("exceed max_sequence_length" in blocker for blocker in report.blockers)


def test_moe_plan_is_statically_blocked_by_current_executor(tmp_path) -> None:
    write_seed_curriculum(tmp_path / "build" / "cyber-training")
    config = CyberSFTConfig(
        run_id="readiness-moe-test",
        stage="DEFENSE_REFLEX",
        execution_profile=CyberExecutionProfile.MOE_DISTRIBUTED_REQUIRED,
        base_model="Qwen/Qwen3.5-35B-A3B-Base",
        base_revision="b" * 40,
        corpus_dir="build/cyber-training/defense-reflex-v3",
        output_dir="build/cyber-training/runs/readiness-moe-test",
        gradient_checkpointing=False,
        enable_router_aux_loss=True,
    )
    report = audit_cyber_training_readiness(config, root=tmp_path)

    assert report.static_plan_ready is True
    assert report.ready_to_execute is False
    assert any("not supported" in blocker for blocker in report.blockers)
