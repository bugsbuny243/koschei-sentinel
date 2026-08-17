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
    assert report.ready_to_execute is False
    assert report.use_class is CyberTrainingUseClass.SMOKE_ONLY
    assert report.plan is not None
    assert report.plan.example_count == 32


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
