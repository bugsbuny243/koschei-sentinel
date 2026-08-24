import hashlib
import json

import pytest

from koschei_sentinel.cyber_megatron_candidate import (
    build_cyber_megatron_candidate,
    verify_cyber_megatron_candidate,
)
from koschei_sentinel.cyber_megatron_training import (
    QWEN35_397B_MODEL,
    QWEN35_397B_REVISION,
    CyberMegatronDatasetManifest,
    CyberMegatronDatasetVerification,
    CyberMegatronLaunchState,
    CyberMegatronPlan,
    CyberMegatronSFTConfig,
    _plan_sha256,
    _run_identity,
)


def _write_json(path, model) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _config() -> CyberMegatronSFTConfig:
    return CyberMegatronSFTConfig.model_validate(
        {
            "run_id": "qwen3p5-397b-gold-candidate-test",
            "model": QWEN35_397B_MODEL,
            "model_revision": QWEN35_397B_REVISION,
            "backend": "megatron-swift",
            "ms_swift_version": "4.5.2",
            "stage": "DEFENSE_REFLEX",
            "corpus_dir": "build/gold-defense-release/train",
            "validation_corpus_dir": "build/gold-defense-release/validation",
            "validation_ratio": 0.0,
            "dataset_dir": "build/megatron/397b-gold-dataset",
            "output_dir": "build/megatron/397b-run",
            "topology": {
                "nodes": 4,
                "gpus_per_node": 8,
                "minimum_gpu_memory_gib": 80,
                "tensor_model_parallel_size": 4,
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": 1,
                "expert_model_parallel_size": 8,
                "sequence_parallel": True,
            },
        }
    )


def _fixture(tmp_path, *, gold_audit=True, completed=True):
    config = _config()
    config_path = tmp_path / "configs" / "397b.json"
    _write_json(config_path, config)

    dataset = CyberMegatronDatasetManifest(
        run_id=config.run_id,
        model=config.model,
        model_revision=config.model_revision,
        stage=config.stage,
        seed=config.seed,
        source_examples_sha256="1" * 64,
        source_manifest_sha256="2" * 64,
        validation_source_examples_sha256="3" * 64,
        validation_source_manifest_sha256="4" * 64,
        source_promotion_eligible=True,
        validation_source_promotion_eligible=True,
        gold_release_audit_sha256="5" * 64 if gold_audit else None,
        training_examples=100,
        validation_examples=20,
        max_observed_sequence_tokens=4096,
        train_jsonl_sha256="6" * 64,
        validation_jsonl_sha256="7" * 64,
    )
    dataset_path = tmp_path / config.dataset_dir / "manifest.json"
    _write_json(dataset_path, dataset)
    dataset_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    verification = CyberMegatronDatasetVerification(
        valid=True,
        manifest=dataset,
        manifest_sha256=dataset_sha,
        blockers=[],
    )
    identity = _run_identity(config, verification)

    output_root = tmp_path / config.output_dir
    output_root.mkdir(parents=True)
    _write_json(output_root / "koschei-run-identity.json", identity)
    launch_state = CyberMegatronLaunchState(
        run_identity_sha256=identity.identity_sha256,
        launch_session_sha256="8" * 64,
        rendezvous_sha256="9" * 64,
        state="completed" if completed else "failed",
        nodes=config.topology.nodes,
    )
    _write_json(output_root / "koschei-launch-state.json", launch_state)

    plan = CyberMegatronPlan(
        run_id=config.run_id,
        model=config.model,
        model_revision=config.model_revision,
        backend="megatron-swift",
        ms_swift_version="4.5.2",
        world_size=config.topology.world_size,
        minimum_gpu_memory_gib=config.topology.minimum_gpu_memory_gib,
        data_parallel_size=config.topology.data_parallel_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataset_verified=True,
        config_sha256=identity.config_sha256,
        dataset_manifest_sha256=dataset_sha,
        source_examples_sha256=dataset.source_examples_sha256,
        source_manifest_sha256=dataset.source_manifest_sha256,
        validation_source_examples_sha256=dataset.validation_source_examples_sha256,
        validation_source_manifest_sha256=dataset.validation_source_manifest_sha256,
        train_jsonl_sha256=dataset.train_jsonl_sha256,
        validation_jsonl_sha256=dataset.validation_jsonl_sha256,
        gold_release_audit_sha256=dataset.gold_release_audit_sha256,
        max_observed_sequence_tokens=dataset.max_observed_sequence_tokens,
        run_identity_sha256=identity.identity_sha256,
        output_available=True,
        static_ready=True,
        launch_approval_env="KOSCHEI_397B_LAUNCH_APPROVED",
        launch_session_env="KOSCHEI_397B_LAUNCH_SESSION",
        command=["megatron", "sft", "--model"],
        blockers=[],
        warnings=[],
    )
    plan_path = tmp_path / "build" / "megatron" / "397b-plan.json"
    _write_json(plan_path, plan)

    checkpoint = output_root / "checkpoint-100"
    (checkpoint / "model").mkdir(parents=True)
    (checkpoint / "adapter").mkdir(parents=True)
    (checkpoint / "model" / "weights.safetensors").write_bytes(b"mcore-weights")
    (checkpoint / "adapter" / "adapter_model.safetensors").write_bytes(b"lora-weights")
    (checkpoint / "metadata.json").write_text('{"iteration":100}\n', encoding="utf-8")

    return config, config_path, dataset, identity, plan, plan_path, checkpoint


def _relative(tmp_path, path) -> str:
    return path.relative_to(tmp_path).as_posix()


def test_candidate_snapshot_binds_gold_run_plan_and_exact_checkpoint(tmp_path) -> None:
    config, config_path, dataset, identity, plan, plan_path, checkpoint = _fixture(tmp_path)
    manifest_path = tmp_path / "build" / "candidates" / "397b.json"

    manifest = build_cyber_megatron_candidate(
        root=tmp_path,
        config_path=_relative(tmp_path, config_path),
        plan_path=_relative(tmp_path, plan_path),
        checkpoint_dir=_relative(tmp_path, checkpoint),
        output_path=_relative(tmp_path, manifest_path),
    )

    assert manifest.model == QWEN35_397B_MODEL
    assert manifest.model_revision == QWEN35_397B_REVISION
    assert manifest.run_id == config.run_id
    assert manifest.run_identity_sha256 == identity.identity_sha256
    assert manifest.gold_release_audit_sha256 == dataset.gold_release_audit_sha256
    assert manifest.plan_contract_sha256 == _plan_sha256(plan)
    assert manifest.checkpoint_relative_path == "checkpoint-100"
    assert manifest.checkpoint_file_count == 3
    assert [row.path for row in manifest.checkpoint_files] == [
        "adapter/adapter_model.safetensors",
        "metadata.json",
        "model/weights.safetensors",
    ]
    assert manifest.candidate_sha256

    verification = verify_cyber_megatron_candidate(
        root=tmp_path,
        manifest_path=_relative(tmp_path, manifest_path),
        config_path=_relative(tmp_path, config_path),
        plan_path=_relative(tmp_path, plan_path),
        checkpoint_dir=_relative(tmp_path, checkpoint),
    )
    assert verification.valid is True
    assert verification.manifest == manifest
    assert verification.violations == []


def test_candidate_verification_fails_after_single_checkpoint_byte_changes(tmp_path) -> None:
    _config_obj, config_path, _dataset, _identity, _plan, plan_path, checkpoint = _fixture(
        tmp_path
    )
    manifest_path = tmp_path / "build" / "candidates" / "397b.json"
    build_cyber_megatron_candidate(
        root=tmp_path,
        config_path=_relative(tmp_path, config_path),
        plan_path=_relative(tmp_path, plan_path),
        checkpoint_dir=_relative(tmp_path, checkpoint),
        output_path=_relative(tmp_path, manifest_path),
    )

    (checkpoint / "model" / "weights.safetensors").write_bytes(b"tampered")
    verification = verify_cyber_megatron_candidate(
        root=tmp_path,
        manifest_path=_relative(tmp_path, manifest_path),
        config_path=_relative(tmp_path, config_path),
        plan_path=_relative(tmp_path, plan_path),
        checkpoint_dir=_relative(tmp_path, checkpoint),
    )

    assert verification.valid is False
    assert any("freshly revalidated" in row for row in verification.violations)


def test_candidate_snapshot_rejects_symlink_inside_checkpoint(tmp_path) -> None:
    _config_obj, config_path, _dataset, _identity, _plan, plan_path, checkpoint = _fixture(
        tmp_path
    )
    target = checkpoint / "model" / "weights.safetensors"
    (checkpoint / "model" / "alias.safetensors").symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        build_cyber_megatron_candidate(
            root=tmp_path,
            config_path=_relative(tmp_path, config_path),
            plan_path=_relative(tmp_path, plan_path),
            checkpoint_dir=_relative(tmp_path, checkpoint),
            output_path="build/candidates/397b.json",
        )


def test_candidate_snapshot_rejects_non_completed_training(tmp_path) -> None:
    _config_obj, config_path, _dataset, _identity, _plan, plan_path, checkpoint = _fixture(
        tmp_path,
        completed=False,
    )

    with pytest.raises(ValueError, match="before training completes"):
        build_cyber_megatron_candidate(
            root=tmp_path,
            config_path=_relative(tmp_path, config_path),
            plan_path=_relative(tmp_path, plan_path),
            checkpoint_dir=_relative(tmp_path, checkpoint),
            output_path="build/candidates/397b.json",
        )


def test_candidate_snapshot_rejects_training_without_gold_audit_binding(tmp_path) -> None:
    _config_obj, config_path, _dataset, _identity, _plan, plan_path, checkpoint = _fixture(
        tmp_path,
        gold_audit=False,
    )

    with pytest.raises(ValueError, match="Gold release audit"):
        build_cyber_megatron_candidate(
            root=tmp_path,
            config_path=_relative(tmp_path, config_path),
            plan_path=_relative(tmp_path, plan_path),
            checkpoint_dir=_relative(tmp_path, checkpoint),
            output_path="build/candidates/397b.json",
        )
