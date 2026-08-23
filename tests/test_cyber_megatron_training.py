import json

import pytest
from pydantic import ValidationError

from koschei_sentinel.cyber_megatron_training import (
    LAUNCH_APPROVAL_ENV,
    MS_SWIFT_VERSION,
    QWEN35_397B_MODEL,
    QWEN35_397B_REVISION,
    CyberMegatronResume,
    CyberMegatronSFTConfig,
    CyberMegatronTopology,
    build_cyber_megatron_command,
    execute_cyber_megatron_sft,
    load_cyber_megatron_config,
    materialize_cyber_megatron_dataset,
    plan_cyber_megatron_sft,
    verify_cyber_megatron_dataset,
)
from koschei_sentinel.cyber_seed_curriculum import write_seed_curriculum


def _payload() -> dict[str, object]:
    return {
        "run_id": "qwen3p5-397b-test",
        "model": QWEN35_397B_MODEL,
        "model_revision": QWEN35_397B_REVISION,
        "backend": "megatron-swift",
        "ms_swift_version": MS_SWIFT_VERSION,
        "stage": "DEFENSE_REFLEX",
        "corpus_dir": "build/cyber-training/defense-reflex-v3",
        "validation_ratio": 0.1,
        "dataset_dir": "build/cyber-training/megatron/test",
        "output_dir": "build/cyber-training/runs/test",
        "topology": {
            "nodes": 4,
            "gpus_per_node": 8,
            "minimum_gpu_memory_gib": 80,
            "tensor_model_parallel_size": 8,
            "pipeline_model_parallel_size": 4,
            "context_parallel_size": 1,
            "expert_model_parallel_size": 8,
            "sequence_parallel": True,
        },
    }


def _config() -> CyberMegatronSFTConfig:
    return CyberMegatronSFTConfig.model_validate(_payload())


def _value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


@pytest.mark.parametrize(
    "model",
    ["Qwen/Qwen3.5-35B-A3B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-0.8B"],
)
def test_single_model_contract_rejects_every_non_397b_target(model: str) -> None:
    payload = _payload()
    payload["model"] = model
    with pytest.raises(ValidationError, match="only Qwen/Qwen3.5-397B-A17B"):
        CyberMegatronSFTConfig.model_validate(payload)


def test_single_model_contract_rejects_unpinned_revision() -> None:
    payload = _payload()
    payload["model_revision"] = "a" * 40
    with pytest.raises(ValidationError, match="pinned weight revision"):
        CyberMegatronSFTConfig.model_validate(payload)


def test_router_loss_and_frozen_vision_are_fail_closed() -> None:
    payload = _payload()
    payload["optimization"] = {"moe_aux_loss_coeff": 0.0}
    with pytest.raises(ValidationError):
        CyberMegatronSFTConfig.model_validate(payload)

    payload = _payload()
    payload["lora"] = {"freeze_vit": False}
    with pytest.raises(ValidationError):
        CyberMegatronSFTConfig.model_validate(payload)


def test_topology_requires_a_valid_model_parallel_partition() -> None:
    with pytest.raises(ValidationError, match=r"TP \* PP \* CP"):
        CyberMegatronTopology(
            nodes=3,
            gpus_per_node=8,
            minimum_gpu_memory_gib=80,
            tensor_model_parallel_size=8,
            pipeline_model_parallel_size=4,
            expert_model_parallel_size=8,
        )


def test_megatron_command_is_pinned_and_keeps_recovery_state() -> None:
    command = build_cyber_megatron_command(_config())

    assert command[:2] == ["megatron", "sft"]
    assert _value(command, "--model") == QWEN35_397B_MODEL
    assert _value(command, "--model_revision") == QWEN35_397B_REVISION
    assert _value(command, "--tuner_type") == "lora"
    assert _value(command, "--language_model_only") == "true"
    assert _value(command, "--freeze_vit") == "true"
    assert _value(command, "--freeze_aligner") == "true"
    assert float(_value(command, "--moe_aux_loss_coeff")) > 0.0
    assert _value(command, "--tensor_model_parallel_size") == "8"
    assert _value(command, "--pipeline_model_parallel_size") == "4"
    assert _value(command, "--expert_model_parallel_size") == "8"
    assert _value(command, "--no_save_optim") == "false"
    assert _value(command, "--no_save_rng") == "false"
    assert _value(command, "--finetune") == "true"
    assert not any("35B-A3B" in token or "9B" in token for token in command)


def test_resume_command_restores_mcore_model_adapter_optimizer_and_rng() -> None:
    resume = CyberMegatronResume(
        mcore_model="/shared/checkpoints/model",
        mcore_adapter="/shared/checkpoints/adapter",
    )
    command = build_cyber_megatron_command(_config(), resume=resume)

    assert _value(command, "--mcore_model") == resume.mcore_model
    assert _value(command, "--mcore_adapter") == resume.mcore_adapter
    assert _value(command, "--no_load_optim") == "false"
    assert _value(command, "--no_load_rng") == "false"
    assert _value(command, "--finetune") == "false"


def test_dataset_materialization_and_plan_are_digest_bound(tmp_path) -> None:
    write_seed_curriculum(tmp_path / "build" / "cyber-training")
    config = _config()
    manifest = materialize_cyber_megatron_dataset(config, root=tmp_path)

    assert manifest.training_examples + manifest.validation_examples == 32
    assert manifest.validation_examples > 0
    verification = verify_cyber_megatron_dataset(config, root=tmp_path)
    assert verification.valid is True

    train_path = tmp_path / config.dataset_dir / "train.jsonl"
    first = json.loads(train_path.read_text(encoding="utf-8").splitlines()[0])
    assert [message["role"] for message in first["messages"]] == [
        "system",
        "user",
        "assistant",
    ]

    plan = plan_cyber_megatron_sft(config, root=tmp_path)
    assert plan.static_ready is True
    assert plan.dataset_verified is True
    assert plan.world_size == 32
    assert plan.data_parallel_size == 1
    assert plan.gradient_accumulation_steps == 32
    assert plan.requires_explicit_launch_approval is True

    train_path.write_text(train_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    tampered = verify_cyber_megatron_dataset(config, root=tmp_path)
    assert tampered.valid is False
    assert any("train.jsonl digest" in blocker for blocker in tampered.blockers)


def test_execute_requires_exact_paid_cluster_approval(tmp_path, monkeypatch) -> None:
    write_seed_curriculum(tmp_path / "build" / "cyber-training")
    config = _config()
    materialize_cyber_megatron_dataset(config, root=tmp_path)
    monkeypatch.delenv(LAUNCH_APPROVAL_ENV, raising=False)

    with pytest.raises(RuntimeError, match=LAUNCH_APPROVAL_ENV):
        execute_cyber_megatron_sft(config, root=tmp_path)


def test_checked_in_config_is_the_only_active_397b_target() -> None:
    config = load_cyber_megatron_config(
        "configs/training/cyber-sft.qwen3.5-397b-a17b.megatron.json"
    )
    gold = load_cyber_megatron_config(
        "configs/training/cyber-sft.qwen3.5-397b-a17b.gold.example.json"
    )
    plan = json.loads(
        open("configs/training/cyber-foundation-v3.plan.json", encoding="utf-8").read()
    )

    assert config.model == QWEN35_397B_MODEL
    assert config.model_revision == QWEN35_397B_REVISION
    assert gold.model == config.model
    assert gold.model_revision == config.model_revision
    assert gold.validation_corpus_dir == "build/gold-defense-release/validation"
    assert plan["active_training_target"]["model"] == QWEN35_397B_MODEL
    assert plan["single_model_policy"]["additional_active_models_allowed"] is False
    assert "development_specialization_tier" not in plan
