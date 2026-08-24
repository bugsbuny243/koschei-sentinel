import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import koschei_sentinel.cyber_megatron_training as megatron_training
from koschei_sentinel.cyber_megatron_training import (
    LAUNCH_APPROVAL_ENV,
    LAUNCH_SESSION_ENV,
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

TEST_MASTER_ADDR = "127.0.0.1"
TEST_MASTER_PORT = 29_500


class _FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert tokenize is False
        assert enable_thinking is False
        rendered = "\n".join(
            f"{message['role']}:{message['content']}" for message in messages
        )
        if add_generation_prompt:
            rendered += "\nassistant:"
        return rendered

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": list(text.encode("utf-8"))}


@pytest.fixture(autouse=True)
def _pin_fake_tokenizer(monkeypatch) -> None:
    monkeypatch.setattr(
        megatron_training,
        "_load_pinned_tokenizer",
        lambda _config: _FakeTokenizer(),
    )


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
            "tensor_model_parallel_size": 4,
            "pipeline_model_parallel_size": 1,
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


def test_topology_requires_expert_parallelism_to_divide_data_parallelism() -> None:
    with pytest.raises(ValidationError, match="data parallel size"):
        CyberMegatronTopology(
            nodes=4,
            gpus_per_node=8,
            minimum_gpu_memory_gib=80,
            tensor_model_parallel_size=8,
            pipeline_model_parallel_size=1,
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
    assert _value(command, "--tensor_model_parallel_size") == "4"
    assert _value(command, "--pipeline_model_parallel_size") == "1"
    assert _value(command, "--expert_model_parallel_size") == "8"
    assert _value(command, "--add_version") == "true"
    assert _value(command, "--no_save_optim") == "false"
    assert _value(command, "--no_save_rng") == "false"
    assert _value(command, "--finetune") == "true"
    assert not any("35B-A3B" in token or "9B" in token for token in command)


def test_resume_command_restores_mcore_model_adapter_optimizer_and_rng() -> None:
    resume = CyberMegatronResume(
        mcore_model="/shared/checkpoints/model",
        mcore_adapter="/shared/checkpoints/adapter",
        binding_manifest="/shared/checkpoints/koschei-run-identity.json",
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
    assert plan.data_parallel_size == 8
    assert plan.gradient_accumulation_steps == 4
    assert plan.config_sha256
    assert plan.dataset_manifest_sha256 == verification.manifest_sha256
    assert plan.source_examples_sha256 == manifest.source_examples_sha256
    assert plan.source_manifest_sha256 == manifest.source_manifest_sha256
    assert plan.train_jsonl_sha256 == manifest.train_jsonl_sha256
    assert plan.validation_jsonl_sha256 == manifest.validation_jsonl_sha256
    assert plan.max_observed_sequence_tokens == manifest.max_observed_sequence_tokens
    assert plan.run_identity_sha256
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


def test_tokenization_preflight_blocks_overlength_examples(tmp_path, monkeypatch) -> None:
    class _OverlengthTokenizer(_FakeTokenizer):
        def __call__(self, text, *, add_special_tokens):
            base = super().__call__(text, add_special_tokens=add_special_tokens)[
                "input_ids"
            ]
            return {"input_ids": [token for token in base for _ in range(20)]}

    write_seed_curriculum(tmp_path / "build" / "cyber-training")
    config = _config()
    monkeypatch.setattr(
        megatron_training,
        "_load_pinned_tokenizer",
        lambda _config: _OverlengthTokenizer(),
    )

    with pytest.raises(ValueError, match="exceed optimization.max_length"):
        materialize_cyber_megatron_dataset(config, root=tmp_path)


def test_promotion_eligible_gold_split_requires_release_audit(
    tmp_path,
    monkeypatch,
) -> None:
    write_seed_curriculum(tmp_path / "build" / "cyber-training")
    source_config = _config()
    rows, examples_sha, manifest_sha, _ = megatron_training.load_cyber_sft_examples(
        megatron_training._source_config(source_config),
        root=tmp_path,
    )
    training_rows = rows[:-4]
    validation_rows = rows[-4:]
    monkeypatch.setattr(
        megatron_training,
        "load_cyber_sft_examples",
        lambda _config, *, root: (training_rows, examples_sha, manifest_sha, True),
    )
    monkeypatch.setattr(
        megatron_training,
        "load_cyber_sft_validation_examples",
        lambda _config, *, root: (
            validation_rows,
            "1" * 64,
            "2" * 64,
            True,
        ),
    )
    audit_calls = []

    def _invalid_audit(release_root):
        audit_calls.append(release_root)
        return SimpleNamespace(
            valid=False,
            violations=["holdout isolation failed"],
            audit_sha256="3" * 64,
        )

    monkeypatch.setattr(
        megatron_training,
        "audit_gold_defense_release",
        _invalid_audit,
    )
    payload = _payload()
    payload.update(
        corpus_dir="build/gold-defense-release/train",
        validation_corpus_dir="build/gold-defense-release/validation",
        validation_ratio=0.0,
    )
    config = CyberMegatronSFTConfig.model_validate(payload)

    with pytest.raises(ValueError, match="Gold Defense release audit failed"):
        materialize_cyber_megatron_dataset(config, root=tmp_path)
    assert audit_calls == [tmp_path / "build" / "gold-defense-release"]


def test_shared_output_waits_for_fresh_resume_session(
    tmp_path,
    monkeypatch,
) -> None:
    write_seed_curriculum(tmp_path / "build" / "cyber-training")
    config = _config()
    materialize_cyber_megatron_dataset(config, root=tmp_path)
    verification = verify_cyber_megatron_dataset(config, root=tmp_path)
    identity = megatron_training._run_identity(config, verification)
    session = "397b-test-session-0001"

    megatron_training._coordinate_launch(
        config,
        identity,
        root=tmp_path,
        node_rank=0,
        launch_session=session,
        master_addr=TEST_MASTER_ADDR,
        master_port=TEST_MASTER_PORT,
        resume=None,
    )
    megatron_training._coordinate_launch(
        config,
        identity,
        root=tmp_path,
        node_rank=1,
        launch_session=session,
        master_addr=TEST_MASTER_ADDR,
        master_port=TEST_MASTER_PORT,
        resume=None,
    )
    plan = plan_cyber_megatron_sft(
        config,
        root=tmp_path,
        _coordinated_identity_sha256=identity.identity_sha256,
    )
    assert plan.static_ready is True
    output = tmp_path / config.output_dir
    model = output / "checkpoint-25" / "model"
    adapter = output / "checkpoint-25" / "adapter"
    model.mkdir(parents=True)
    adapter.mkdir(parents=True)
    resume = CyberMegatronResume(
        mcore_model=str(model),
        mcore_adapter=str(adapter),
        binding_manifest=str(output / megatron_training.RUN_IDENTITY_FILENAME),
    )
    fresh_session = "397b-fresh-resume-session-0002"
    sleep_calls = []

    def _publish_fresh_state(_seconds):
        if not sleep_calls:
            megatron_training._write_launch_state(
                output,
                identity=identity,
                launch_session=fresh_session,
                master_addr=TEST_MASTER_ADDR,
                master_port=TEST_MASTER_PORT,
                nodes=config.topology.nodes,
                state="launching",
            )
        sleep_calls.append(True)

    monkeypatch.setattr(megatron_training.time, "sleep", _publish_fresh_state)
    megatron_training._coordinate_launch(
        config,
        identity,
        root=tmp_path,
        node_rank=1,
        launch_session=fresh_session,
        master_addr=TEST_MASTER_ADDR,
        master_port=TEST_MASTER_PORT,
        resume=resume,
    )
    assert sleep_calls
    with pytest.raises(RuntimeError, match="new unique"):
        megatron_training._coordinate_launch(
            config,
            identity,
            root=tmp_path,
            node_rank=0,
            launch_session=fresh_session,
            master_addr=TEST_MASTER_ADDR,
            master_port=TEST_MASTER_PORT,
            resume=resume,
        )


def test_all_nodes_must_pass_the_same_final_plan_before_launch(
    tmp_path,
    monkeypatch,
) -> None:
    write_seed_curriculum(tmp_path / "build" / "cyber-training")
    config = _config()
    materialize_cyber_megatron_dataset(config, root=tmp_path)
    verification = verify_cyber_megatron_dataset(config, root=tmp_path)
    identity = megatron_training._run_identity(config, verification)
    session = "397b-readiness-session-0001"
    megatron_training._coordinate_launch(
        config,
        identity,
        root=tmp_path,
        node_rank=0,
        launch_session=session,
        master_addr=TEST_MASTER_ADDR,
        master_port=TEST_MASTER_PORT,
        resume=None,
    )
    plan = plan_cyber_megatron_sft(
        config,
        root=tmp_path,
        _coordinated_identity_sha256=identity.identity_sha256,
    )
    output = tmp_path / config.output_dir

    monkeypatch.setattr(
        megatron_training,
        "ALL_NODE_READINESS_TIMEOUT_SECONDS",
        0.0,
    )
    with pytest.raises(RuntimeError, match="every node"):
        megatron_training._wait_for_all_nodes_ready(
            config,
            identity,
            plan,
            root=tmp_path,
            node_rank=0,
            launch_session=session,
            master_addr=TEST_MASTER_ADDR,
            master_port=TEST_MASTER_PORT,
        )

    monkeypatch.setattr(
        megatron_training,
        "ALL_NODE_READINESS_TIMEOUT_SECONDS",
        1.0,
    )
    for node_rank in range(1, config.topology.nodes):
        megatron_training._write_node_readiness(
            output,
            identity=identity,
            launch_session=session,
            master_addr=TEST_MASTER_ADDR,
            master_port=TEST_MASTER_PORT,
            plan=plan,
            node_rank=node_rank,
        )
    megatron_training._write_node_readiness(
        output,
        identity=identity,
        launch_session=session,
        master_addr="127.0.0.2",
        master_port=TEST_MASTER_PORT,
        plan=plan,
        node_rank=1,
    )
    with pytest.raises(RuntimeError, match="different rendezvous endpoint"):
        megatron_training._wait_for_all_nodes_ready(
            config,
            identity,
            plan,
            root=tmp_path,
            node_rank=0,
            launch_session=session,
            master_addr=TEST_MASTER_ADDR,
            master_port=TEST_MASTER_PORT,
        )
    megatron_training._write_node_readiness(
        output,
        identity=identity,
        launch_session=session,
        master_addr=TEST_MASTER_ADDR,
        master_port=TEST_MASTER_PORT,
        plan=plan,
        node_rank=1,
    )
    megatron_training._wait_for_all_nodes_ready(
        config,
        identity,
        plan,
        root=tmp_path,
        node_rank=0,
        launch_session=session,
        master_addr=TEST_MASTER_ADDR,
        master_port=TEST_MASTER_PORT,
    )
    state = megatron_training.CyberMegatronLaunchState.model_validate_json(
        (output / megatron_training.LAUNCH_STATE_FILENAME).read_bytes()
    )
    assert state.state == "ready"
    megatron_training._wait_for_all_nodes_ready(
        config,
        identity,
        plan,
        root=tmp_path,
        node_rank=1,
        launch_session=session,
        master_addr=TEST_MASTER_ADDR,
        master_port=TEST_MASTER_PORT,
    )


def test_resume_requires_matching_run_config_and_dataset_binding(tmp_path) -> None:
    write_seed_curriculum(tmp_path / "build" / "cyber-training")
    config = _config()
    materialize_cyber_megatron_dataset(config, root=tmp_path)
    verification = verify_cyber_megatron_dataset(config, root=tmp_path)
    identity = megatron_training._run_identity(config, verification)
    output = tmp_path / config.output_dir
    model = output / "checkpoint-25" / "model"
    adapter = output / "checkpoint-25" / "adapter"
    model.mkdir(parents=True)
    adapter.mkdir(parents=True)
    binding = output / megatron_training.RUN_IDENTITY_FILENAME
    megatron_training._write_run_identity(binding, identity)
    resume = CyberMegatronResume(
        mcore_model=str(model),
        mcore_adapter=str(adapter),
        binding_manifest=str(binding),
    )

    plan = plan_cyber_megatron_sft(config, root=tmp_path, resume=resume)
    assert plan.static_ready is True
    payload = _payload()
    payload["optimization"] = {"learning_rate": 0.0002}
    changed_config = CyberMegatronSFTConfig.model_validate(payload)
    changed = plan_cyber_megatron_sft(
        changed_config,
        root=tmp_path,
        resume=resume,
    )
    assert changed.static_ready is False
    assert any("run, config and dataset digests" in row for row in changed.blockers)


def test_runtime_rejects_duplicate_visible_gpu_indices(monkeypatch) -> None:
    config = _config()
    monkeypatch.setenv(LAUNCH_APPROVAL_ENV, config.run_id)
    monkeypatch.setenv(LAUNCH_SESSION_ENV, "397b-runtime-test-0001")
    monkeypatch.setenv("NNODES", "4")
    monkeypatch.setenv("NPROC_PER_NODE", "8")
    monkeypatch.setenv("NODE_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,6")
    monkeypatch.setattr(megatron_training.shutil, "which", lambda _name: "/bin/true")
    monkeypatch.setattr(
        megatron_training.importlib.metadata,
        "version",
        lambda _name: MS_SWIFT_VERSION,
    )

    with pytest.raises(RuntimeError, match="duplicate GPU indices"):
        megatron_training._assert_runtime_contract(config)


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
