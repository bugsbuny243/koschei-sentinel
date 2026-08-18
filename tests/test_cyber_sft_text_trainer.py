import pytest

from koschei_sentinel.cyber_sft_text_trainer import (
    _assert_text_only_model,
    _last_checkpoint,
    _prepare_resume_directory,
    _text_lora_targets,
)
from koschei_sentinel.cyber_sft_training import CyberSFTConfig, CyberSFTPlan


def test_text_lora_targets_resolve_qwen35_attention_and_linear_attention_modules() -> None:
    class FakeModel:
        def named_modules(self):
            return [
                ("model.layers.0.self_attn.q_proj", object()),
                ("model.layers.0.mlp.up_proj", object()),
                ("model.layers.1.linear_attn.in_proj_qkv", object()),
                ("model.layers.1.linear_attn.out_proj", object()),
            ]

    targets = _text_lora_targets(
        FakeModel(),
        ["q_proj", "up_proj", "in_proj_qkv", "out_proj"],
    )

    assert targets == [
        "model.layers.0.mlp.up_proj",
        "model.layers.0.self_attn.q_proj",
        "model.layers.1.linear_attn.in_proj_qkv",
        "model.layers.1.linear_attn.out_proj",
    ]


def test_text_lora_targets_reject_matching_vision_module() -> None:
    class FakeModel:
        def named_modules(self):
            return [
                ("model.layers.0.self_attn.q_proj", object()),
                ("model.visual.blocks.0.attn.q_proj", object()),
            ]

    with pytest.raises(RuntimeError, match="vision module"):
        _text_lora_targets(FakeModel(), ["q_proj"])


def test_text_executor_requires_official_qwen35_causal_lm_class() -> None:
    class WrongModel:
        def named_modules(self):
            return [("", self)]

    with pytest.raises(RuntimeError, match="Qwen3.5 text-only causal-LM class"):
        _assert_text_only_model(WrongModel())


def test_text_executor_accepts_expected_class_without_vision_modules() -> None:
    ExpectedModel = type(
        "Qwen3_5ForCausalLM",
        (),
        {
            "named_modules": lambda self: [
                ("", self),
                ("model.layers.0.self_attn.q_proj", object()),
            ]
        },
    )

    _assert_text_only_model(ExpectedModel())


def test_text_executor_rejects_expected_class_if_vision_module_is_present() -> None:
    ExpectedModel = type(
        "Qwen3_5ForCausalLM",
        (),
        {
            "named_modules": lambda self: [
                ("", self),
                ("model.visual.blocks.0", object()),
            ]
        },
    )

    with pytest.raises(RuntimeError, match="loaded vision modules unexpectedly"):
        _assert_text_only_model(ExpectedModel())


def _resume_config(*, learning_rate: float = 0.0001) -> CyberSFTConfig:
    return CyberSFTConfig(
        run_id="resume-test",
        stage="DEFENSE_REFLEX",
        base_model="Qwen/Qwen3.5-9B-Base",
        base_revision="a" * 40,
        corpus_dir="build/corpus",
        output_dir="build/runs/resume-test",
        learning_rate=learning_rate,
        minimum_cuda_memory_gb=0.0,
    )


def _resume_plan(config: CyberSFTConfig) -> CyberSFTPlan:
    return CyberSFTPlan(
        run_id=config.run_id,
        stage=config.stage,
        execution_profile=config.execution_profile,
        executable_with_current_trainer=True,
        corpus_promotion_eligible=False,
        base_model=config.base_model,
        base_revision=config.base_revision,
        corpus_examples_sha256="b" * 64,
        corpus_manifest_sha256="c" * 64,
        example_count=32,
        training_examples=29,
        validation_examples=3,
        effective_batch_size=config.effective_batch_size,
        estimated_optimizer_steps=8,
        input_adapter_dir=None,
        output_dir=config.output_dir,
        warnings=[],
    )


def test_resume_binding_rejects_changed_training_config(tmp_path) -> None:
    first = _resume_config()
    _prepare_resume_directory(tmp_path, first, _resume_plan(first))

    changed = _resume_config(learning_rate=0.0002)
    with pytest.raises(RuntimeError, match="resume binding differs"):
        _prepare_resume_directory(tmp_path, changed, _resume_plan(changed))


def test_last_checkpoint_selects_highest_optimizer_step(tmp_path) -> None:
    checkpoints = tmp_path / "checkpoints"
    (checkpoints / "checkpoint-2").mkdir(parents=True)
    (checkpoints / "checkpoint-8").mkdir()
    (checkpoints / "checkpoint-not-a-step").mkdir()

    selected = _last_checkpoint(checkpoints)
    assert selected is not None
    assert selected.name == "checkpoint-8"
