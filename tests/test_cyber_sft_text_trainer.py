import pytest

from koschei_sentinel.cyber_sft_text_trainer import (
    _assert_text_only_model,
    _text_lora_targets,
)


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
