from types import SimpleNamespace

from koschei_sentinel.gold_holdout_evaluation import GoldHoldoutInferenceCase
from koschei_sentinel.gold_holdout_inference_runner import _prepare_prompt


class FakeTokenizer:
    def __init__(self, tokens: int) -> None:
        self.tokens = tokens

    def apply_chat_template(self, *_args, **_kwargs) -> str:
        return "prompt"

    def __call__(self, *_args, **_kwargs):
        return {"input_ids": SimpleNamespace(shape=(1, self.tokens))}


def _case() -> GoldHoldoutInferenceCase:
    return GoldHoldoutInferenceCase(
        case_id="gold-holdout:test",
        scenario_id="scenario:test",
        input_context={
            "scenario_id": "scenario:test",
            "critical_entity_ids": ["entity:critical"],
            "graph_snapshots": [{"graph_id": "graph:test"}],
        },
        input_context_sha256="a" * 64,
    )


def test_prompt_preflight_accepts_case_inside_total_context_limit() -> None:
    encoded, prompt_tokens, failure = _prepare_prompt(
        _case(),
        FakeTokenizer(tokens=1024),
        2048,
        512,
    )

    assert encoded is not None
    assert prompt_tokens == 1024
    assert failure is None


def test_prompt_preflight_reserves_generation_budget_before_model_execution() -> None:
    encoded, prompt_tokens, failure = _prepare_prompt(
        _case(),
        FakeTokenizer(tokens=1600),
        2048,
        512,
    )

    assert encoded is None
    assert prompt_tokens == 1600
    assert failure is not None
    assert failure.failure_type == "PROMPT_TOO_LONG"
    assert "max_new_tokens=512" in failure.detail
    assert "total=2112" in failure.detail
    assert "max_sequence_length=2048" in failure.detail


def test_prompt_preflight_marks_prompt_overlength_without_model_execution() -> None:
    encoded, prompt_tokens, failure = _prepare_prompt(
        _case(),
        FakeTokenizer(tokens=2300),
        2048,
        64,
    )

    assert encoded is None
    assert prompt_tokens == 2300
    assert failure is not None
    assert failure.failure_type == "PROMPT_TOO_LONG"
    assert "max_sequence_length=2048" in failure.detail
