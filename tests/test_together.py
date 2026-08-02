from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from koschei_sentinel.adapters import (
    AdapterError,
    CandidateSpec,
    build_adapter,
    load_candidate_registry,
    plan_registry_costs,
    select_candidates,
)
from koschei_sentinel.benchmark import load_benchmark_suite
from koschei_sentinel.matrix import run_comparison
from koschei_sentinel.matrix_cli import main as matrix_main

FIXTURES = Path(__file__).parents[1] / "fixtures"
REGISTRY_PATH = FIXTURES / "models/candidates.together.low-cost.json"
SUITE_PATH = FIXTURES / "evals/suite.safe.jsonl"


def together_spec() -> CandidateSpec:
    return load_candidate_registry(REGISTRY_PATH).candidates[0]


def suite():
    return load_benchmark_suite(SUITE_PATH)


def test_together_candidate_cannot_override_endpoint() -> None:
    raw = together_spec().model_dump(mode="json")
    raw["base_url"] = "https://example.com/v1"
    with pytest.raises(ValidationError, match="cannot override"):
        CandidateSpec.model_validate(raw)


def test_together_candidate_requires_named_key_and_cost_policy() -> None:
    raw = together_spec().model_dump(mode="json")
    raw["api_key_env"] = "OTHER_KEY"
    with pytest.raises(ValidationError, match="TOGETHER_API_KEY"):
        CandidateSpec.model_validate(raw)

    raw = together_spec().model_dump(mode="json")
    raw["cost_policy"] = None
    with pytest.raises(ValidationError, match="cost policy"):
        CandidateSpec.model_validate(raw)


def test_low_cost_registry_plan_stays_below_one_cent_per_candidate() -> None:
    registry = load_candidate_registry(REGISTRY_PATH)
    plan = plan_registry_costs(registry, suite())
    assert plan.all_within_limits
    assert all(item.estimated_max_cost_usd is not None for item in plan.candidates)
    assert all(item.estimated_max_cost_usd < 0.01 for item in plan.candidates)


def test_together_preset_uses_fixed_official_endpoint() -> None:
    adapter = build_adapter(
        together_spec(),
        base_dir=REGISTRY_PATH.parent,
    )
    assert adapter.spec.base_url == "https://api.together.ai/v1"


def test_request_limit_fails_before_key_or_network_call() -> None:
    spec = together_spec()
    assert spec.cost_policy is not None
    constrained = spec.model_copy(
        update={"cost_policy": spec.cost_policy.model_copy(update={"max_requests": 2})}
    )
    adapter = build_adapter(
        constrained,
        base_dir=REGISTRY_PATH.parent,
        allow_network=True,
    )
    with pytest.raises(AdapterError) as error:
        adapter.predict(suite())
    assert error.value.code == "request_limit_exceeded"


def test_budget_limit_fails_before_key_or_network_call() -> None:
    spec = together_spec()
    assert spec.cost_policy is not None
    constrained = spec.model_copy(
        update={
            "cost_policy": spec.cost_policy.model_copy(
                update={"max_estimated_cost_usd": 0.000001}
            )
        }
    )
    adapter = build_adapter(
        constrained,
        base_dir=REGISTRY_PATH.parent,
        allow_network=True,
    )
    with pytest.raises(AdapterError) as error:
        adapter.predict(suite())
    assert error.value.code == "budget_exceeded"


def test_missing_key_fails_closed_after_budget_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    adapter = build_adapter(
        together_spec(),
        base_dir=REGISTRY_PATH.parent,
        allow_network=True,
    )
    with pytest.raises(AdapterError) as error:
        adapter.predict(suite())
    assert error.value.code == "missing_api_key"


def test_candidate_selection_rejects_unknown_ids() -> None:
    registry = load_candidate_registry(REGISTRY_PATH)
    selected = select_candidates(registry, ["sentinel-together-qwen3.5-9b"])
    assert [item.candidate_id for item in selected.candidates] == [
        "sentinel-together-qwen3.5-9b"
    ]
    with pytest.raises(ValueError, match="unknown candidate_id"):
        select_candidates(registry, ["missing"])


def test_plan_only_performs_no_network_call(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("plan-only attempted network access")

    monkeypatch.setattr("koschei_sentinel.adapters.urlopen", forbidden)
    exit_code = matrix_main(
        [
            "--suite",
            str(SUITE_PATH),
            "--registry",
            str(REGISTRY_PATH),
            "--candidate",
            "sentinel-together-gpt-oss-20b",
            "--plan-only",
        ]
    )
    assert exit_code == 0
    assert '"all_within_limits": true' in capsys.readouterr().out


def test_matrix_records_cost_plan_when_network_is_disabled() -> None:
    registry = select_candidates(
        load_candidate_registry(REGISTRY_PATH),
        ["sentinel-together-gpt-oss-20b"],
    )
    run = run_comparison(
        suite(),
        registry,
        registry_base_dir=REGISTRY_PATH.parent,
    )
    outcome = run.matrix.candidates[0]
    assert outcome.error_code == "network_disabled"
    assert outcome.cost_plan is not None
    assert outcome.cost_plan.within_budget
