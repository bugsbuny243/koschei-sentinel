from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from koschei_sentinel.adapters import (
    AdapterError,
    CandidateSpec,
    _request_payload,
    build_adapter,
    load_candidate_registry,
    parse_chat_completion,
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


def provider_envelope(content: dict[str, object], *, finish_reason: str = "stop") -> str:
    return json.dumps(
        {
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {"content": json.dumps(content)},
                }
            ]
        }
    )


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


def test_together_candidate_requires_json_schema_output() -> None:
    raw = together_spec().model_dump(mode="json")
    raw["json_mode"] = "json-object"
    with pytest.raises(ValidationError, match="json-schema"):
        CandidateSpec.model_validate(raw)


def test_reasoning_controls_are_mutually_exclusive() -> None:
    raw = together_spec().model_dump(mode="json")
    raw["reasoning_enabled"] = False
    with pytest.raises(ValidationError, match="mutually exclusive"):
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


def test_case_aware_schema_requires_every_provider_field() -> None:
    spec = together_spec()
    benchmark = suite()[0]
    payload = _request_payload(spec, benchmark)
    schema = payload["response_format"]["json_schema"]["schema"]
    properties = schema["properties"]

    assert set(schema["required"]) == set(properties)
    assert properties["case_id"]["enum"] == [benchmark.case.case_id]
    assert properties["verdict_signature"]["enum"] == [
        benchmark.case.signed_verdict.signature
    ]
    assert properties["engine"]["enum"] == [spec.candidate_id]
    assert properties["claims"]["minItems"] == 1
    assert properties["claims"]["maxItems"] == 2
    claim_properties = properties["claims"]["items"]["properties"]
    assert claim_properties["evidence_ids"]["items"]["enum"] == ["E-HOLDER-001"]
    assert claim_properties["confidence"]["enum"] == ["VERIFIED"]


def test_case_aware_schema_requires_exact_abstention_limitations() -> None:
    spec = together_spec()
    missing_evidence = suite()[1]
    no_rule = suite()[2]

    missing_schema = _request_payload(spec, missing_evidence)["response_format"][
        "json_schema"
    ]["schema"]
    missing_properties = missing_schema["properties"]
    assert missing_properties["claims"]["maxItems"] == 0
    assert missing_properties["limitations"]["minItems"] == 1
    assert missing_properties["limitations"]["maxItems"] == 1
    assert missing_properties["limitations"]["items"]["enum"] == ["KS-LINK-404"]

    no_rule_schema = _request_payload(spec, no_rule)["response_format"]["json_schema"][
        "schema"
    ]
    assert no_rule_schema["properties"]["claims"]["maxItems"] == 0
    assert no_rule_schema["properties"]["limitations"]["items"]["enum"] == [
        "No triggered rule"
    ]


def test_together_payload_uses_schema_and_reasoning_budget() -> None:
    registry = load_candidate_registry(REGISTRY_PATH)
    gpt_oss, qwen = registry.candidates

    gpt_payload = _request_payload(gpt_oss, suite()[0])
    assert gpt_payload["reasoning_effort"] == "low"
    assert gpt_payload["max_tokens"] == 1024
    assert gpt_payload["response_format"]["type"] == "json_schema"
    assert gpt_payload["response_format"]["json_schema"]["name"] == "sentinel_opinion"
    system_prompt = gpt_payload["messages"][0]["content"]
    assert "Required JSON Schema" in system_prompt
    assert "Case-specific mandatory requirements" in system_prompt
    assert "No triggered rule" in system_prompt

    qwen_payload = _request_payload(qwen, suite()[0])
    assert qwen_payload["reasoning"] == {"enabled": False}
    assert "reasoning_effort" not in qwen_payload


def test_sparse_default_like_provider_output_is_rejected() -> None:
    benchmark = suite()[0]
    sparse = {
        "schema_version": "sentinel.opinion.v1",
        "case_id": benchmark.case.case_id,
        "verdict_signature": benchmark.case.signed_verdict.signature,
    }
    with pytest.raises(AdapterError) as error:
        parse_chat_completion(provider_envelope(sparse))
    assert error.value.code == "malformed_provider_response"


def test_complete_provider_output_is_converted_to_sentinel_opinion() -> None:
    benchmark = suite()[0]
    candidate = together_spec().candidate_id
    complete = {
        "schema_version": "sentinel.opinion.v1",
        "case_id": benchmark.case.case_id,
        "verdict_signature": benchmark.case.signed_verdict.signature,
        "authority": (
            "The signed deterministic verdict is final; this output is commentary only."
        ),
        "assessment": "EXPLANATION_ONLY",
        "claims": [
            {
                "text": "The resolved holder controls a material supply share.",
                "evidence_ids": ["E-HOLDER-001"],
                "confidence": "VERIFIED",
            }
        ],
        "limitations": [],
        "recommended_actions": [],
        "engine": candidate,
    }
    opinion = parse_chat_completion(
        provider_envelope(complete),
        benchmark=benchmark,
        expected_candidate=candidate,
    )
    assert opinion.case_id == benchmark.case.case_id
    assert opinion.engine == candidate
    assert opinion.claims[0].evidence_ids == ["E-HOLDER-001"]
    assert opinion.generated_at is not None


def test_provider_identity_mismatch_fails_closed() -> None:
    benchmark = suite()[2]
    candidate = together_spec().candidate_id
    wrong = {
        "schema_version": "sentinel.opinion.v1",
        "case_id": "changed-case",
        "verdict_signature": benchmark.case.signed_verdict.signature,
        "authority": (
            "The signed deterministic verdict is final; this output is commentary only."
        ),
        "assessment": "EXPLANATION_ONLY",
        "claims": [],
        "limitations": ["No triggered rule"],
        "recommended_actions": [],
        "engine": candidate,
    }
    with pytest.raises(AdapterError) as error:
        parse_chat_completion(
            provider_envelope(wrong),
            benchmark=benchmark,
            expected_candidate=candidate,
        )
    assert error.value.code == "provider_identity_mismatch"


def test_truncated_reasoning_output_gets_specific_error() -> None:
    response = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": ""},
            }
        ]
    }
    with pytest.raises(AdapterError) as error:
        parse_chat_completion(json.dumps(response))
    assert error.value.code == "provider_output_truncated"


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
