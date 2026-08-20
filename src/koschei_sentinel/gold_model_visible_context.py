from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from koschei_sentinel.cyber_range import CyberRangeScenario
from koschei_sentinel.training import canonical_json

_FORBIDDEN_VISIBLE_KEYS = frozenset(
    {
        "action_outcomes",
        "evaluation_authorization",
        "expected_interpretation",
        "expected_reroute_ticks",
        "expected_sequence",
        "expected_steps",
        "failure_candidates",
        "packet_sha256",
        "promotion_eligible",
        "range_report",
        "review_only_context",
        "review_sha256",
        "reviewer_id",
        "scenario_truth",
        "simulated_action_outcomes",
        "training_authorization",
        "truth",
    }
)


def build_gold_model_visible_context(
    scenario: CyberRangeScenario,
) -> dict[str, object]:
    return {
        "scenario_id": scenario.scenario_id,
        "critical_entity_ids": sorted(scenario.critical_entity_ids),
        "graph_snapshots": [
            graph.model_dump(mode="json") for graph in scenario.graph_snapshots
        ],
    }


def gold_model_visible_context_sha256(context: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json(dict(context)).encode("utf-8")).hexdigest()


def _forbidden_paths(value: object, *, prefix: str = "$") -> list[str]:
    violations: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}"
            if key_text in _FORBIDDEN_VISIBLE_KEYS:
                violations.append(path)
            violations.extend(_forbidden_paths(nested, prefix=path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            violations.extend(_forbidden_paths(nested, prefix=f"{prefix}[{index}]"))
    return violations


def assert_gold_model_visible_context_is_answer_key_safe(
    context: Mapping[str, object],
) -> None:
    expected_top_level = {"scenario_id", "critical_entity_ids", "graph_snapshots"}
    observed_top_level = set(context)
    if observed_top_level != expected_top_level:
        extra = sorted(observed_top_level - expected_top_level)
        missing = sorted(expected_top_level - observed_top_level)
        detail: list[str] = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if extra:
            detail.append("extra=" + ",".join(extra))
        raise ValueError(
            "Gold model-visible context violates the answer-key-isolated top-level contract: "
            + "; ".join(detail)
        )

    forbidden = _forbidden_paths(context)
    if forbidden:
        raise ValueError(
            "Gold model-visible context contains forbidden answer-key/review fields: "
            + ", ".join(forbidden[:8])
        )


def verify_gold_model_visible_context(
    *,
    scenario: CyberRangeScenario,
    context: Mapping[str, object],
    context_sha256: str,
) -> None:
    expected = build_gold_model_visible_context(scenario)
    assert_gold_model_visible_context_is_answer_key_safe(expected)
    assert_gold_model_visible_context_is_answer_key_safe(context)
    if dict(context) != expected:
        raise ValueError("Gold model-visible context differs from the canonical scenario view")
    expected_sha = gold_model_visible_context_sha256(expected)
    if context_sha256 != expected_sha:
        raise ValueError("Gold model-visible context SHA-256 differs from canonical scenario view")
