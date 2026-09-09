from pathlib import Path

import pytest

from koschei_sentinel.web4_research_readiness import (
    Web4ResearchReadinessReport,
    evaluate_web4_research_readiness,
)

_ROOT = Path(__file__).resolve().parents[1]


def _evaluate() -> Web4ResearchReadinessReport:
    return evaluate_web4_research_readiness(
        source_registry_path=_ROOT / "configs/corpus/web4-v1.sources.proposed.jsonl",
        protocol_tracking_path=_ROOT / "configs/corpus/web4-v1.protocol-tracking.json",
        curriculum_path=_ROOT / "configs/curriculum/web4-security.v1.json",
        benchmark_path=_ROOT / "evals/web4-security-benchmark.v1.json",
        event_schema_path=_ROOT / "schemas/web4-security-event-v1.schema.json",
    )


def test_web4_research_contract_is_internally_consistent() -> None:
    report = _evaluate()

    assert report.ready_for_research is True
    assert report.training_allowed is False
    assert report.source_count >= 22
    assert report.protocol_family_count >= 7
    assert report.violations == []


def test_web4_research_receipt_rejects_tampering() -> None:
    report = _evaluate()
    tampered = report.model_dump(mode="json")
    tampered["source_count"] += 1

    with pytest.raises(ValueError, match="self-hash does not verify"):
        Web4ResearchReadinessReport.model_validate(tampered)
