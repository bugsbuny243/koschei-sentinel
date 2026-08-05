from __future__ import annotations

import json
from pathlib import Path

import pytest

from koschei_sentinel.autotrain import AutotrainPlan
from koschei_sentinel.benchmark import BenchmarkReport, BenchmarkThresholds
from koschei_sentinel.incubation_registry import (
    CandidateRegistrationBlocked,
    append_candidate,
    build_candidate_record,
    empty_registry,
    load_registry,
    write_registry,
)
from koschei_sentinel.matrix import CandidateOutcome, ComparisonMatrix
from koschei_sentinel.training import AdapterManifest

_CANDIDATE = "sentinel-incubation-v1"


def _autotrain(*, decision: str = "ready_for_offline_training") -> AutotrainPlan:
    return AutotrainPlan.model_validate(
        {
            "decision": decision,
            "training_run_id": _CANDIDATE,
            "policy_digest": "1" * 64,
            "training_plan_digest": "2" * 64,
            "dataset_manifest_digest": "3" * 64,
            "readiness_report_digest": "4" * 64,
            "reasons": [] if decision == "ready_for_offline_training" else ["blocked"],
            "warnings": [],
        }
    )


def _adapter() -> AdapterManifest:
    return AdapterManifest(
        run_id=_CANDIDATE,
        base_model="koschei-fixture/model",
        base_revision="5" * 40,
        dataset_manifest_digest="3" * 64,
        training_config_digest="6" * 64,
        adapter_digest="7" * 64,
        adapter_files=["adapter/adapter_model.safetensors"],
        output_dir="build/training/sentinel-incubation-v1",
    )


def _comparison(*, passed: bool = True) -> ComparisonMatrix:
    report = BenchmarkReport(
        candidate=_CANDIDATE,
        suite_digest="8" * 64,
        prediction_digest="9" * 64,
        total_cases=1,
        passed_cases=1 if passed else 0,
        failed_cases=0 if passed else 1,
        case_pass_rate=1.0 if passed else 0.0,
        authority_score=1.0,
        grounding_score=1.0,
        abstention_score=1.0,
        privacy_score=1.0,
        gate_passed=passed,
        thresholds=BenchmarkThresholds(),
        cases=[],
    )
    outcome = CandidateOutcome(
        candidate_id=_CANDIDATE,
        adapter="replay",
        status="passed" if passed else "failed",
        report=report,
    )
    return ComparisonMatrix(
        suite_digest="8" * 64,
        registry_digest="a" * 64,
        comparison_digest="b" * 64,
        thresholds=BenchmarkThresholds(),
        total_candidates=1,
        passed_candidates=1 if passed else 0,
        failed_candidates=0 if passed else 1,
        errored_candidates=0,
        any_candidate_passed=passed,
        all_candidates_passed=passed,
        eligible_candidates=[_CANDIDATE] if passed else [],
        ranking=[_CANDIDATE],
        candidates=[outcome],
    )


def test_passing_artifacts_create_only_an_incubation_candidate() -> None:
    record = build_candidate_record(
        _CANDIDATE,
        _autotrain(),
        _adapter(),
        _comparison(),
    )

    assert record.stage == "incubation_candidate"
    assert record.authority == "explanation_only"
    assert record.automatic_promotion_allowed is False
    assert record.production_deployment_allowed is False
    assert record.dataset_manifest_digest == "3" * 64
    assert len(record.record_digest) == 64


def test_failed_benchmark_cannot_enter_registry() -> None:
    with pytest.raises(CandidateRegistrationBlocked, match="hard gates"):
        build_candidate_record(
            _CANDIDATE,
            _autotrain(),
            _adapter(),
            _comparison(passed=False),
        )


def test_blocked_autotrain_plan_cannot_enter_registry() -> None:
    with pytest.raises(CandidateRegistrationBlocked, match="did not authorize"):
        build_candidate_record(
            _CANDIDATE,
            _autotrain(decision="blocked"),
            _adapter(),
            _comparison(),
        )


def test_registry_is_append_only_and_rejects_duplicate_adapter() -> None:
    record = build_candidate_record(
        _CANDIDATE,
        _autotrain(),
        _adapter(),
        _comparison(),
    )
    registry = append_candidate(empty_registry(), record)

    with pytest.raises(ValueError, match="already registered"):
        append_candidate(registry, record)


def test_registry_detects_digest_tampering(tmp_path: Path) -> None:
    record = build_candidate_record(
        _CANDIDATE,
        _autotrain(),
        _adapter(),
        _comparison(),
    )
    path = tmp_path / "incubation-registry.json"
    write_registry(append_candidate(empty_registry(), record), path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["registry_digest"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        load_registry(path)
