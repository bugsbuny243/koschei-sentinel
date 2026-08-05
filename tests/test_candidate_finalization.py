from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from koschei_sentinel.autotrain import AutotrainPlan
from koschei_sentinel.benchmark import BenchmarkReport, BenchmarkThresholds
from koschei_sentinel.candidate_finalization import (
    CandidateFinalizationBlocked,
    finalize_candidate,
    write_finalization_bundle,
)
from koschei_sentinel.incubation_registry import empty_registry
from koschei_sentinel.matrix import CandidateOutcome, ComparisonMatrix
from koschei_sentinel.offline_receipt import OfflineTrainingReceipt
from koschei_sentinel.training import AdapterManifest, model_digest

_CANDIDATE = "sentinel-finalize-v1"


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _autotrain() -> AutotrainPlan:
    return AutotrainPlan(
        decision="ready_for_offline_training",
        training_run_id=_CANDIDATE,
        policy_digest="1" * 64,
        training_plan_digest="2" * 64,
        dataset_manifest_digest="3" * 64,
        readiness_report_digest="4" * 64,
        reasons=[],
        warnings=[],
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
        output_dir="build/training/sentinel-finalize-v1",
    )


def _receipt(adapter: AdapterManifest) -> OfflineTrainingReceipt:
    payload = {
        "schema_version": "sentinel.offline-training-receipt.v1",
        "receipt_id": _CANDIDATE,
        "state": "completed_offline",
        "candidate_stage": "incubation_candidate",
        "training_run_id": _CANDIDATE,
        "job_digest": "8" * 64,
        "adapter_manifest_digest": model_digest(adapter),
        "adapter_digest": adapter.adapter_digest,
        "dataset_manifest_digest": adapter.dataset_manifest_digest,
        "training_config_digest": adapter.training_config_digest,
        "base_model": adapter.base_model,
        "base_revision": adapter.base_revision,
        "output_dir": adapter.output_dir,
        "adapter_files": list(adapter.adapter_files),
        "adapter_files_verified": True,
        "benchmark_required": True,
        "benchmark_passed": False,
        "automatic_registration_allowed": False,
        "automatic_promotion_allowed": False,
        "production_deployment_allowed": False,
    }
    return OfflineTrainingReceipt.model_validate(
        {**payload, "receipt_digest": _digest(payload)}
    )


def _comparison(*, digest_override: str | None = None) -> ComparisonMatrix:
    report = BenchmarkReport(
        candidate=_CANDIDATE,
        suite_digest="9" * 64,
        prediction_digest="a" * 64,
        total_cases=1,
        passed_cases=1,
        failed_cases=0,
        case_pass_rate=1.0,
        authority_score=1.0,
        grounding_score=1.0,
        abstention_score=1.0,
        privacy_score=1.0,
        gate_passed=True,
        thresholds=BenchmarkThresholds(),
        cases=[],
    )
    outcome = CandidateOutcome(
        candidate_id=_CANDIDATE,
        adapter="replay",
        status="passed",
        report=report,
    )
    payload = {
        "suite_digest": report.suite_digest,
        "registry_digest": "b" * 64,
        "thresholds": BenchmarkThresholds().model_dump(mode="json"),
        "ranking": [_CANDIDATE],
        "candidates": [outcome.model_dump(mode="json")],
    }
    return ComparisonMatrix(
        suite_digest=report.suite_digest,
        registry_digest="b" * 64,
        comparison_digest=digest_override or _digest(payload),
        thresholds=BenchmarkThresholds(),
        total_candidates=1,
        passed_candidates=1,
        failed_candidates=0,
        errored_candidates=0,
        any_candidate_passed=True,
        all_candidates_passed=True,
        eligible_candidates=[_CANDIDATE],
        ranking=[_CANDIDATE],
        candidates=[outcome],
    )


def test_receipt_and_benchmark_finalize_one_incubation_bundle(tmp_path: Path) -> None:
    adapter = _adapter()
    registry = empty_registry()

    finalization, updated_registry, record = finalize_candidate(
        _CANDIDATE,
        _receipt(adapter),
        _autotrain(),
        adapter,
        _comparison(),
        registry,
    )
    output = tmp_path / "finalized"
    write_finalization_bundle(finalization, updated_registry, output)

    assert finalization.state == "finalized_incubation"
    assert finalization.authority == "explanation_only"
    assert finalization.previous_registry_digest == registry.registry_digest
    assert finalization.updated_registry_digest == updated_registry.registry_digest
    assert finalization.candidate_record_digest == record.record_digest
    assert finalization.automatic_promotion_allowed is False
    assert finalization.production_deployment_allowed is False
    assert updated_registry.records == [record]
    assert (output / "candidate-finalization.json").is_file()
    assert (output / "incubation-registry.json").is_file()


def test_tampered_receipt_digest_is_rejected() -> None:
    adapter = _adapter()
    receipt = _receipt(adapter).model_copy(update={"receipt_digest": "0" * 64})

    with pytest.raises(CandidateFinalizationBlocked, match="receipt.*digest"):
        finalize_candidate(
            _CANDIDATE,
            receipt,
            _autotrain(),
            adapter,
            _comparison(),
            empty_registry(),
        )


def test_tampered_comparison_digest_is_rejected() -> None:
    adapter = _adapter()

    with pytest.raises(CandidateFinalizationBlocked, match="comparison matrix digest"):
        finalize_candidate(
            _CANDIDATE,
            _receipt(adapter),
            _autotrain(),
            adapter,
            _comparison(digest_override="0" * 64),
            empty_registry(),
        )


def test_receipt_from_another_adapter_is_rejected() -> None:
    adapter = _adapter()
    receipt = _receipt(adapter).model_copy(update={"adapter_digest": "f" * 64})
    payload = receipt.model_dump(mode="json")
    payload.pop("receipt_digest")
    receipt = receipt.model_copy(update={"receipt_digest": _digest(payload)})

    with pytest.raises(CandidateFinalizationBlocked, match="adapter digest"):
        finalize_candidate(
            _CANDIDATE,
            receipt,
            _autotrain(),
            adapter,
            _comparison(),
            empty_registry(),
        )


def test_finalization_bundle_is_not_overwritten(tmp_path: Path) -> None:
    adapter = _adapter()
    finalization, registry, _ = finalize_candidate(
        _CANDIDATE,
        _receipt(adapter),
        _autotrain(),
        adapter,
        _comparison(),
        empty_registry(),
    )
    output = tmp_path / "finalized"
    write_finalization_bundle(finalization, registry, output)

    with pytest.raises(FileExistsError, match="already exists"):
        write_finalization_bundle(finalization, registry, output)
