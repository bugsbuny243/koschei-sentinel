from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.adapters import (
    AdapterError,
    CandidateCostPlan,
    CandidateRegistry,
    build_adapter,
    plan_candidate_cost,
)
from koschei_sentinel.benchmark import (
    BenchmarkCase,
    BenchmarkReport,
    BenchmarkThresholds,
    PredictionRecord,
    evaluate_benchmark,
)
from koschei_sentinel.models import StrictModel


class CandidateOutcome(StrictModel):
    candidate_id: str
    adapter: str
    model: str | None = None
    status: Literal["passed", "failed", "error"]
    report: BenchmarkReport | None = None
    error_code: str | None = None
    error_message: str | None = None
    cost_plan: CandidateCostPlan | None = None


class ComparisonMatrix(StrictModel):
    schema_version: Literal["sentinel.comparison-matrix.v1"] = (
        "sentinel.comparison-matrix.v1"
    )
    suite_digest: str
    registry_digest: str
    comparison_digest: str
    thresholds: BenchmarkThresholds
    total_candidates: int
    passed_candidates: int
    failed_candidates: int
    errored_candidates: int
    any_candidate_passed: bool
    all_candidates_passed: bool
    eligible_candidates: list[str] = Field(default_factory=list)
    ranking: list[str] = Field(default_factory=list)
    candidates: list[CandidateOutcome]


@dataclass(frozen=True)
class ComparisonRun:
    matrix: ComparisonMatrix
    predictions: dict[str, list[PredictionRecord]]
    reports: dict[str, BenchmarkReport]


def run_comparison(
    suite: list[BenchmarkCase],
    registry: CandidateRegistry,
    *,
    registry_base_dir: Path,
    thresholds: BenchmarkThresholds | None = None,
    allow_network: bool = False,
    allow_local_network: bool = False,
) -> ComparisonRun:
    if not suite:
        raise ValueError("benchmark suite must contain at least one case")
    active_thresholds = thresholds or BenchmarkThresholds()
    outcomes: list[CandidateOutcome] = []
    predictions_by_candidate: dict[str, list[PredictionRecord]] = {}
    reports_by_candidate: dict[str, BenchmarkReport] = {}

    for spec in sorted(registry.candidates, key=lambda item: item.candidate_id):
        cost_plan = plan_candidate_cost(spec, suite)
        try:
            adapter = build_adapter(
                spec,
                base_dir=registry_base_dir,
                allow_network=allow_network,
                allow_local_network=allow_local_network,
            )
            predictions = adapter.predict(suite)
            report = evaluate_benchmark(
                suite,
                predictions,
                thresholds=active_thresholds,
            )
            if report.candidate != spec.candidate_id:
                raise AdapterError(
                    "candidate_mismatch",
                    "adapter output does not match the configured candidate_id",
                )
            predictions_by_candidate[spec.candidate_id] = predictions
            reports_by_candidate[spec.candidate_id] = report
            outcomes.append(
                CandidateOutcome(
                    candidate_id=spec.candidate_id,
                    adapter=spec.adapter,
                    model=spec.model,
                    status="passed" if report.gate_passed else "failed",
                    report=report,
                    cost_plan=cost_plan,
                )
            )
        except AdapterError as exc:
            outcomes.append(
                CandidateOutcome(
                    candidate_id=spec.candidate_id,
                    adapter=spec.adapter,
                    model=spec.model,
                    status="error",
                    error_code=exc.code,
                    error_message=str(exc),
                    cost_plan=cost_plan,
                )
            )
        except (OSError, ValueError):
            outcomes.append(
                CandidateOutcome(
                    candidate_id=spec.candidate_id,
                    adapter=spec.adapter,
                    model=spec.model,
                    status="error",
                    error_code="benchmark_rejected",
                    error_message="candidate output was rejected by the benchmark contract",
                    cost_plan=cost_plan,
                )
            )

    ranking = [item.candidate_id for item in sorted(outcomes, key=_ranking_key)]
    eligible = sorted(item.candidate_id for item in outcomes if item.status == "passed")
    passed = len(eligible)
    failed = sum(item.status == "failed" for item in outcomes)
    errored = sum(item.status == "error" for item in outcomes)
    suite_digest = _digest_cases(suite)
    registry_digest = _digest_registry(registry)
    digest_payload = {
        "suite_digest": suite_digest,
        "registry_digest": registry_digest,
        "thresholds": active_thresholds.model_dump(mode="json"),
        "ranking": ranking,
        "candidates": [item.model_dump(mode="json") for item in outcomes],
    }
    comparison_digest = hashlib.sha256(_canonical_json(digest_payload).encode()).hexdigest()
    matrix = ComparisonMatrix(
        suite_digest=suite_digest,
        registry_digest=registry_digest,
        comparison_digest=comparison_digest,
        thresholds=active_thresholds,
        total_candidates=len(outcomes),
        passed_candidates=passed,
        failed_candidates=failed,
        errored_candidates=errored,
        any_candidate_passed=passed > 0,
        all_candidates_passed=passed == len(outcomes),
        eligible_candidates=eligible,
        ranking=ranking,
        candidates=outcomes,
    )
    return ComparisonRun(
        matrix=matrix,
        predictions=predictions_by_candidate,
        reports=reports_by_candidate,
    )


def write_comparison(run: ComparisonRun, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        (staging / "comparison-matrix.json").write_text(
            json.dumps(run.matrix.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for candidate_id in sorted(run.predictions):
            candidate_dir = staging / candidate_id
            candidate_dir.mkdir()
            candidate_dir.joinpath("predictions.jsonl").write_text(
                _prediction_payload(run.predictions[candidate_id]),
                encoding="utf-8",
            )
            candidate_dir.joinpath("benchmark-report.json").write_text(
                json.dumps(
                    run.reports[candidate_id].model_dump(mode="json"),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        _fsync_tree(staging)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _ranking_key(outcome: CandidateOutcome) -> tuple[float | str, ...]:
    if outcome.report is None:
        return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, outcome.candidate_id)
    report = outcome.report
    return (
        0.0 if report.gate_passed else 0.5,
        -report.case_pass_rate,
        -report.authority_score,
        -report.grounding_score,
        -report.abstention_score,
        -report.privacy_score,
        outcome.candidate_id,
    )


def _digest_cases(suite: list[BenchmarkCase]) -> str:
    payload = "".join(
        _canonical_json(item.model_dump(mode="json")) + "\n"
        for item in sorted(suite, key=lambda value: value.test_id)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _digest_registry(registry: CandidateRegistry) -> str:
    return hashlib.sha256(
        _canonical_json(registry.model_dump(mode="json")).encode()
    ).hexdigest()


def _prediction_payload(predictions: list[PredictionRecord]) -> str:
    rows = []
    for item in sorted(predictions, key=lambda value: value.test_id):
        payload = item.model_dump(mode="json")
        payload["opinion"].pop("generated_at", None)
        rows.append(_canonical_json(payload) + "\n")
    return "".join(rows)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in [*directories, root]:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
