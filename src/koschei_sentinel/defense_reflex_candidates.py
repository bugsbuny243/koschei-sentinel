from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Literal

from pydantic import Field

from koschei_sentinel.cyber_range import CyberRangeReport, ScenarioTruth
from koschei_sentinel.models import StrictModel


class DefenseReflexFailureType(StrEnum):
    NO_CONTAINMENT = "NO_CONTAINMENT"
    MISSED_REROUTE = "MISSED_REROUTE"
    BENIGN_HIGH_IMPACT = "BENIGN_HIGH_IMPACT"
    RANGE_GATE_FAILURE = "RANGE_GATE_FAILURE"


class DefenseReflexReviewStatus(StrEnum):
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class DefenseReflexCandidate(StrictModel):
    schema_version: Literal["sentinel.defense-reflex-candidate.v1"] = (
        "sentinel.defense-reflex-candidate.v1"
    )
    candidate_id: str
    scenario_id: str
    failure_type: DefenseReflexFailureType
    source_report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    failing_ticks: list[int] = Field(default_factory=list)
    observed_modes: list[str] = Field(default_factory=list)
    attempted_actions: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    review_status: DefenseReflexReviewStatus = DefenseReflexReviewStatus.REVIEW_REQUIRED
    training_authorization: bool = False


def _report_sha256(report: CyberRangeReport) -> str:
    return hashlib.sha256(report.model_dump_json().encode("utf-8")).hexdigest()


def _candidate(
    report: CyberRangeReport,
    *,
    failure_type: DefenseReflexFailureType,
    failing_ticks: list[int],
    rationale: list[str],
) -> DefenseReflexCandidate:
    digest = _report_sha256(report)
    suffix = failure_type.value.lower().replace("_", "-")
    modes = [report.ticks[index].defense_mode.value for index in failing_ticks]
    actions: list[str] = []
    for index in failing_ticks:
        actions.extend(action.value for action in report.ticks[index].attempted_actions)
    return DefenseReflexCandidate(
        candidate_id=f"reflex:{report.scenario_id}:{suffix}",
        scenario_id=report.scenario_id,
        failure_type=failure_type,
        source_report_sha256=digest,
        failing_ticks=failing_ticks,
        observed_modes=list(dict.fromkeys(modes)),
        attempted_actions=list(dict.fromkeys(actions)),
        rationale=rationale,
        review_status=DefenseReflexReviewStatus.REVIEW_REQUIRED,
        training_authorization=False,
    )


def mine_defense_reflex_candidates(report: CyberRangeReport) -> list[DefenseReflexCandidate]:
    candidates: list[DefenseReflexCandidate] = []

    if report.truth is ScenarioTruth.MALICIOUS and report.containment_tick is None:
        candidates.append(
            _candidate(
                report,
                failure_type=DefenseReflexFailureType.NO_CONTAINMENT,
                failing_ticks=[tick.tick for tick in report.ticks],
                rationale=[
                    "malicious scenario reached the end of the simulated timeline without verified containment",
                    "candidate requires review of evidence thresholds, cut-point choice, sequencing and simulated outcomes",
                ],
            )
        )

    missed = [
        tick.tick
        for tick in report.ticks
        if tick.reroute_expected and not tick.reroute_detected
    ]
    if missed:
        candidates.append(
            _candidate(
                report,
                failure_type=DefenseReflexFailureType.MISSED_REROUTE,
                failing_ticks=missed,
                rationale=[
                    "expected attacker reroute was not recognized by adaptive reassessment",
                    "candidate requires review of graph deltas and alternate-path reasoning",
                ],
            )
        )

    benign_harm_ticks = [
        tick.tick
        for tick in report.ticks
        if report.truth is ScenarioTruth.BENIGN
        and any(
            action.value
            not in {"OBSERVE", "COLLECT_EVIDENCE", "BLOCK_IOC"}
            for action in tick.attempted_actions
        )
    ]
    if benign_harm_ticks:
        candidates.append(
            _candidate(
                report,
                failure_type=DefenseReflexFailureType.BENIGN_HIGH_IMPACT,
                failing_ticks=benign_harm_ticks,
                rationale=[
                    "benign scenario received a high-impact containment action",
                    "candidate requires review of confidence calibration and human-protective guard behavior",
                ],
            )
        )

    if report.violations and not candidates:
        candidates.append(
            _candidate(
                report,
                failure_type=DefenseReflexFailureType.RANGE_GATE_FAILURE,
                failing_ticks=[tick.tick for tick in report.ticks],
                rationale=list(report.violations),
            )
        )

    return candidates
