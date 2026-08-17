from __future__ import annotations

from statistics import mean
from typing import Literal

from pydantic import Field

from koschei_sentinel.cyber_range import (
    CyberRangeReport,
    CyberRangeScenario,
    ScenarioTruth,
    run_cyber_range_scenario,
)
from koschei_sentinel.models import StrictModel


class CyberRangeGatePolicy(StrictModel):
    schema_version: Literal["sentinel.cyber-range-gate-policy.v1"] = (
        "sentinel.cyber-range-gate-policy.v1"
    )
    malicious_containment_rate_min: float = Field(default=0.95, ge=0.0, le=1.0)
    reroute_detection_rate_min: float = Field(default=0.90, ge=0.0, le=1.0)
    benign_high_impact_false_positive_rate_max: float = Field(
        default=0.01, ge=0.0, le=1.0
    )
    mean_containment_tick_max: float = Field(default=2.0, ge=0.0)


class CyberRangeSuiteReport(StrictModel):
    schema_version: Literal["sentinel.cyber-range-suite-report.v1"] = (
        "sentinel.cyber-range-suite-report.v1"
    )
    scenarios: int = Field(ge=1)
    malicious_scenarios: int = Field(ge=0)
    benign_scenarios: int = Field(ge=0)
    malicious_contained: int = Field(ge=0)
    malicious_containment_rate: float = Field(ge=0.0, le=1.0)
    reroutes_expected: int = Field(ge=0)
    reroutes_detected: int = Field(ge=0)
    reroute_detection_rate: float = Field(ge=0.0, le=1.0)
    benign_with_high_impact_false_positive: int = Field(ge=0)
    benign_high_impact_false_positive_rate: float = Field(ge=0.0, le=1.0)
    mean_containment_tick: float | None = Field(default=None, ge=0.0)
    passed: bool
    violations: list[str]
    scenario_reports: list[CyberRangeReport]


def run_cyber_range_suite(
    scenarios: list[CyberRangeScenario],
    *,
    policy: CyberRangeGatePolicy | None = None,
) -> CyberRangeSuiteReport:
    if not scenarios:
        raise ValueError("cyber range suite requires at least one scenario")
    gate = policy or CyberRangeGatePolicy()
    reports = [run_cyber_range_scenario(row) for row in scenarios]

    malicious = [row for row in reports if row.truth is ScenarioTruth.MALICIOUS]
    benign = [row for row in reports if row.truth is ScenarioTruth.BENIGN]
    malicious_contained = sum(row.containment_tick is not None for row in malicious)
    containment_rate = (
        1.0 if not malicious else malicious_contained / len(malicious)
    )

    reroutes_expected = sum(row.reroutes_expected for row in reports)
    reroutes_detected = sum(row.reroutes_detected for row in reports)
    reroute_rate = (
        1.0 if reroutes_expected == 0 else reroutes_detected / reroutes_expected
    )

    benign_harm = sum(
        row.false_positive_high_impact_actions > 0 for row in benign
    )
    benign_harm_rate = 0.0 if not benign else benign_harm / len(benign)

    containment_ticks = [
        float(row.containment_tick)
        for row in malicious
        if row.containment_tick is not None
    ]
    mean_tick = mean(containment_ticks) if containment_ticks else None

    violations: list[str] = []
    if containment_rate < gate.malicious_containment_rate_min:
        violations.append(
            "malicious containment rate below gate: "
            f"{containment_rate:.4f} < {gate.malicious_containment_rate_min:.4f}"
        )
    if reroute_rate < gate.reroute_detection_rate_min:
        violations.append(
            "attacker reroute detection rate below gate: "
            f"{reroute_rate:.4f} < {gate.reroute_detection_rate_min:.4f}"
        )
    if benign_harm_rate > gate.benign_high_impact_false_positive_rate_max:
        violations.append(
            "benign high-impact false-positive rate above gate: "
            f"{benign_harm_rate:.4f} > "
            f"{gate.benign_high_impact_false_positive_rate_max:.4f}"
        )
    if malicious and mean_tick is None:
        violations.append("no malicious scenario reached verified containment")
    elif mean_tick is not None and mean_tick > gate.mean_containment_tick_max:
        violations.append(
            "mean containment tick above gate: "
            f"{mean_tick:.4f} > {gate.mean_containment_tick_max:.4f}"
        )

    return CyberRangeSuiteReport(
        scenarios=len(reports),
        malicious_scenarios=len(malicious),
        benign_scenarios=len(benign),
        malicious_contained=malicious_contained,
        malicious_containment_rate=containment_rate,
        reroutes_expected=reroutes_expected,
        reroutes_detected=reroutes_detected,
        reroute_detection_rate=reroute_rate,
        benign_with_high_impact_false_positive=benign_harm,
        benign_high_impact_false_positive_rate=benign_harm_rate,
        mean_containment_tick=mean_tick,
        passed=not violations,
        violations=violations,
        scenario_reports=reports,
    )
