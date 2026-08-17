from __future__ import annotations

from statistics import mean
from typing import Literal

from pydantic import Field

from koschei_sentinel.models import StrictModel
from koschei_sentinel.multi_incident_cyber_range import (
    MultiIncidentCyberRangeReport,
    MultiIncidentCyberRangeScenario,
    run_multi_incident_cyber_range,
)


class MultiIncidentCyberRangeGatePolicy(StrictModel):
    schema_version: Literal["sentinel.multi-incident-cyber-range-gate-policy.v1"] = (
        "sentinel.multi-incident-cyber-range-gate-policy.v1"
    )
    component_count_accuracy_min: float = Field(default=1.0, ge=0.0, le=1.0)
    world_line_transition_accuracy_min: float = Field(default=1.0, ge=0.0, le=1.0)
    component_containment_rate_min: float = Field(default=0.95, ge=0.0, le=1.0)
    world_line_containment_rate_min: float = Field(default=0.95, ge=0.0, le=1.0)
    cut_point_leakage_max: int = Field(default=0, ge=0)
    mean_containment_step_max: float = Field(default=3.0, ge=0.0)
    mean_world_line_containment_tick_latency_max: float = Field(default=2.0, ge=0.0)


class MultiIncidentCyberRangeSuiteReport(StrictModel):
    schema_version: Literal["sentinel.multi-incident-cyber-range-suite-report.v1"] = (
        "sentinel.multi-incident-cyber-range-suite-report.v1"
    )
    scenarios: int = Field(ge=1)
    component_count_accuracy: float = Field(ge=0.0, le=1.0)
    world_line_transition_accuracy: float = Field(ge=0.0, le=1.0)
    total_component_instances: int = Field(ge=0)
    contained_component_instances: int = Field(ge=0)
    component_containment_rate: float = Field(ge=0.0, le=1.0)
    total_world_lines: int = Field(ge=0)
    contained_world_lines: int = Field(ge=0)
    world_line_containment_rate: float = Field(ge=0.0, le=1.0)
    cut_point_leakage_count: int = Field(ge=0)
    mean_containment_step: float | None = Field(default=None, ge=0.0)
    mean_world_line_containment_tick_latency: float | None = Field(default=None, ge=0.0)
    passed: bool
    violations: list[str]
    scenario_reports: list[MultiIncidentCyberRangeReport]


def run_multi_incident_cyber_range_suite(
    scenarios: list[MultiIncidentCyberRangeScenario],
    *,
    policy: MultiIncidentCyberRangeGatePolicy | None = None,
) -> MultiIncidentCyberRangeSuiteReport:
    if not scenarios:
        raise ValueError("multi-incident cyber range suite requires at least one scenario")
    gate = policy or MultiIncidentCyberRangeGatePolicy()
    reports = [run_multi_incident_cyber_range(row) for row in scenarios]

    expected_component_ticks = sum(row.expected_component_ticks for row in reports)
    matched_component_ticks = sum(row.component_count_matches for row in reports)
    component_accuracy = (
        1.0 if expected_component_ticks == 0 else matched_component_ticks / expected_component_ticks
    )

    expected_lineage = sum(row.expected_world_line_transitions for row in reports)
    matched_lineage = sum(row.matched_world_line_transitions for row in reports)
    lineage_accuracy = 1.0 if expected_lineage == 0 else matched_lineage / expected_lineage

    total_components = sum(row.total_component_instances for row in reports)
    contained_components = sum(row.contained_component_instances for row in reports)
    containment_rate = 1.0 if total_components == 0 else contained_components / total_components

    total_world_lines = sum(row.world_line_count for row in reports)
    contained_world_lines = sum(row.contained_world_lines for row in reports)
    world_line_containment_rate = (
        1.0 if total_world_lines == 0 else contained_world_lines / total_world_lines
    )

    leakage = sum(row.cut_point_leakage_count for row in reports)
    containment_steps = [
        row.mean_containment_step for row in reports if row.mean_containment_step is not None
    ]
    mean_step = mean(containment_steps) if containment_steps else None
    world_line_latencies = [
        row.mean_world_line_containment_tick_latency
        for row in reports
        if row.mean_world_line_containment_tick_latency is not None
    ]
    mean_world_line_latency = mean(world_line_latencies) if world_line_latencies else None

    violations: list[str] = []
    if component_accuracy < gate.component_count_accuracy_min:
        violations.append(
            "component-count accuracy below gate: "
            f"{component_accuracy:.4f} < {gate.component_count_accuracy_min:.4f}"
        )
    if lineage_accuracy < gate.world_line_transition_accuracy_min:
        violations.append(
            "world-line transition accuracy below gate: "
            f"{lineage_accuracy:.4f} < {gate.world_line_transition_accuracy_min:.4f}"
        )
    if containment_rate < gate.component_containment_rate_min:
        violations.append(
            "component containment rate below gate: "
            f"{containment_rate:.4f} < {gate.component_containment_rate_min:.4f}"
        )
    if world_line_containment_rate < gate.world_line_containment_rate_min:
        violations.append(
            "world-line containment rate below gate: "
            f"{world_line_containment_rate:.4f} < {gate.world_line_containment_rate_min:.4f}"
        )
    if leakage > gate.cut_point_leakage_max:
        violations.append(
            "cross-component cut-point leakage above gate: "
            f"{leakage} > {gate.cut_point_leakage_max}"
        )
    if mean_step is not None and mean_step > gate.mean_containment_step_max:
        violations.append(
            "mean containment step above gate: "
            f"{mean_step:.4f} > {gate.mean_containment_step_max:.4f}"
        )
    if (
        mean_world_line_latency is not None
        and mean_world_line_latency > gate.mean_world_line_containment_tick_latency_max
    ):
        violations.append(
            "mean world-line containment tick latency above gate: "
            f"{mean_world_line_latency:.4f} > "
            f"{gate.mean_world_line_containment_tick_latency_max:.4f}"
        )

    return MultiIncidentCyberRangeSuiteReport(
        scenarios=len(reports),
        component_count_accuracy=component_accuracy,
        world_line_transition_accuracy=lineage_accuracy,
        total_component_instances=total_components,
        contained_component_instances=contained_components,
        component_containment_rate=containment_rate,
        total_world_lines=total_world_lines,
        contained_world_lines=contained_world_lines,
        world_line_containment_rate=world_line_containment_rate,
        cut_point_leakage_count=leakage,
        mean_containment_step=mean_step,
        mean_world_line_containment_tick_latency=mean_world_line_latency,
        passed=not violations,
        violations=violations,
        scenario_reports=reports,
    )
