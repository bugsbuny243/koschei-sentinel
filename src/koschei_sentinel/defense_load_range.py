from __future__ import annotations

from statistics import mean
from typing import Literal

from pydantic import Field

from koschei_sentinel.assured_multi_incident_defense import (
    AssuredComponentDefensePlan,
    AssuredMultiIncidentDefensePlan,
)
from koschei_sentinel.defense_authority import DefenseMode
from koschei_sentinel.defense_resource_scheduler import (
    DefenseResourceClass,
    DefenseResourcePolicy,
    DefenseSchedulerState,
    build_defense_resource_schedule,
)
from koschei_sentinel.interception_planner import build_interception_plan
from koschei_sentinel.models import StrictModel

_MODE_RANK = {
    DefenseMode.GUARD: 0,
    DefenseMode.COMBAT: 1,
    DefenseMode.SIEGE: 2,
}


class DefenseLoadGatePolicy(StrictModel):
    schema_version: Literal["sentinel.defense-load-gate-policy.v1"] = (
        "sentinel.defense-load-gate-policy.v1"
    )
    max_waves: int = Field(default=64, ge=1, le=10000)
    service_coverage_min: float = Field(default=1.0, ge=0.0, le=1.0)
    critical_first_service_wave_max: int = Field(default=1, ge=0)
    max_wait_cycles_allowed: int = Field(default=8, ge=0)
    capacity_violation_max: int = Field(default=0, ge=0)


class DefenseLoadWave(StrictModel):
    wave: int = Field(ge=0)
    pending_before: int = Field(ge=0)
    scheduled_component_ids: list[str]
    deferred_component_ids: list[str]
    no_action_component_ids: list[str]
    resource_usage: dict[DefenseResourceClass, int]
    max_wait_cycles_after_wave: int = Field(ge=0)


class DefenseLoadRangeReport(StrictModel):
    schema_version: Literal["sentinel.defense-load-range-report.v1"] = (
        "sentinel.defense-load-range-report.v1"
    )
    graph_id: str
    resource_policy: DefenseResourcePolicy
    gate_policy: DefenseLoadGatePolicy
    eligible_components: int = Field(ge=0)
    critical_eligible_components: int = Field(ge=0)
    serviced_components: int = Field(ge=0)
    service_coverage: float = Field(ge=0.0, le=1.0)
    waves_run: int = Field(ge=0)
    mean_first_service_wave: float | None = Field(default=None, ge=0.0)
    max_first_service_wave: int | None = Field(default=None, ge=0)
    critical_max_first_service_wave: int | None = Field(default=None, ge=0)
    max_wait_cycles: int = Field(ge=0)
    starved_component_ids: list[str]
    critical_starved_component_ids: list[str]
    max_resource_utilization: dict[DefenseResourceClass, float]
    capacity_violation_count: int = Field(ge=0)
    passed: bool
    violations: list[str]
    waves: list[DefenseLoadWave]


def _eligible_components(
    plan: AssuredMultiIncidentDefensePlan,
) -> list[AssuredComponentDefensePlan]:
    return [
        row
        for row in plan.component_plans
        if build_interception_plan(row.assured_plan.active_defense_plan).steps
    ]


def _subplan(
    source: AssuredMultiIncidentDefensePlan,
    remaining: list[AssuredComponentDefensePlan],
) -> AssuredMultiIncidentDefensePlan:
    highest = max(
        (row.assured_plan.assurance.effective_mode for row in remaining),
        key=lambda mode: _MODE_RANK[mode],
        default=DefenseMode.GUARD,
    )
    return AssuredMultiIncidentDefensePlan(
        graph_id=source.graph_id,
        active_component_count=len(remaining),
        critical_component_count=sum(bool(row.critical_entity_ids) for row in remaining),
        high_impact_authorized_components=sum(
            row.assured_plan.assurance.high_impact_authorized for row in remaining
        ),
        highest_effective_mode=highest,
        component_plans=remaining,
        inactive_critical_entity_ids=source.inactive_critical_entity_ids,
        rationale=source.rationale
        + ["defense load range subplan contains only not-yet-serviced components"],
    )


def run_defense_load_range(
    plan: AssuredMultiIncidentDefensePlan,
    *,
    resource_policy: DefenseResourcePolicy | None = None,
    gate_policy: DefenseLoadGatePolicy | None = None,
) -> DefenseLoadRangeReport:
    resources = resource_policy or DefenseResourcePolicy()
    gate = gate_policy or DefenseLoadGatePolicy()
    eligible = _eligible_components(plan)
    eligible_ids = {row.component_id for row in eligible}
    critical_ids = {
        row.component_id for row in eligible if row.critical_entity_ids
    }
    remaining = list(eligible)
    state = DefenseSchedulerState()
    first_service_wave: dict[str, int] = {}
    waves: list[DefenseLoadWave] = []
    max_wait = 0
    capacity_violations = 0
    max_utilization = {resource: 0.0 for resource in DefenseResourceClass}

    for wave in range(gate.max_waves):
        if not remaining:
            break
        pending_before = len(remaining)
        current = _subplan(plan, remaining)
        schedule = build_defense_resource_schedule(
            current,
            policy=resources,
            state=state,
        )
        scheduled_ids = [row.component_id for row in schedule.scheduled]
        deferred_ids = [row.component_id for row in schedule.deferred]
        for component_id in scheduled_ids:
            first_service_wave.setdefault(component_id, wave)

        for resource in DefenseResourceClass:
            used = schedule.resource_usage[resource]
            capacity = resources.resource_capacities[resource]
            if used > capacity:
                capacity_violations += 1
            utilization = 0.0 if capacity == 0 else used / capacity
            max_utilization[resource] = max(max_utilization[resource], utilization)
        if len(schedule.scheduled) > resources.max_parallel_total:
            capacity_violations += 1

        state = schedule.next_state
        max_wait_after = max(state.wait_cycles_by_subject.values(), default=0)
        max_wait = max(max_wait, max_wait_after)
        waves.append(
            DefenseLoadWave(
                wave=wave,
                pending_before=pending_before,
                scheduled_component_ids=sorted(scheduled_ids),
                deferred_component_ids=sorted(deferred_ids),
                no_action_component_ids=schedule.no_action_component_ids,
                resource_usage=schedule.resource_usage,
                max_wait_cycles_after_wave=max_wait_after,
            )
        )
        scheduled_set = set(scheduled_ids)
        remaining = [row for row in remaining if row.component_id not in scheduled_set]
        if not scheduled_set:
            # No forward progress means current resource capacities cannot service the
            # remaining authorized action classes. Stop instead of spinning forever.
            break

    serviced_ids = set(first_service_wave)
    starved = sorted(eligible_ids - serviced_ids)
    critical_starved = sorted(critical_ids - serviced_ids)
    service_waves = list(first_service_wave.values())
    critical_waves = [
        first_service_wave[component_id]
        for component_id in critical_ids
        if component_id in first_service_wave
    ]
    service_coverage = (
        1.0 if not eligible_ids else len(serviced_ids) / len(eligible_ids)
    )

    violations: list[str] = []
    if service_coverage < gate.service_coverage_min:
        violations.append(
            "defense load service coverage below gate: "
            f"{service_coverage:.4f} < {gate.service_coverage_min:.4f}"
        )
    if starved:
        violations.append("one or more authorized attack components were starved")
    if critical_starved:
        violations.append("one or more critical attack components were starved")
    critical_max = max(critical_waves, default=None)
    if (
        critical_max is not None
        and critical_max > gate.critical_first_service_wave_max
    ):
        violations.append(
            "critical component first-service latency exceeded the load-range gate"
        )
    if max_wait > gate.max_wait_cycles_allowed:
        violations.append("scheduler wait cycles exceeded the load-range gate")
    if capacity_violations > gate.capacity_violation_max:
        violations.append("scheduler exceeded one or more configured resource capacities")

    return DefenseLoadRangeReport(
        graph_id=plan.graph_id,
        resource_policy=resources,
        gate_policy=gate,
        eligible_components=len(eligible_ids),
        critical_eligible_components=len(critical_ids),
        serviced_components=len(serviced_ids),
        service_coverage=service_coverage,
        waves_run=len(waves),
        mean_first_service_wave=(mean(service_waves) if service_waves else None),
        max_first_service_wave=max(service_waves, default=None),
        critical_max_first_service_wave=critical_max,
        max_wait_cycles=max_wait,
        starved_component_ids=starved,
        critical_starved_component_ids=critical_starved,
        max_resource_utilization=max_utilization,
        capacity_violation_count=capacity_violations,
        passed=not violations,
        violations=violations,
        waves=waves,
    )
