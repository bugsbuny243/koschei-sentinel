from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.assured_multi_incident_defense import AssuredMultiIncidentDefensePlan
from koschei_sentinel.defense_authority import DefenseActionType, DefenseMode
from koschei_sentinel.interception_planner import (
    InterceptionUrgency,
    build_interception_plan,
)
from koschei_sentinel.models import StrictModel


class DefenseResourceClass(StrEnum):
    EVIDENCE = "EVIDENCE"
    NETWORK = "NETWORK"
    IDENTITY = "IDENTITY"
    ENDPOINT = "ENDPOINT"
    CLOUD = "CLOUD"
    CICD = "CICD"
    SIGNER = "SIGNER"
    TRANSACTION = "TRANSACTION"
    EMERGENCY = "EMERGENCY"


class SchedulingDisposition(StrEnum):
    SCHEDULED = "SCHEDULED"
    DEFERRED_GLOBAL_CAPACITY = "DEFERRED_GLOBAL_CAPACITY"
    DEFERRED_RESOURCE_CAPACITY = "DEFERRED_RESOURCE_CAPACITY"
    NO_AUTHORIZED_STEP = "NO_AUTHORIZED_STEP"


_ACTION_RESOURCE = {
    DefenseActionType.OBSERVE: DefenseResourceClass.EVIDENCE,
    DefenseActionType.COLLECT_EVIDENCE: DefenseResourceClass.EVIDENCE,
    DefenseActionType.BLOCK_IOC: DefenseResourceClass.NETWORK,
    DefenseActionType.REVOKE_CREDENTIAL: DefenseResourceClass.IDENTITY,
    DefenseActionType.TERMINATE_SESSION: DefenseResourceClass.IDENTITY,
    DefenseActionType.KILL_PROCESS: DefenseResourceClass.ENDPOINT,
    DefenseActionType.ISOLATE_ENDPOINT: DefenseResourceClass.ENDPOINT,
    DefenseActionType.QUARANTINE_WORKLOAD: DefenseResourceClass.CLOUD,
    DefenseActionType.PAUSE_PIPELINE: DefenseResourceClass.CICD,
    DefenseActionType.FREEZE_SIGNER: DefenseResourceClass.SIGNER,
    DefenseActionType.HOLD_TRANSACTION: DefenseResourceClass.TRANSACTION,
    DefenseActionType.ENABLE_EMERGENCY_POLICY: DefenseResourceClass.EMERGENCY,
}

_URGENCY_BONUS = {
    InterceptionUrgency.IMMEDIATE: 0.30,
    InterceptionUrgency.HIGH: 0.15,
    InterceptionUrgency.NORMAL: 0.0,
}

_MODE_RANK = {
    DefenseMode.GUARD: 0,
    DefenseMode.COMBAT: 1,
    DefenseMode.SIEGE: 2,
}


class DefenseResourcePolicy(StrictModel):
    schema_version: Literal["sentinel.defense-resource-policy.v1"] = (
        "sentinel.defense-resource-policy.v1"
    )
    max_parallel_total: int = Field(default=4, ge=1, le=1024)
    reserved_critical_slots: int = Field(default=1, ge=0, le=1024)
    resource_capacities: dict[DefenseResourceClass, int] = Field(
        default_factory=lambda: {
            DefenseResourceClass.EVIDENCE: 8,
            DefenseResourceClass.NETWORK: 4,
            DefenseResourceClass.IDENTITY: 2,
            DefenseResourceClass.ENDPOINT: 2,
            DefenseResourceClass.CLOUD: 2,
            DefenseResourceClass.CICD: 1,
            DefenseResourceClass.SIGNER: 1,
            DefenseResourceClass.TRANSACTION: 2,
            DefenseResourceClass.EMERGENCY: 1,
        }
    )
    wait_cycle_boost: float = Field(default=0.05, ge=0.0, le=1.0)
    max_wait_boost: float = Field(default=0.30, ge=0.0, le=2.0)

    @model_validator(mode="after")
    def capacities_are_complete_and_coherent(self) -> "DefenseResourcePolicy":
        missing = sorted(
            set(DefenseResourceClass) - set(self.resource_capacities),
            key=lambda row: row.value,
        )
        if missing:
            raise ValueError(
                "defense resource policy is missing capacities: "
                + ", ".join(row.value for row in missing)
            )
        if any(value < 0 for value in self.resource_capacities.values()):
            raise ValueError("defense resource capacities cannot be negative")
        if self.reserved_critical_slots > self.max_parallel_total:
            raise ValueError("reserved critical slots cannot exceed max_parallel_total")
        return self


class DefenseSchedulerState(StrictModel):
    schema_version: Literal["sentinel.defense-scheduler-state.v1"] = (
        "sentinel.defense-scheduler-state.v1"
    )
    wait_cycles_by_component: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def wait_cycles_are_valid(self) -> "DefenseSchedulerState":
        if any(value < 0 for value in self.wait_cycles_by_component.values()):
            raise ValueError("scheduler wait cycles cannot be negative")
        return self


class DefenseScheduleItem(StrictModel):
    component_id: str
    component_entity_ids: list[str]
    critical_entity_ids: list[str]
    step_id: str
    action: DefenseActionType
    target_entity_id: str
    resource_class: DefenseResourceClass
    defense_mode: DefenseMode
    urgency: InterceptionUrgency
    expected_effect_score: float = Field(ge=0.0, le=1.0)
    priority_score: float = Field(ge=0.0, le=4.0)
    wait_cycles: int = Field(ge=0)
    predicted_impact_present: bool
    supporting_relation_ids: list[str]
    disposition: SchedulingDisposition
    rationale: list[str]

    @model_validator(mode="after")
    def target_remains_inside_component(self) -> "DefenseScheduleItem":
        if self.target_entity_id not in set(self.component_entity_ids):
            raise ValueError("scheduled defense target crossed the attack component boundary")
        return self


class DefenseResourceSchedule(StrictModel):
    schema_version: Literal["sentinel.defense-resource-schedule.v1"] = (
        "sentinel.defense-resource-schedule.v1"
    )
    graph_id: str
    schedule_id: str
    policy: DefenseResourcePolicy
    scheduled: list[DefenseScheduleItem]
    deferred: list[DefenseScheduleItem]
    no_action_component_ids: list[str]
    resource_usage: dict[DefenseResourceClass, int]
    previous_state: DefenseSchedulerState
    next_state: DefenseSchedulerState
    schedule_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    rationale: list[str]

    @model_validator(mode="after")
    def schedule_obeys_parallelism(self) -> "DefenseResourceSchedule":
        scheduled_components = [row.component_id for row in self.scheduled]
        if len(scheduled_components) != len(set(scheduled_components)):
            raise ValueError("only one interception step per component may be scheduled in a wave")
        scheduled_targets = [
            (row.resource_class, row.target_entity_id) for row in self.scheduled
        ]
        if len(scheduled_targets) != len(set(scheduled_targets)):
            raise ValueError("the same resource target cannot be scheduled twice in one wave")
        if len(self.scheduled) > self.policy.max_parallel_total:
            raise ValueError("scheduled wave exceeds global parallel capacity")
        for resource, used in self.resource_usage.items():
            if used > self.policy.resource_capacities[resource]:
                raise ValueError("scheduled wave exceeds a resource-class capacity")
        return self


def _priority(
    *,
    component_priority: float,
    critical: bool,
    mode: DefenseMode,
    urgency: InterceptionUrgency,
    expected_effect: float,
    predicted_impact: bool,
    wait_cycles: int,
    policy: DefenseResourcePolicy,
) -> float:
    # The assured component priority already contains risk, criticality and effective
    # mode. Scheduling can add local urgency and bounded aging but cannot raise authority.
    wait_bonus = min(policy.max_wait_boost, wait_cycles * policy.wait_cycle_boost)
    impact_bonus = 0.20 if predicted_impact else 0.0
    mode_tiebreak = 0.02 * _MODE_RANK[mode]
    critical_tiebreak = 0.05 if critical else 0.0
    return min(
        4.0,
        component_priority
        + _URGENCY_BONUS[urgency]
        + 0.20 * expected_effect
        + impact_bonus
        + wait_bonus
        + mode_tiebreak
        + critical_tiebreak,
    )


def _candidate_items(
    plan: AssuredMultiIncidentDefensePlan,
    state: DefenseSchedulerState,
    policy: DefenseResourcePolicy,
) -> tuple[list[DefenseScheduleItem], list[str]]:
    candidates: list[DefenseScheduleItem] = []
    no_action: list[str] = []
    for component in plan.component_plans:
        active = component.assured_plan.active_defense_plan
        interception = build_interception_plan(active)
        if not interception.steps:
            no_action.append(component.component_id)
            continue
        step = interception.steps[0]
        resource = _ACTION_RESOURCE[step.action]
        wait_cycles = state.wait_cycles_by_component.get(component.component_id, 0)
        candidates.append(
            DefenseScheduleItem(
                component_id=component.component_id,
                component_entity_ids=component.entity_ids,
                critical_entity_ids=component.critical_entity_ids,
                step_id=step.step_id,
                action=step.action,
                target_entity_id=step.target_entity_id,
                resource_class=resource,
                defense_mode=component.assured_plan.assurance.effective_mode,
                urgency=step.urgency,
                expected_effect_score=step.expected_effect_score,
                priority_score=_priority(
                    component_priority=component.priority_score,
                    critical=bool(component.critical_entity_ids),
                    mode=component.assured_plan.assurance.effective_mode,
                    urgency=step.urgency,
                    expected_effect=step.expected_effect_score,
                    predicted_impact=interception.predicted_impact_present,
                    wait_cycles=wait_cycles,
                    policy=policy,
                ),
                wait_cycles=wait_cycles,
                predicted_impact_present=interception.predicted_impact_present,
                supporting_relation_ids=step.supporting_relation_ids,
                disposition=SchedulingDisposition.SCHEDULED,
                rationale=[
                    "candidate is the first currently authorized interception step for its component",
                    "scheduler priority cannot raise the component above its assured defense mode",
                ],
            )
        )
    candidates.sort(
        key=lambda row: (
            -int(bool(row.critical_entity_ids)),
            -row.priority_score,
            -_MODE_RANK[row.defense_mode],
            row.component_id,
            row.step_id,
        )
    )
    return candidates, sorted(no_action)


def _can_allocate(
    item: DefenseScheduleItem,
    *,
    scheduled_count: int,
    usage: dict[DefenseResourceClass, int],
    policy: DefenseResourcePolicy,
) -> SchedulingDisposition | None:
    if scheduled_count >= policy.max_parallel_total:
        return SchedulingDisposition.DEFERRED_GLOBAL_CAPACITY
    if usage[item.resource_class] >= policy.resource_capacities[item.resource_class]:
        return SchedulingDisposition.DEFERRED_RESOURCE_CAPACITY
    return None


def _allocate(
    item: DefenseScheduleItem,
    *,
    scheduled: list[DefenseScheduleItem],
    deferred: list[DefenseScheduleItem],
    usage: dict[DefenseResourceClass, int],
    policy: DefenseResourcePolicy,
) -> None:
    reason = _can_allocate(
        item,
        scheduled_count=len(scheduled),
        usage=usage,
        policy=policy,
    )
    if reason is None:
        scheduled.append(
            item.model_copy(update={"disposition": SchedulingDisposition.SCHEDULED})
        )
        usage[item.resource_class] += 1
        return
    deferred.append(
        item.model_copy(
            update={
                "disposition": reason,
                "rationale": item.rationale
                + [
                    "step remains authorized by its component plan but is deferred by scheduler capacity"
                ],
            }
        )
    )


def _schedule_digest(
    *,
    graph_id: str,
    policy: DefenseResourcePolicy,
    scheduled: list[DefenseScheduleItem],
    deferred: list[DefenseScheduleItem],
    no_action: list[str],
    previous_state: DefenseSchedulerState,
    next_state: DefenseSchedulerState,
) -> str:
    payload = {
        "graph_id": graph_id,
        "policy": policy.model_dump(mode="json"),
        "scheduled": [row.model_dump(mode="json") for row in scheduled],
        "deferred": [row.model_dump(mode="json") for row in deferred],
        "no_action": no_action,
        "previous_state": previous_state.model_dump(mode="json"),
        "next_state": next_state.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def defense_resource_schedule_sha256(schedule: DefenseResourceSchedule) -> str:
    return _schedule_digest(
        graph_id=schedule.graph_id,
        policy=schedule.policy,
        scheduled=schedule.scheduled,
        deferred=schedule.deferred,
        no_action=schedule.no_action_component_ids,
        previous_state=schedule.previous_state,
        next_state=schedule.next_state,
    )


def verify_defense_resource_schedule(
    schedule: DefenseResourceSchedule,
) -> DefenseResourceSchedule:
    expected = defense_resource_schedule_sha256(schedule)
    if expected != schedule.schedule_sha256:
        raise ValueError("defense resource schedule digest does not match its contents")
    expected_id = f"schedule:{schedule.graph_id}:{expected[:20]}"
    if schedule.schedule_id != expected_id:
        raise ValueError("defense resource schedule_id does not match its digest")
    return schedule


def build_defense_resource_schedule(
    plan: AssuredMultiIncidentDefensePlan,
    *,
    policy: DefenseResourcePolicy | None = None,
    state: DefenseSchedulerState | None = None,
) -> DefenseResourceSchedule:
    gate = policy or DefenseResourcePolicy()
    previous = state or DefenseSchedulerState()
    candidates, no_action = _candidate_items(plan, previous, gate)
    scheduled: list[DefenseScheduleItem] = []
    deferred: list[DefenseScheduleItem] = []
    usage = {resource: 0 for resource in DefenseResourceClass}

    critical = [row for row in candidates if row.critical_entity_ids]

    # Reserve a bounded number of slots for components touching declared critical assets.
    # Unused reservations are released to the global queue immediately.
    for item in critical[: gate.reserved_critical_slots]:
        _allocate(
            item,
            scheduled=scheduled,
            deferred=deferred,
            usage=usage,
            policy=gate,
        )

    already_considered = {row.component_id for row in scheduled + deferred}
    remaining = [
        row for row in candidates if row.component_id not in already_considered
    ]
    remaining.sort(
        key=lambda row: (
            -row.priority_score,
            -int(bool(row.critical_entity_ids)),
            -_MODE_RANK[row.defense_mode],
            row.component_id,
        )
    )
    for item in remaining:
        _allocate(
            item,
            scheduled=scheduled,
            deferred=deferred,
            usage=usage,
            policy=gate,
        )

    next_wait: dict[str, int] = {}
    scheduled_ids = {row.component_id for row in scheduled}
    deferred_ids = {row.component_id for row in deferred}
    for component in plan.component_plans:
        if component.component_id in scheduled_ids:
            next_wait[component.component_id] = 0
        elif component.component_id in deferred_ids:
            next_wait[component.component_id] = (
                previous.wait_cycles_by_component.get(component.component_id, 0) + 1
            )
        else:
            next_wait[component.component_id] = previous.wait_cycles_by_component.get(
                component.component_id,
                0,
            )
    next_state = DefenseSchedulerState(
        wait_cycles_by_component=dict(sorted(next_wait.items()))
    )

    scheduled.sort(key=lambda row: (-row.priority_score, row.component_id))
    deferred.sort(key=lambda row: (-row.priority_score, row.component_id))
    digest = _schedule_digest(
        graph_id=plan.graph_id,
        policy=gate,
        scheduled=scheduled,
        deferred=deferred,
        no_action=no_action,
        previous_state=previous,
        next_state=next_state,
    )
    schedule = DefenseResourceSchedule(
        graph_id=plan.graph_id,
        schedule_id=f"schedule:{plan.graph_id}:{digest[:20]}",
        policy=gate,
        scheduled=scheduled,
        deferred=deferred,
        no_action_component_ids=no_action,
        resource_usage=usage,
        previous_state=previous,
        next_state=next_state,
        schedule_sha256=digest,
        rationale=[
            "resource scheduling never grants authority beyond the assured component plan",
            "only the first interception step of a component can enter a scheduling wave",
            "critical protected-asset components receive bounded reserved capacity",
            "per-control-plane capacities prevent one defensive subsystem from being overcommitted",
            "deferred components accumulate bounded deterministic wait aging to reduce starvation",
            "execution and post-action verification remain separate mandatory gates",
        ],
    )
    return verify_defense_resource_schedule(schedule)
