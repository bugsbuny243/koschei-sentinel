from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.assured_multi_incident_defense import AssuredMultiIncidentDefensePlan
from koschei_sentinel.defense_resource_scheduler import (
    DefenseResourceSchedule,
    DefenseScheduleItem,
    verify_defense_resource_schedule,
)
from koschei_sentinel.interception_execution import (
    InterceptionExecution,
    InterceptionStepStatus,
    authorize_next_step,
    record_step_execution,
    start_interception_execution,
    verify_step_outcome,
)
from koschei_sentinel.interception_planner import InterceptionPlan, build_interception_plan
from koschei_sentinel.models import StrictModel

_DIGEST = r"^[a-f0-9]{64}$"


class DefenseWaveComponentStatus(StrEnum):
    PENDING_AUTHORIZATION = "PENDING_AUTHORIZATION"
    AUTHORIZED = "AUTHORIZED"
    EXECUTED = "EXECUTED"
    VERIFIED_SUCCEEDED = "VERIFIED_SUCCEEDED"
    VERIFIED_FAILED = "VERIFIED_FAILED"


class DefenseWaveComponentExecution(StrictModel):
    component_id: str
    scheduling_subject_id: str
    schedule_item: DefenseScheduleItem
    interception_plan: InterceptionPlan
    interception_plan_sha256: str = Field(pattern=_DIGEST)
    interception_execution: InterceptionExecution
    status: DefenseWaveComponentStatus = DefenseWaveComponentStatus.PENDING_AUTHORIZATION

    @model_validator(mode="after")
    def component_binding_is_valid(self) -> "DefenseWaveComponentExecution":
        if self.component_id != self.schedule_item.component_id:
            raise ValueError("wave component_id differs from schedule item")
        if self.scheduling_subject_id != self.schedule_item.scheduling_subject_id:
            raise ValueError("wave scheduling subject differs from schedule item")
        if _interception_plan_sha256(self.interception_plan) != self.interception_plan_sha256:
            raise ValueError("wave interception plan digest mismatch")
        if self.interception_execution.graph_id != self.interception_plan.graph_id:
            raise ValueError("wave interception execution graph differs from plan")
        scheduled_step = next(
            (row for row in self.interception_plan.steps if row.step_id == self.schedule_item.step_id),
            None,
        )
        if scheduled_step is None:
            raise ValueError("scheduled wave step is absent from interception plan")
        if (
            scheduled_step.action is not self.schedule_item.action
            or scheduled_step.target_entity_id != self.schedule_item.target_entity_id
        ):
            raise ValueError("scheduled wave action/target differs from interception plan")
        execution_step = next(
            (
                row
                for row in self.interception_execution.steps
                if row.step_id == self.schedule_item.step_id
            ),
            None,
        )
        if execution_step is None:
            raise ValueError("scheduled wave step is absent from interception execution")
        expected_status = {
            DefenseWaveComponentStatus.PENDING_AUTHORIZATION: InterceptionStepStatus.PENDING,
            DefenseWaveComponentStatus.AUTHORIZED: InterceptionStepStatus.AUTHORIZED,
            DefenseWaveComponentStatus.EXECUTED: InterceptionStepStatus.EXECUTED,
            DefenseWaveComponentStatus.VERIFIED_SUCCEEDED: InterceptionStepStatus.VERIFIED_SUCCEEDED,
            DefenseWaveComponentStatus.VERIFIED_FAILED: InterceptionStepStatus.VERIFIED_FAILED,
        }[self.status]
        if execution_step.status is not expected_status:
            raise ValueError("wave component status differs from interception execution state")
        return self


class DefenseWaveExecution(StrictModel):
    schema_version: Literal["sentinel.defense-wave-execution.v1"] = (
        "sentinel.defense-wave-execution.v1"
    )
    wave_id: str
    graph_id: str
    schedule_id: str
    schedule_sha256: str = Field(pattern=_DIGEST)
    schedule_context_id: str
    components: list[DefenseWaveComponentExecution]
    complete: bool = False
    succeeded_component_ids: list[str] = Field(default_factory=list)
    failed_component_ids: list[str] = Field(default_factory=list)
    wave_sha256: str = Field(pattern=_DIGEST)
    rationale: list[str]

    @model_validator(mode="after")
    def wave_integrity(self) -> "DefenseWaveExecution":
        ids = [row.component_id for row in self.components]
        if len(ids) != len(set(ids)):
            raise ValueError("defense wave contains duplicate component executions")
        subjects = [row.scheduling_subject_id for row in self.components]
        if len(subjects) != len(set(subjects)):
            raise ValueError("defense wave contains duplicate scheduling subjects")
        terminal = {
            DefenseWaveComponentStatus.VERIFIED_SUCCEEDED,
            DefenseWaveComponentStatus.VERIFIED_FAILED,
        }
        expected_complete = all(row.status in terminal for row in self.components)
        if self.complete != expected_complete:
            raise ValueError("defense wave completion does not match component terminal states")
        expected_succeeded = sorted(
            row.component_id
            for row in self.components
            if row.status is DefenseWaveComponentStatus.VERIFIED_SUCCEEDED
        )
        expected_failed = sorted(
            row.component_id
            for row in self.components
            if row.status is DefenseWaveComponentStatus.VERIFIED_FAILED
        )
        if self.succeeded_component_ids != expected_succeeded:
            raise ValueError("defense wave succeeded component list is inconsistent")
        if self.failed_component_ids != expected_failed:
            raise ValueError("defense wave failed component list is inconsistent")
        return self


def _interception_plan_sha256(plan: InterceptionPlan) -> str:
    payload = json.dumps(
        plan.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _wave_digest_payload(wave: DefenseWaveExecution) -> str:
    payload = {
        "wave_id": wave.wave_id,
        "graph_id": wave.graph_id,
        "schedule_id": wave.schedule_id,
        "schedule_sha256": wave.schedule_sha256,
        "schedule_context_id": wave.schedule_context_id,
        "components": [row.model_dump(mode="json") for row in wave.components],
        "complete": wave.complete,
        "succeeded_component_ids": wave.succeeded_component_ids,
        "failed_component_ids": wave.failed_component_ids,
        "rationale": wave.rationale,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def defense_wave_execution_sha256(wave: DefenseWaveExecution) -> str:
    return hashlib.sha256(_wave_digest_payload(wave).encode("utf-8")).hexdigest()


def verify_defense_wave_execution(wave: DefenseWaveExecution) -> DefenseWaveExecution:
    expected = defense_wave_execution_sha256(wave)
    if expected != wave.wave_sha256:
        raise ValueError("defense wave execution digest does not match its contents")
    return wave


def _finalize_wave(wave: DefenseWaveExecution) -> DefenseWaveExecution:
    succeeded = sorted(
        row.component_id
        for row in wave.components
        if row.status is DefenseWaveComponentStatus.VERIFIED_SUCCEEDED
    )
    failed = sorted(
        row.component_id
        for row in wave.components
        if row.status is DefenseWaveComponentStatus.VERIFIED_FAILED
    )
    terminal = {
        DefenseWaveComponentStatus.VERIFIED_SUCCEEDED,
        DefenseWaveComponentStatus.VERIFIED_FAILED,
    }
    complete = all(row.status in terminal for row in wave.components)
    updated = wave.model_copy(
        update={
            "complete": complete,
            "succeeded_component_ids": succeeded,
            "failed_component_ids": failed,
            "wave_sha256": "0" * 64,
        }
    )
    digest = defense_wave_execution_sha256(updated)
    return DefenseWaveExecution.model_validate(
        updated.model_copy(update={"wave_sha256": digest}).model_dump()
    )


def start_defense_wave_execution(
    schedule: DefenseResourceSchedule,
    multi_plan: AssuredMultiIncidentDefensePlan,
) -> DefenseWaveExecution:
    verify_defense_resource_schedule(schedule)
    if schedule.graph_id != multi_plan.graph_id:
        raise ValueError("defense resource schedule belongs to a different graph")
    if not schedule.scheduled:
        raise ValueError("defense wave requires at least one scheduled component")

    component_by_id = {row.component_id: row for row in multi_plan.component_plans}
    executions: list[DefenseWaveComponentExecution] = []
    for item in schedule.scheduled:
        component = component_by_id.get(item.component_id)
        if component is None:
            raise ValueError("scheduled component is absent from assured multi-incident plan")
        if item.target_entity_id not in set(component.entity_ids):
            raise ValueError("scheduled wave target crossed component boundary")
        active = component.assured_plan.active_defense_plan
        plan = build_interception_plan(active)
        if not plan.steps or plan.steps[0].step_id != item.step_id:
            raise ValueError("scheduled wave item is not the component's first interception step")
        if plan.steps[0].action is not item.action or plan.steps[0].target_entity_id != item.target_entity_id:
            raise ValueError("scheduled wave item differs from assured interception plan")
        execution = start_interception_execution(plan)
        executions.append(
            DefenseWaveComponentExecution(
                component_id=item.component_id,
                scheduling_subject_id=item.scheduling_subject_id,
                schedule_item=item,
                interception_plan=plan,
                interception_plan_sha256=_interception_plan_sha256(plan),
                interception_execution=execution,
            )
        )

    wave_id_seed = "|".join(
        [
            schedule.schedule_id,
            schedule.schedule_sha256,
            ",".join(sorted(row.component_id for row in executions)),
        ]
    )
    wave_id = "wave:" + hashlib.sha256(wave_id_seed.encode("utf-8")).hexdigest()[:24]
    draft = DefenseWaveExecution(
        wave_id=wave_id,
        graph_id=schedule.graph_id,
        schedule_id=schedule.schedule_id,
        schedule_sha256=schedule.schedule_sha256,
        schedule_context_id=schedule.scheduling_context.context_id,
        components=executions,
        complete=False,
        succeeded_component_ids=[],
        failed_component_ids=[],
        wave_sha256="0" * 64,
        rationale=[
            "components in a defense wave may progress independently and concurrently",
            "each component receives only the single step selected by the resource scheduler",
            "no component may advance to a second interception step inside the same wave",
            "every scheduled step still requires precondition evidence, execution receipt, and verified outcome",
            "after the wave, Sentinel must re-read telemetry and re-plan before any next component step",
        ],
    )
    digest = defense_wave_execution_sha256(draft)
    return DefenseWaveExecution.model_validate(
        draft.model_copy(update={"wave_sha256": digest}).model_dump()
    )


def _component(wave: DefenseWaveExecution, component_id: str) -> DefenseWaveComponentExecution:
    row = next((item for item in wave.components if item.component_id == component_id), None)
    if row is None:
        raise ValueError("component is not part of the scheduled defense wave")
    return row


def _replace_component(
    wave: DefenseWaveExecution,
    updated_component: DefenseWaveComponentExecution,
) -> DefenseWaveExecution:
    rows = [
        updated_component if row.component_id == updated_component.component_id else row
        for row in wave.components
    ]
    draft = wave.model_copy(update={"components": rows, "wave_sha256": "0" * 64})
    return _finalize_wave(draft)


def authorize_wave_component(
    wave: DefenseWaveExecution,
    *,
    component_id: str,
    precondition_evidence_ids: list[str],
) -> DefenseWaveExecution:
    verify_defense_wave_execution(wave)
    row = _component(wave, component_id)
    if row.status is not DefenseWaveComponentStatus.PENDING_AUTHORIZATION:
        raise ValueError("wave component is not pending authorization")
    execution = authorize_next_step(
        row.interception_execution,
        row.interception_plan,
        precondition_evidence_ids=precondition_evidence_ids,
    )
    updated = row.model_copy(
        update={
            "interception_execution": execution,
            "status": DefenseWaveComponentStatus.AUTHORIZED,
        }
    )
    return _replace_component(wave, DefenseWaveComponentExecution.model_validate(updated.model_dump()))


def record_wave_component_execution(
    wave: DefenseWaveExecution,
    *,
    component_id: str,
    execution_receipt_ids: list[str],
) -> DefenseWaveExecution:
    verify_defense_wave_execution(wave)
    row = _component(wave, component_id)
    if row.status is not DefenseWaveComponentStatus.AUTHORIZED:
        raise ValueError("wave component must be AUTHORIZED before execution is recorded")
    execution = record_step_execution(
        row.interception_execution,
        row.interception_plan,
        step_id=row.schedule_item.step_id,
        execution_receipt_ids=execution_receipt_ids,
    )
    updated = row.model_copy(
        update={
            "interception_execution": execution,
            "status": DefenseWaveComponentStatus.EXECUTED,
        }
    )
    return _replace_component(wave, DefenseWaveComponentExecution.model_validate(updated.model_dump()))


def verify_wave_component_outcome(
    wave: DefenseWaveExecution,
    *,
    component_id: str,
    succeeded: bool,
    outcome_evidence_ids: list[str],
) -> DefenseWaveExecution:
    verify_defense_wave_execution(wave)
    row = _component(wave, component_id)
    if row.status is not DefenseWaveComponentStatus.EXECUTED:
        raise ValueError("wave component must be EXECUTED before outcome verification")
    execution = verify_step_outcome(
        row.interception_execution,
        row.interception_plan,
        step_id=row.schedule_item.step_id,
        succeeded=succeeded,
        outcome_evidence_ids=outcome_evidence_ids,
    )
    updated = row.model_copy(
        update={
            "interception_execution": execution,
            "status": (
                DefenseWaveComponentStatus.VERIFIED_SUCCEEDED
                if succeeded
                else DefenseWaveComponentStatus.VERIFIED_FAILED
            ),
        }
    )
    return _replace_component(wave, DefenseWaveComponentExecution.model_validate(updated.model_dump()))
