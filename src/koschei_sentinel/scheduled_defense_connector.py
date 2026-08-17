from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.assured_defense_connector import (
    AssuredDefenseConnectorEnvelope,
    build_assured_connector_envelope,
)
from koschei_sentinel.assured_multi_incident_defense import AssuredMultiIncidentDefensePlan
from koschei_sentinel.defense_connector_contract import ProtectedScope
from koschei_sentinel.defense_resource_scheduler import (
    DefenseResourceSchedule,
    DefenseScheduleItem,
    SchedulingDisposition,
    verify_defense_resource_schedule,
)
from koschei_sentinel.interception_planner import build_interception_plan
from koschei_sentinel.models import StrictModel

_DIGEST = r"^[a-f0-9]{64}$"


class ScheduledAssuredConnectorEnvelope(StrictModel):
    schema_version: Literal["sentinel.scheduled-assured-connector-envelope.v1"] = (
        "sentinel.scheduled-assured-connector-envelope.v1"
    )
    schedule_id: str
    schedule_sha256: str = Field(pattern=_DIGEST)
    component_id: str
    schedule_item: DefenseScheduleItem
    assured_connector: AssuredDefenseConnectorEnvelope
    envelope_sha256: str = Field(pattern=_DIGEST)
    production_authorized: Literal[True] = True

    @model_validator(mode="after")
    def bindings_are_coherent(self) -> "ScheduledAssuredConnectorEnvelope":
        command = self.assured_connector.command
        if self.schedule_item.disposition is not SchedulingDisposition.SCHEDULED:
            raise ValueError("scheduled connector envelope requires a SCHEDULED resource item")
        if self.component_id != self.schedule_item.component_id:
            raise ValueError("scheduled connector component_id does not match schedule item")
        if command.graph_id != self.assured_connector.assurance_receipt.graph_id:
            raise ValueError("scheduled connector command graph is inconsistent")
        if command.interception_step_id != self.schedule_item.step_id:
            raise ValueError("scheduled connector step differs from schedule item")
        if command.action is not self.schedule_item.action:
            raise ValueError("scheduled connector action differs from schedule item")
        if command.target_entity_id != self.schedule_item.target_entity_id:
            raise ValueError("scheduled connector target differs from schedule item")
        if command.defense_mode is not self.schedule_item.defense_mode:
            raise ValueError("scheduled connector mode differs from schedule item")
        expected = scheduled_connector_envelope_sha256(
            schedule_id=self.schedule_id,
            schedule_sha256=self.schedule_sha256,
            component_id=self.component_id,
            schedule_item=self.schedule_item,
            assured_connector=self.assured_connector,
        )
        if expected != self.envelope_sha256:
            raise ValueError("scheduled connector envelope digest mismatch")
        return self


def scheduled_connector_envelope_sha256(
    *,
    schedule_id: str,
    schedule_sha256: str,
    component_id: str,
    schedule_item: DefenseScheduleItem,
    assured_connector: AssuredDefenseConnectorEnvelope,
) -> str:
    payload = {
        "schedule_id": schedule_id,
        "schedule_sha256": schedule_sha256,
        "component_id": component_id,
        "schedule_item": schedule_item.model_dump(mode="json"),
        "assured_connector": assured_connector.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_scheduled_assured_connector_envelope(
    *,
    schedule: DefenseResourceSchedule,
    multi_plan: AssuredMultiIncidentDefensePlan,
    component_id: str,
    scope: ProtectedScope,
    precondition_evidence_ids: list[str],
    dry_run: bool = True,
) -> ScheduledAssuredConnectorEnvelope:
    verify_defense_resource_schedule(schedule, multi_plan)

    item = next(
        (row for row in schedule.scheduled if row.component_id == component_id),
        None,
    )
    if item is None:
        if any(row.component_id == component_id for row in schedule.deferred):
            raise ValueError("deferred defense component cannot produce a connector command")
        raise ValueError("component is not present in the scheduled defense wave")

    component = next(
        (row for row in multi_plan.component_plans if row.component_id == component_id),
        None,
    )
    if component is None:
        raise ValueError("scheduled component is absent from assured multi-incident plan")
    if item.target_entity_id not in set(component.entity_ids):
        raise ValueError("scheduled connector target crossed component boundary")

    active = component.assured_plan.active_defense_plan
    interception = build_interception_plan(active)
    step = next((row for row in interception.steps if row.step_id == item.step_id), None)
    if step is None:
        raise ValueError("scheduled interception step is absent from assured component plan")
    if step.action is not item.action or step.target_entity_id != item.target_entity_id:
        raise ValueError("scheduled item action/target differs from assured interception step")
    if sorted(step.supporting_relation_ids) != sorted(item.supporting_relation_ids):
        raise ValueError("scheduled item supporting relations differ from interception step")
    if component.assured_plan.assurance.effective_mode is not item.defense_mode:
        raise ValueError("scheduled item defense mode differs from component assurance")

    assured_connector = build_assured_connector_envelope(
        scope=scope,
        assured_plan=component.assured_plan,
        interception_plan=interception,
        step_id=step.step_id,
        precondition_evidence_ids=precondition_evidence_ids,
        dry_run=dry_run,
    )
    digest = scheduled_connector_envelope_sha256(
        schedule_id=schedule.schedule_id,
        schedule_sha256=schedule.schedule_sha256,
        component_id=component_id,
        schedule_item=item,
        assured_connector=assured_connector,
    )
    return ScheduledAssuredConnectorEnvelope(
        schedule_id=schedule.schedule_id,
        schedule_sha256=schedule.schedule_sha256,
        component_id=component_id,
        schedule_item=item,
        assured_connector=assured_connector,
        envelope_sha256=digest,
    )
