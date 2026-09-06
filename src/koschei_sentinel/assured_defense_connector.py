from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.assured_active_defense import (
    ActiveDefenseAssurancePolicy,
    ActiveDefenseAssuranceReceipt,
    AssuredActiveDefensePlan,
)
from koschei_sentinel.defense_authority import DefenseMode
from koschei_sentinel.defense_connector_contract import (
    DefenseConnectorCommand,
    ProtectedScope,
    build_connector_command,
)
from koschei_sentinel.interception_planner import InterceptionPlan
from koschei_sentinel.models import StrictModel

_DIGEST = r"^[a-f0-9]{64}$"


def active_defense_assurance_sha256(receipt: ActiveDefenseAssuranceReceipt) -> str:
    payload = json.dumps(
        receipt.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def interception_plan_sha256(plan: InterceptionPlan) -> str:
    payload = json.dumps(
        plan.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AssuredDefenseConnectorEnvelope(StrictModel):
    schema_version: Literal["sentinel.assured-defense-connector-envelope.v1"] = (
        "sentinel.assured-defense-connector-envelope.v1"
    )
    command: DefenseConnectorCommand
    assurance_receipt: ActiveDefenseAssuranceReceipt
    assurance_policy: ActiveDefenseAssurancePolicy
    assurance_sha256: str = Field(pattern=_DIGEST)
    interception_plan: InterceptionPlan
    interception_plan_sha256: str = Field(pattern=_DIGEST)
    production_authorized: Literal[True] = True

    @model_validator(mode="after")
    def production_binding_is_valid(self) -> AssuredDefenseConnectorEnvelope:
        if active_defense_assurance_sha256(self.assurance_receipt) != self.assurance_sha256:
            raise ValueError("assured connector envelope assurance digest mismatch")
        if interception_plan_sha256(self.interception_plan) != self.interception_plan_sha256:
            raise ValueError("assured connector envelope interception plan digest mismatch")
        if self.command.graph_id != self.assurance_receipt.graph_id:
            raise ValueError("connector command graph differs from assurance receipt")
        if self.interception_plan.graph_id != self.command.graph_id:
            raise ValueError("connector command graph differs from interception plan")
        if self.command.defense_mode is not self.assurance_receipt.effective_mode:
            raise ValueError("connector command defense mode differs from assurance mode")
        if self.interception_plan.defense_mode is not self.command.defense_mode:
            raise ValueError("connector command defense mode differs from interception plan")

        step = next(
            (row for row in self.interception_plan.steps if row.step_id == self.command.interception_step_id),
            None,
        )
        if step is None:
            raise ValueError("connector command references a missing interception step")
        if step.action is not self.command.action or step.target_entity_id != self.command.target_entity_id:
            raise ValueError("connector command action/target differs from interception step")

        count = self.assurance_receipt.active_independent_domain_count
        if self.command.defense_mode is DefenseMode.SIEGE:
            if count < self.assurance_policy.min_independent_domains_siege:
                raise ValueError("Siege connector command lacks required independent evidence domains")
        elif self.command.defense_mode is DefenseMode.COMBAT:
            if count < self.assurance_policy.min_independent_domains_combat:
                raise ValueError("Combat connector command lacks required independent evidence domains")
        elif not self.command.dry_run:
            # Guard may still perform low-impact controls, but a real connector command must
            # be an actual Guard-authorized step in the interception plan, as checked above.
            pass
        return self


def build_assured_connector_envelope(
    *,
    scope: ProtectedScope,
    assured_plan: AssuredActiveDefensePlan,
    interception_plan: InterceptionPlan,
    step_id: str,
    precondition_evidence_ids: list[str],
    dry_run: bool = True,
) -> AssuredDefenseConnectorEnvelope:
    active = assured_plan.active_defense_plan
    if interception_plan.graph_id != active.graph_id:
        raise ValueError("interception plan graph does not match assured active-defense plan")
    if interception_plan.defense_mode is not active.decision.mode:
        raise ValueError("interception plan mode does not match assured active-defense plan")

    step = next((row for row in interception_plan.steps if row.step_id == step_id), None)
    if step is None:
        raise ValueError("requested interception step does not exist")
    authorized_pairs = {
        (row.action, row.entity_id) for row in active.authorized_cut_points
    }
    if (step.action, step.target_entity_id) not in authorized_pairs:
        raise ValueError("interception step is not an authorized defensive cut point")
    if step.action not in set(active.decision.permitted_actions):
        raise ValueError("interception step action is not permitted by assured defense mode")

    command = build_connector_command(
        scope=scope,
        graph_id=active.graph_id,
        interception_step_id=step.step_id,
        defense_mode=active.decision.mode,
        action=step.action,
        target_entity_id=step.target_entity_id,
        precondition_evidence_ids=precondition_evidence_ids,
        dry_run=dry_run,
    )
    return AssuredDefenseConnectorEnvelope(
        command=command,
        assurance_receipt=assured_plan.assurance,
        assurance_policy=assured_plan.policy,
        assurance_sha256=active_defense_assurance_sha256(assured_plan.assurance),
        interception_plan=interception_plan,
        interception_plan_sha256=interception_plan_sha256(interception_plan),
    )
