from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.defense_authority import DefenseActionType, DefenseMode
from koschei_sentinel.models import StrictModel


class ConnectorExecutionStatus(StrEnum):
    DRY_RUN = "DRY_RUN"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class ProtectedScope(StrictModel):
    schema_version: Literal["sentinel.protected-scope.v1"] = "sentinel.protected-scope.v1"
    scope_id: str = Field(min_length=3, max_length=256)
    entity_ids: list[str] = Field(min_length=1, max_length=100000)
    permitted_actions: list[DefenseActionType] = Field(min_length=1)

    @model_validator(mode="after")
    def scope_values_are_unique(self) -> ProtectedScope:
        if len(self.entity_ids) != len(set(self.entity_ids)):
            raise ValueError("protected scope entity_ids must be unique")
        if len(self.permitted_actions) != len(set(self.permitted_actions)):
            raise ValueError("protected scope permitted_actions must be unique")
        return self


class DefenseConnectorCommand(StrictModel):
    schema_version: Literal["sentinel.defense-connector-command.v1"] = (
        "sentinel.defense-connector-command.v1"
    )
    command_id: str = Field(min_length=3, max_length=256)
    idempotency_key: str = Field(min_length=16, max_length=128)
    graph_id: str = Field(min_length=3, max_length=256)
    interception_step_id: str = Field(min_length=3, max_length=256)
    defense_mode: DefenseMode
    action: DefenseActionType
    target_entity_id: str = Field(min_length=3, max_length=256)
    scope_id: str = Field(min_length=3, max_length=256)
    precondition_evidence_ids: list[str] = Field(min_length=1, max_length=256)
    dry_run: bool = True


class DefenseConnectorReceipt(StrictModel):
    schema_version: Literal["sentinel.defense-connector-receipt.v1"] = (
        "sentinel.defense-connector-receipt.v1"
    )
    command_id: str
    idempotency_key: str
    connector: str = Field(min_length=2, max_length=128)
    status: ConnectorExecutionStatus
    target_entity_id: str
    provider_receipt_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    message: str | None = None

    @model_validator(mode="after")
    def completed_receipt_requires_evidence(self) -> DefenseConnectorReceipt:
        if self.status in {
            ConnectorExecutionStatus.SUCCEEDED,
            ConnectorExecutionStatus.FAILED,
        } and not self.evidence_ids:
            raise ValueError("completed connector receipt requires evidence")
        return self


def command_idempotency_key(
    *,
    graph_id: str,
    interception_step_id: str,
    action: DefenseActionType,
    target_entity_id: str,
    scope_id: str,
) -> str:
    payload = "|".join(
        [graph_id, interception_step_id, action.value, target_entity_id, scope_id]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_connector_command(
    *,
    scope: ProtectedScope,
    graph_id: str,
    interception_step_id: str,
    defense_mode: DefenseMode,
    action: DefenseActionType,
    target_entity_id: str,
    precondition_evidence_ids: list[str],
    dry_run: bool = True,
) -> DefenseConnectorCommand:
    if target_entity_id not in set(scope.entity_ids):
        raise ValueError("defense target is outside the authorized protected scope")
    if action not in set(scope.permitted_actions):
        raise ValueError("defense action is not permitted by the protected scope")
    if not precondition_evidence_ids:
        raise ValueError("connector command requires precondition evidence")

    key = command_idempotency_key(
        graph_id=graph_id,
        interception_step_id=interception_step_id,
        action=action,
        target_entity_id=target_entity_id,
        scope_id=scope.scope_id,
    )
    return DefenseConnectorCommand(
        command_id=f"cmd:{key[:24]}",
        idempotency_key=key,
        graph_id=graph_id,
        interception_step_id=interception_step_id,
        defense_mode=defense_mode,
        action=action,
        target_entity_id=target_entity_id,
        scope_id=scope.scope_id,
        precondition_evidence_ids=list(dict.fromkeys(precondition_evidence_ids)),
        dry_run=dry_run,
    )
