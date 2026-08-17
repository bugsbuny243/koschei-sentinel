import pytest

from koschei_sentinel.defense_authority import DefenseActionType, DefenseMode
from koschei_sentinel.defense_connector_contract import (
    ProtectedScope,
    build_connector_command,
)


def _scope() -> ProtectedScope:
    return ProtectedScope(
        scope_id="scope:prod",
        entity_ids=["device:prod", "credential:prod", "wallet:treasury"],
        permitted_actions=[
            DefenseActionType.ISOLATE_ENDPOINT,
            DefenseActionType.REVOKE_CREDENTIAL,
            DefenseActionType.FREEZE_SIGNER,
        ],
    )


def test_scope_rejects_target_outside_protected_environment():
    with pytest.raises(ValueError, match="outside the authorized protected scope"):
        build_connector_command(
            scope=_scope(),
            graph_id="incident:1",
            interception_step_id="intercept:1",
            defense_mode=DefenseMode.COMBAT,
            action=DefenseActionType.ISOLATE_ENDPOINT,
            target_entity_id="device:external",
            precondition_evidence_ids=["evidence:1"],
            dry_run=False,
        )


def test_scope_rejects_action_not_explicitly_permitted():
    with pytest.raises(ValueError, match="not permitted by the protected scope"):
        build_connector_command(
            scope=_scope(),
            graph_id="incident:1",
            interception_step_id="intercept:1",
            defense_mode=DefenseMode.COMBAT,
            action=DefenseActionType.KILL_PROCESS,
            target_entity_id="device:prod",
            precondition_evidence_ids=["evidence:1"],
            dry_run=False,
        )


def test_same_defense_intent_has_stable_idempotency_key():
    kwargs = dict(
        scope=_scope(),
        graph_id="incident:1",
        interception_step_id="intercept:1",
        defense_mode=DefenseMode.COMBAT,
        action=DefenseActionType.ISOLATE_ENDPOINT,
        target_entity_id="device:prod",
        precondition_evidence_ids=["evidence:1", "evidence:1"],
        dry_run=True,
    )
    first = build_connector_command(**kwargs)
    second = build_connector_command(**kwargs)

    assert first.idempotency_key == second.idempotency_key
    assert first.command_id == second.command_id
    assert first.precondition_evidence_ids == ["evidence:1"]
