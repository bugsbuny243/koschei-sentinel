from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from koschei_sentinel.defense_consent import ConsentBoundDefense, DefenseConsent
from koschei_sentinel.defensive_response import (
    DefensiveAction,
    DefensiveActionRequest,
    DefensiveResponsePlane,
)


class RecordingExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: DefensiveActionRequest) -> dict[str, object]:
        self.calls += 1
        return {"verified": True, "action_id": request.action_id}


def _request(*, tenant: str = "team", target: str = "host-1") -> DefensiveActionRequest:
    return DefensiveActionRequest(
        action_id="action-1",
        tenant_id=tenant,
        action=DefensiveAction.BLOCK_INGRESS,
        target_id=target,
        finding_ids=("finding-1",),
        evidence_ids=("evidence-1",),
        reason="corroborated hostile ingress",
    )


def _consent(now: datetime, **overrides: object) -> DefenseConsent:
    values = {
        "tenant_id": "team",
        "allowed_targets": frozenset({"host-1"}),
        "allowed_actions": frozenset({DefensiveAction.BLOCK_INGRESS}),
        "expires_at": now + timedelta(minutes=10),
        "granted_by": "operator-1",
        "enabled": True,
    }
    values.update(overrides)
    return DefenseConsent(**values)


def test_valid_consent_allows_verified_action() -> None:
    now = datetime.now(UTC)
    executor = RecordingExecutor()
    defense = ConsentBoundDefense(DefensiveResponsePlane(executor=executor))
    result = defense.contain(_request(), _consent(now), now=now)
    assert result["verified"] is True
    assert executor.calls == 1


@pytest.mark.parametrize(
    ("request", "changes"),
    [
        (_request(tenant="other"), {}),
        (_request(target="host-2"), {}),
        (_request(), {"enabled": False}),
        (_request(), {"allowed_actions": frozenset({DefensiveAction.PAUSE_SERVICE})}),
    ],
)
def test_out_of_scope_consent_never_executes(
    request: DefensiveActionRequest, changes: dict[str, object]
) -> None:
    now = datetime.now(UTC)
    executor = RecordingExecutor()
    defense = ConsentBoundDefense(DefensiveResponsePlane(executor=executor))
    with pytest.raises(PermissionError):
        defense.contain(request, _consent(now, **changes), now=now)
    assert executor.calls == 0


def test_expired_consent_never_executes() -> None:
    now = datetime.now(UTC)
    executor = RecordingExecutor()
    defense = ConsentBoundDefense(DefensiveResponsePlane(executor=executor))
    with pytest.raises(PermissionError, match="expired"):
        defense.contain(
            _request(), _consent(now, expires_at=now - timedelta(seconds=1)), now=now
        )
    assert executor.calls == 0


def test_naive_expiry_rejected() -> None:
    with pytest.raises(ValueError, match="timezone"):
        _consent(datetime.now(UTC), expires_at=datetime.now())
