import pytest

from koschei_sentinel.defensive_response import (
    DefensiveAction,
    DefensiveActionRequest,
    DefensivePolicy,
    DefensiveResponsePlane,
)


class FixtureExecutor:
    def __init__(self, verified=True):
        self.verified = verified
        self.calls = []

    def execute(self, request):
        self.calls.append(request)
        return {"verified": self.verified, "action_id": request.action_id}


def req(action=DefensiveAction.ISOLATE_WORKLOAD, target="workload-1", tenant="t1"):
    return DefensiveActionRequest(
        "a1", tenant, action, target, ("f1",), ("e1",), "contain verified threat"
    )


def policy(*, approval=frozenset()):
    return DefensivePolicy(
        "t1",
        frozenset({DefensiveAction.ISOLATE_WORKLOAD, DefensiveAction.REVOKE_SESSION}),
        frozenset({"workload-1", "session-1"}),
        approval,
    )


def test_allowed_containment_executes_and_requires_verification():
    executor = FixtureExecutor()
    result = DefensiveResponsePlane(executor=executor).contain(req(), policy())
    assert result["verified"] is True
    assert len(executor.calls) == 1


def test_target_outside_boundary_never_executes():
    executor = FixtureExecutor()
    with pytest.raises(PermissionError, match="outside defensive boundary"):
        DefensiveResponsePlane(executor=executor).contain(req(target="external-host"), policy())
    assert executor.calls == []


def test_high_impact_action_can_require_human_approval():
    executor = FixtureExecutor()
    p = policy(approval=frozenset({DefensiveAction.ISOLATE_WORKLOAD}))
    with pytest.raises(PermissionError, match="human approval"):
        DefensiveResponsePlane(executor=executor).contain(req(), p)
    assert executor.calls == []


def test_unverified_execution_fails_closed():
    executor = FixtureExecutor(verified=False)
    with pytest.raises(RuntimeError, match="not verified"):
        DefensiveResponsePlane(executor=executor).contain(req(), policy())
