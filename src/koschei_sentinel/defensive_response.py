from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


class DefensiveAction(str, Enum):
    REVOKE_SESSION = "revoke_session"
    DISABLE_CREDENTIAL = "disable_credential"
    ISOLATE_WORKLOAD = "isolate_workload"
    QUARANTINE_ARTIFACT = "quarantine_artifact"
    NARROW_CAPABILITY = "narrow_capability"
    BLOCK_INGRESS = "block_ingress"
    PAUSE_SERVICE = "pause_service"


@dataclass(frozen=True)
class DefensiveActionRequest:
    action_id: str
    tenant_id: str
    action: DefensiveAction
    target_id: str
    finding_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    reason: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.finding_ids or not self.evidence_ids:
            raise ValueError("defensive action must be finding- and evidence-bound")
        if not self.target_id.strip():
            raise ValueError("defensive action requires a target")


@dataclass(frozen=True)
class DefensivePolicy:
    tenant_id: str
    allowed_actions: frozenset[DefensiveAction]
    allowed_targets: frozenset[str]
    require_human_approval: frozenset[DefensiveAction] = frozenset()


@dataclass(frozen=True)
class DefensiveDecision:
    allowed: bool
    requires_approval: bool
    reason: str


class DefensiveExecutor(Protocol):
    def execute(self, request: DefensiveActionRequest) -> Mapping[str, Any]: ...


class DefensivePolicyGate:
    """Authorizes containment only inside the tenant-owned defensive boundary."""

    def evaluate(self, request: DefensiveActionRequest, policy: DefensivePolicy) -> DefensiveDecision:
        if request.tenant_id != policy.tenant_id:
            return DefensiveDecision(False, False, "tenant mismatch")
        if request.action not in policy.allowed_actions:
            return DefensiveDecision(False, False, "action is not allowed")
        if request.target_id not in policy.allowed_targets:
            return DefensiveDecision(False, False, "target is outside defensive boundary")
        approval = request.action in policy.require_human_approval
        return DefensiveDecision(True, approval, "approved by defensive policy")


class DefensiveResponsePlane:
    def __init__(self, *, executor: DefensiveExecutor, gate: DefensivePolicyGate | None = None) -> None:
        self._executor = executor
        self._gate = gate or DefensivePolicyGate()

    def contain(
        self,
        request: DefensiveActionRequest,
        policy: DefensivePolicy,
        *,
        human_approved: bool = False,
    ) -> Mapping[str, Any]:
        decision = self._gate.evaluate(request, policy)
        if not decision.allowed:
            raise PermissionError(decision.reason)
        if decision.requires_approval and not human_approved:
            raise PermissionError("human approval required")
        result = self._executor.execute(request)
        if not result.get("verified", False):
            raise RuntimeError("defensive action execution was not verified")
        return result
