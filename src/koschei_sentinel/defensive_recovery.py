from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol

from .defensive_response import DefensiveActionRequest


class RecoveryExecutor(Protocol):
    def rollback(self, request: DefensiveActionRequest, execution: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class DefensiveActionReceipt:
    action_id: str
    tenant_id: str
    target_id: str
    action: str
    execution_verified: bool
    effect_verified: bool
    rolled_back: bool
    rollback_verified: bool
    finding_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    issued_at: str
    schema_version: str = "sentinel.defensive-receipt.v1"

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()


class DefensiveRecoveryCoordinator:
    """Verifies defensive effect and fails closed into rollback when required."""

    def __init__(self, *, recovery_executor: RecoveryExecutor) -> None:
        self._recovery = recovery_executor

    def finalize(
        self,
        *,
        request: DefensiveActionRequest,
        execution: Mapping[str, Any],
        effect_verified: bool,
        rollback_on_unverified_effect: bool = True,
        issued_at: datetime | None = None,
    ) -> DefensiveActionReceipt:
        execution_verified = bool(execution.get("verified", False))
        if not execution_verified:
            raise RuntimeError("cannot finalize an unverified defensive execution")

        rolled_back = False
        rollback_verified = False
        if not effect_verified and rollback_on_unverified_effect:
            rollback = self._recovery.rollback(request, execution)
            rolled_back = True
            rollback_verified = bool(rollback.get("verified", False))
            if not rollback_verified:
                raise RuntimeError("defensive rollback could not be verified")

        when = (issued_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        return DefensiveActionReceipt(
            action_id=request.action_id,
            tenant_id=request.tenant_id,
            target_id=request.target_id,
            action=request.action.value,
            execution_verified=execution_verified,
            effect_verified=effect_verified,
            rolled_back=rolled_back,
            rollback_verified=rollback_verified,
            finding_ids=request.finding_ids,
            evidence_ids=request.evidence_ids,
            issued_at=when,
        )
