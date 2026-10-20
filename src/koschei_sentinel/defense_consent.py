"""Explicit, expiring consent for bounded defensive containment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from .defensive_response import (
    DefensiveAction,
    DefensiveActionRequest,
    DefensivePolicy,
    DefensiveResponsePlane,
)


@dataclass(frozen=True)
class DefenseConsent:
    """An operator-granted capability scoped to a tenant, target, action and expiry."""

    tenant_id: str
    allowed_targets: frozenset[str]
    allowed_actions: frozenset[DefensiveAction]
    expires_at: datetime
    granted_by: str
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.tenant_id or not self.granted_by:
            raise ValueError("consent requires tenant and granting operator")
        if not self.allowed_targets or not self.allowed_actions:
            raise ValueError("consent must specify targets and actions")
        if self.expires_at.tzinfo is None or self.expires_at.utcoffset() is None:
            raise ValueError("consent expiry must include a timezone")

    def authorize(self, request: DefensiveActionRequest, *, now: datetime) -> DefensivePolicy:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("authorization time must include a timezone")
        if not self.enabled:
            raise PermissionError("automatic defense disabled by operator")
        if now.astimezone(UTC) >= self.expires_at.astimezone(UTC):
            raise PermissionError("defensive consent expired")
        if request.tenant_id != self.tenant_id:
            raise PermissionError("defensive consent tenant mismatch")
        if request.target_id not in self.allowed_targets:
            raise PermissionError("target is outside granted consent")
        if request.action not in self.allowed_actions:
            raise PermissionError("action is outside granted consent")
        return DefensivePolicy(
            tenant_id=self.tenant_id,
            allowed_targets=self.allowed_targets,
            allowed_actions=self.allowed_actions,
        )


class ConsentBoundDefense:
    """Require current operator consent before invoking the existing policy gate."""

    def __init__(self, response_plane: DefensiveResponsePlane) -> None:
        self._response_plane = response_plane

    def contain(
        self,
        request: DefensiveActionRequest,
        consent: DefenseConsent,
        *,
        now: datetime,
    ) -> dict[str, object]:
        policy = consent.authorize(request, now=now)
        return dict(self._response_plane.contain(request, policy))
