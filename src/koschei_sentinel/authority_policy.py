from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .inference_contract import AuthorityEnvelope


class AuthorityPolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class CapabilityGrant:
    grant_id: str
    issuer_id: str
    subject_id: str
    scopes: tuple[str, ...]
    audience: tuple[str, ...] = ()
    not_before: datetime | None = None
    expires_at: datetime | None = None
    constraints: Mapping[str, Any] = field(default_factory=dict)
    no_further_delegation: bool = False


@dataclass(frozen=True)
class DelegationChain:
    grants: tuple[CapabilityGrant, ...]

    def __post_init__(self) -> None:
        if not self.grants:
            raise ValueError("delegation chain must contain at least one grant")


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    effective_scopes: tuple[str, ...]
    reason: str
    chain_depth: int


class AuthorityPolicy:
    """Sentinel-owned, protocol-neutral delegation enforcement."""

    def __init__(self, *, max_depth: int = 5) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        self.max_depth = max_depth

    def evaluate(
        self,
        authority: AuthorityEnvelope,
        chain: DelegationChain,
        *,
        required_scopes: Sequence[str],
        audience: str | None = None,
        now: datetime | None = None,
    ) -> PolicyDecision:
        now = now or datetime.now(timezone.utc)
        grants = chain.grants
        if len(grants) > self.max_depth:
            raise AuthorityPolicyError("delegation chain exceeds maximum depth")

        root = grants[0]
        if root.issuer_id != authority.principal_id:
            raise AuthorityPolicyError("root issuer does not match authority principal")

        parent_scopes = set(root.scopes)
        self._validate_time(root, now)
        self._validate_audience(root, audience)

        for index, child in enumerate(grants[1:], start=1):
            parent = grants[index - 1]
            if child.issuer_id != parent.subject_id:
                raise AuthorityPolicyError("delegation chain linkage is invalid")
            if parent.no_further_delegation:
                raise AuthorityPolicyError("parent grant forbids further delegation")

            self._validate_time(child, now)
            self._validate_audience(child, audience)

            child_scopes = set(child.scopes)
            if not child_scopes.issubset(parent_scopes):
                raise AuthorityPolicyError("delegated authority widened")
            parent_scopes = child_scopes

        leaf = grants[-1]
        expected_delegate = authority.delegate_id or authority.principal_id
        if leaf.subject_id != expected_delegate:
            raise AuthorityPolicyError("leaf subject does not match active delegate")

        envelope_scopes = set(authority.scopes)
        effective = parent_scopes & envelope_scopes if envelope_scopes else parent_scopes
        required = set(required_scopes)
        allowed = required.issubset(effective)
        return PolicyDecision(
            allowed=allowed,
            effective_scopes=tuple(sorted(effective)),
            reason="authorized" if allowed else "required scope is not delegated",
            chain_depth=len(grants),
        )

    @staticmethod
    def _validate_time(grant: CapabilityGrant, now: datetime) -> None:
        if grant.not_before is not None and now < grant.not_before:
            raise AuthorityPolicyError("delegation grant is not active yet")
        if grant.expires_at is not None and now >= grant.expires_at:
            raise AuthorityPolicyError("delegation grant has expired")

    @staticmethod
    def _validate_audience(grant: CapabilityGrant, audience: str | None) -> None:
        if audience is not None and grant.audience and audience not in grant.audience:
            raise AuthorityPolicyError("delegation audience mismatch")
