from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from .authority_policy import DelegationChain, PolicyDecision
from .inference_contract import InferenceRequest


@dataclass(frozen=True)
class AuditEvidence:
    audit_id: str
    request_id: str
    tenant_id: str
    principal_id: str
    controller_id: str | None
    delegate_id: str | None
    required_scopes: tuple[str, ...]
    effective_scopes: tuple[str, ...]
    allowed: bool
    reason: str
    chain_depth: int
    delegation_digest: str
    evidence_digests: tuple[str, ...]
    issued_at: str
    schema_version: str = "sentinel.audit.v1"

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.canonical_bytes()).hexdigest()


def _canonical_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def build_authorization_audit(
    *,
    audit_id: str,
    request: InferenceRequest,
    chain: DelegationChain,
    decision: PolicyDecision,
    required_scopes: tuple[str, ...],
    issued_at: datetime | None = None,
) -> AuditEvidence:
    if request.authority is None:
        raise ValueError("authorization audit requires an authority envelope")

    issued_at = issued_at or datetime.now(timezone.utc)
    authority = request.authority

    chain_payload = {
        "grants": [
            {
                "grant_id": grant.grant_id,
                "issuer_id": grant.issuer_id,
                "subject_id": grant.subject_id,
                "scopes": list(grant.scopes),
                "audience": list(grant.audience),
                "not_before": grant.not_before,
                "expires_at": grant.expires_at,
                "constraints": dict(grant.constraints),
                "no_further_delegation": grant.no_further_delegation,
            }
            for grant in chain.grants
        ]
    }

    return AuditEvidence(
        audit_id=audit_id,
        request_id=request.request_id,
        tenant_id=request.tenant_id,
        principal_id=authority.principal_id,
        controller_id=authority.controller_id,
        delegate_id=authority.delegate_id,
        required_scopes=tuple(sorted(set(required_scopes))),
        effective_scopes=decision.effective_scopes,
        allowed=decision.allowed,
        reason=decision.reason,
        chain_depth=decision.chain_depth,
        delegation_digest=_canonical_digest(chain_payload),
        evidence_digests=tuple(sorted(item.digest for item in request.evidence)),
        issued_at=issued_at.astimezone(timezone.utc).isoformat(),
    )
