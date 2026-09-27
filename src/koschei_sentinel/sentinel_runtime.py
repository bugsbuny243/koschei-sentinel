from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from .audit_evidence import AuditEvidence, build_authorization_audit
from .authority_policy import AuthorityPolicy, DelegationChain, PolicyDecision
from .inference_contract import InferenceRequest, InferenceResponse, ReasoningEngine, ResponseValidator


@dataclass(frozen=True)
class SentinelResult:
    request_id: str
    policy_decision: PolicyDecision
    inference: InferenceResponse
    audit: AuditEvidence


class SentinelRuntime:
    """Canonical V1 path: authority -> policy -> reasoning -> validation -> audit."""

    def __init__(
        self,
        *,
        engine: ReasoningEngine,
        validators: Sequence[ResponseValidator],
        authority_policy: AuthorityPolicy | None = None,
    ) -> None:
        self._engine = engine
        self._validators = tuple(validators)
        self._authority_policy = authority_policy or AuthorityPolicy()

    def analyze(
        self,
        *,
        request: InferenceRequest,
        delegation_chain: DelegationChain,
        required_scopes: Sequence[str],
        audit_id: str,
        audience: str | None = None,
        now: datetime | None = None,
    ) -> SentinelResult:
        if request.authority is None:
            raise ValueError("Sentinel runtime requires an authority envelope")

        decision = self._authority_policy.evaluate(
            request.authority,
            delegation_chain,
            required_scopes=required_scopes,
            audience=audience,
            now=now,
        )
        if not decision.allowed:
            raise PermissionError(decision.reason)

        response = self._engine.infer(request)
        if response.request_id != request.request_id:
            raise RuntimeError("reasoning engine returned a mismatched request_id")
        if response.engine_id != self._engine.engine_id:
            raise RuntimeError("reasoning engine identity mismatch")
        if response.engine_version != self._engine.engine_version:
            raise RuntimeError("reasoning engine version mismatch")

        for validator in self._validators:
            validator.validate(request, response)

        audit = build_authorization_audit(
            audit_id=audit_id,
            request=request,
            chain=delegation_chain,
            decision=decision,
            required_scopes=tuple(required_scopes),
            issued_at=now,
        )
        return SentinelResult(
            request_id=request.request_id,
            policy_decision=decision,
            inference=response,
            audit=audit,
        )
