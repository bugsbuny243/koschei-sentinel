from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Mapping, Sequence

from .authority_policy import DelegationChain
from .cybersecurity_core import SecurityCase, SecurityFinding, build_security_inference_request, validate_security_findings
from .defensive_recovery import DefensiveActionReceipt, DefensiveRecoveryCoordinator
from .defensive_response import DefensiveActionRequest, DefensivePolicy, DefensiveResponsePlane
from .inference_contract import AuthorityEnvelope, RiskClass
from .sentinel_runtime import SentinelResult, SentinelRuntime


@dataclass(frozen=True)
class DefensiveSentinelResult:
    analysis: SentinelResult
    findings: tuple[SecurityFinding, ...]
    action_receipts: tuple[DefensiveActionReceipt, ...]


class DefensiveSentinelRuntime:
    """End-to-end defensive path. Reasoning proposes; policy authorizes; executors contain."""

    def __init__(
        self,
        *,
        analysis_runtime: SentinelRuntime,
        response_plane: DefensiveResponsePlane,
        recovery: DefensiveRecoveryCoordinator,
        finding_parser: Callable[[SentinelResult], Sequence[SecurityFinding]],
        action_planner: Callable[[Sequence[SecurityFinding]], Sequence[DefensiveActionRequest]],
        effect_verifier: Callable[[DefensiveActionRequest, Mapping[str, object]], bool],
    ) -> None:
        self._analysis = analysis_runtime
        self._response = response_plane
        self._recovery = recovery
        self._finding_parser = finding_parser
        self._action_planner = action_planner
        self._effect_verifier = effect_verifier

    def defend(
        self,
        *,
        case: SecurityCase,
        authority: AuthorityEnvelope,
        delegation_chain: DelegationChain,
        defensive_policy: DefensivePolicy,
        risk_class: RiskClass,
        required_scopes: Sequence[str],
        audit_id: str,
        now: datetime | None = None,
        human_approved_actions: frozenset[str] = frozenset(),
    ) -> DefensiveSentinelResult:
        request = build_security_inference_request(case, authority=authority, risk_class=risk_class)
        analysis = self._analysis.analyze(
            request=request,
            delegation_chain=delegation_chain,
            required_scopes=required_scopes,
            audit_id=audit_id,
            now=now,
        )
        findings = tuple(self._finding_parser(analysis))
        validate_security_findings(findings, case)

        receipts: list[DefensiveActionReceipt] = []
        for action in self._action_planner(findings):
            if action.tenant_id != case.tenant_id:
                raise PermissionError("planned defensive action crosses tenant boundary")
            known_findings = {finding.finding_id for finding in findings}
            if not set(action.finding_ids).issubset(known_findings):
                raise ValueError("defensive action references unknown finding")
            known_evidence = {artifact.artifact_id for artifact in case.artifacts}
            if not set(action.evidence_ids).issubset(known_evidence):
                raise ValueError("defensive action references unknown evidence")

            execution = self._response.contain(
                action,
                defensive_policy,
                human_approved=action.action_id in human_approved_actions,
            )
            effect_verified = self._effect_verifier(action, execution)
            receipts.append(
                self._recovery.finalize(
                    request=action,
                    execution=execution,
                    effect_verified=effect_verified,
                    issued_at=now,
                )
            )

        return DefensiveSentinelResult(analysis, findings, tuple(receipts))
