from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from .inference_contract import EvidenceRef, InferenceRequest, RiskClass


class SecurityDomain(str, Enum):
    CODE = "code"
    APPLICATION = "application"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    LOG = "log"
    NETWORK = "network"
    CLOUD = "cloud"
    IDENTITY = "identity"
    AGENT = "agent"
    WEB3 = "web3"
    INCIDENT = "incident"


class AnalysisMode(str, Enum):
    ASSESS = "assess"
    CORRELATE = "correlate"
    INVESTIGATE = "investigate"
    VERIFY = "verify"


@dataclass(frozen=True)
class SecurityArtifact:
    artifact_id: str
    domain: SecurityDomain
    media_type: str
    content: str
    provenance: str
    digest: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SecurityCase:
    case_id: str
    tenant_id: str
    mode: AnalysisMode
    artifacts: tuple[SecurityArtifact, ...]
    objective: str
    constraints: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifacts:
            raise ValueError("security case requires at least one artifact")
        ids = [item.artifact_id for item in self.artifacts]
        if len(ids) != len(set(ids)):
            raise ValueError("security artifact ids must be unique")


@dataclass(frozen=True)
class SecurityFinding:
    finding_id: str
    title: str
    severity: str
    confidence: float
    evidence_ids: tuple[str, ...]
    defensive_action: str
    verification_state: str

    def __post_init__(self) -> None:
        if self.severity not in {"info", "low", "medium", "high", "critical"}:
            raise ValueError("invalid finding severity")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not self.evidence_ids:
            raise ValueError("security finding must be evidence-bound")


def build_security_inference_request(
    case: SecurityCase,
    *,
    authority,
    risk_class: RiskClass,
) -> InferenceRequest:
    evidence = tuple(
        EvidenceRef(
            evidence_id=item.artifact_id,
            digest=item.digest,
            kind=item.domain.value,
            source=item.provenance,
        )
        for item in case.artifacts
    )
    artifact_manifest = "\n".join(
        f"[{item.artifact_id}] domain={item.domain.value} media={item.media_type}"
        for item in case.artifacts
    )
    prompt = (
        "Perform defensive cybersecurity analysis. Bind every material finding "
        "to supplied evidence. Do not treat unverified model output as evidence.\n"
        f"mode={case.mode.value}\nobjective={case.objective}\nartifacts:\n{artifact_manifest}"
    )
    return InferenceRequest(
        request_id=case.case_id,
        tenant_id=case.tenant_id,
        task_class=f"cybersecurity:{case.mode.value}",
        risk_class=risk_class,
        prompt=prompt,
        evidence=evidence,
        authority=authority,
        metadata={
            "security_domains": tuple(sorted({a.domain.value for a in case.artifacts})),
            "artifact_count": len(case.artifacts),
        },
    )


def validate_security_findings(
    findings: Sequence[SecurityFinding],
    case: SecurityCase,
) -> None:
    known = {item.artifact_id for item in case.artifacts}
    seen: set[str] = set()
    for finding in findings:
        if finding.finding_id in seen:
            raise ValueError("duplicate security finding id")
        seen.add(finding.finding_id)
        unknown = set(finding.evidence_ids) - known
        if unknown:
            raise ValueError(f"finding references unknown evidence: {sorted(unknown)!r}")
