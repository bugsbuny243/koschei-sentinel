import pytest

from koschei_sentinel.cybersecurity_core import (
    AnalysisMode,
    SecurityArtifact,
    SecurityCase,
    SecurityDomain,
    SecurityFinding,
    build_security_inference_request,
    validate_security_findings,
)
from koschei_sentinel.inference_contract import AuthorityEnvelope, RiskClass


def artifact(aid="code-1"):
    return SecurityArtifact(
        artifact_id=aid,
        domain=SecurityDomain.CODE,
        media_type="text/plain",
        content="fixture",
        provenance="sentinel-lab",
        digest=f"sha256:{aid}",
    )


def case():
    return SecurityCase(
        case_id="case-001",
        tenant_id="tenant-1",
        mode=AnalysisMode.ASSESS,
        artifacts=(artifact(),),
        objective="identify defensive security findings",
    )


def test_security_case_becomes_evidence_bound_inference_request():
    request = build_security_inference_request(
        case(),
        authority=AuthorityEnvelope("operator", "controller", "sentinel", ("analyze",)),
        risk_class=RiskClass.HIGH,
    )
    assert request.task_class == "cybersecurity:assess"
    assert request.evidence[0].evidence_id == "code-1"
    assert request.metadata["security_domains"] == ("code",)


def test_finding_must_reference_known_case_evidence():
    finding = SecurityFinding(
        "f1", "unsafe configuration", "high", 0.9, ("missing",),
        "apply defensive configuration", "pending",
    )
    with pytest.raises(ValueError, match="unknown evidence"):
        validate_security_findings((finding,), case())


def test_valid_evidence_bound_finding_passes():
    finding = SecurityFinding(
        "f1", "review required", "medium", 0.8, ("code-1",),
        "review and harden the component", "verified",
    )
    validate_security_findings((finding,), case())
