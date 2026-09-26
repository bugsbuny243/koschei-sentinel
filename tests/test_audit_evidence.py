from datetime import datetime, timezone

from koschei_sentinel.audit_evidence import build_authorization_audit
from koschei_sentinel.authority_policy import CapabilityGrant, DelegationChain, PolicyDecision
from koschei_sentinel.inference_contract import (
    AuthorityEnvelope,
    EvidenceRef,
    InferenceRequest,
    RiskClass,
)


NOW = datetime(2026, 9, 26, tzinfo=timezone.utc)


def test_authorization_audit_is_deterministic_and_evidence_bound():
    request = InferenceRequest(
        request_id="r1",
        tenant_id="t1",
        task_class="security-analysis",
        risk_class=RiskClass.HIGH,
        prompt="analyze",
        evidence=(EvidenceRef("e1", "sha256:evidence", "observation", "sentinel"),),
        authority=AuthorityEnvelope(
            principal_id="user-a",
            controller_id="controller-a",
            delegate_id="agent-b",
            scopes=("read", "analyze"),
        ),
    )
    chain = DelegationChain((
        CapabilityGrant("g1", "user-a", "agent-a", ("read", "analyze")),
        CapabilityGrant("g2", "agent-a", "agent-b", ("analyze",)),
    ))
    decision = PolicyDecision(True, ("analyze",), "authorized", 2)

    first = build_authorization_audit(
        audit_id="a1",
        request=request,
        chain=chain,
        decision=decision,
        required_scopes=("analyze",),
        issued_at=NOW,
    )
    second = build_authorization_audit(
        audit_id="a1",
        request=request,
        chain=chain,
        decision=decision,
        required_scopes=("analyze",),
        issued_at=NOW,
    )

    assert first == second
    assert first.digest() == second.digest()
    assert first.evidence_digests == ("sha256:evidence",)
    assert first.delegation_digest.startswith("sha256:")
    assert first.allowed is True


def test_authorization_audit_changes_when_delegation_changes():
    request = InferenceRequest(
        request_id="r1",
        tenant_id="t1",
        task_class="security-analysis",
        risk_class=RiskClass.HIGH,
        prompt="analyze",
        authority=AuthorityEnvelope("user-a", "controller-a", "agent-b", ("read",)),
    )
    decision = PolicyDecision(True, ("read",), "authorized", 1)

    one = build_authorization_audit(
        audit_id="a1",
        request=request,
        chain=DelegationChain((CapabilityGrant("g1", "user-a", "agent-b", ("read",)),)),
        decision=decision,
        required_scopes=("read",),
        issued_at=NOW,
    )
    two = build_authorization_audit(
        audit_id="a1",
        request=request,
        chain=DelegationChain((CapabilityGrant("g2", "user-a", "agent-b", ("read",)),)),
        decision=decision,
        required_scopes=("read",),
        issued_at=NOW,
    )

    assert one.delegation_digest != two.delegation_digest
    assert one.digest() != two.digest()
