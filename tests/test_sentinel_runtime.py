from datetime import UTC, datetime

import pytest

from koschei_sentinel.authority_policy import (
    CapabilityGrant,
    DelegationChain,
)
from koschei_sentinel.inference_contract import (
    AuthorityEnvelope,
    EvidenceBinding,
    EvidenceRef,
    InferenceRequest,
    InferenceResponse,
    RiskClass,
)
from koschei_sentinel.inference_validation import (
    FailClosedResponseValidator,
)
from koschei_sentinel.sentinel_runtime import SentinelRuntime

NOW = datetime(2026, 9, 27, tzinfo=UTC)


class FixtureEngine:
    engine_id = "fixture"
    engine_version = "1"

    def infer(self, request):
        return InferenceResponse(
            request_id=request.request_id,
            engine_id=self.engine_id,
            engine_version=self.engine_version,
            output={
                "severity": "high",
                "finding": "delegated security analysis completed",
                "policy": "allow",
            },
            evidence_bindings=(
                EvidenceBinding("finding-1", ("e1",), "bound to fixture evidence"),
            ),
        )


def make_request():
    return InferenceRequest(
        request_id="fixture-001",
        tenant_id="tenant-1",
        task_class="security-analysis",
        risk_class=RiskClass.HIGH,
        prompt="analyze fixture",
        evidence=(EvidenceRef("e1", "sha256:fixture", "fixture", "sentinel-lab"),),
        authority=AuthorityEnvelope(
            principal_id="operator",
            controller_id="controller",
            delegate_id="sentinel",
            scopes=("analyze",),
        ),
    )


def make_chain(scopes=("analyze",)):
    return DelegationChain((
        CapabilityGrant("g1", "operator", "sentinel", tuple(scopes)),
    ))


def test_runtime_connects_policy_reasoning_validation_and_audit():
    runtime = SentinelRuntime(
        engine=FixtureEngine(),
        validators=(FailClosedResponseValidator(),),
    )
    result = runtime.analyze(
        request=make_request(),
        delegation_chain=make_chain(),
        required_scopes=("analyze",),
        audit_id="audit-001",
        now=NOW,
    )

    assert result.request_id == "fixture-001"
    assert result.policy_decision.allowed is True
    assert result.inference.output["severity"] == "high"
    assert result.audit.request_id == "fixture-001"
    assert result.audit.evidence_digests == ("sha256:fixture",)
    assert result.audit.digest().startswith("sha256:")


def test_runtime_denies_before_reasoning_when_scope_missing():
    runtime = SentinelRuntime(
        engine=FixtureEngine(),
        validators=(FailClosedResponseValidator(),),
    )
    with pytest.raises(PermissionError):
        runtime.analyze(
            request=make_request(),
            delegation_chain=make_chain(scopes=("read",)),
            required_scopes=("analyze",),
            audit_id="audit-002",
            now=NOW,
        )
