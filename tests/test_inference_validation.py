import pytest

from koschei_sentinel.inference_contract import (
    EvidenceBinding,
    EvidenceRef,
    InferenceRequest,
    InferenceResponse,
    RiskClass,
)
from koschei_sentinel.inference_validation import (
    FailClosedResponseValidator,
    InferenceValidationError,
)


def request(risk=RiskClass.HIGH, evidence=True):
    refs = (
        EvidenceRef("e1", "sha256:abc", "observation", "sentinel"),
    ) if evidence else ()
    return InferenceRequest(
        request_id="r1",
        tenant_id="t1",
        task_class="security-analysis",
        risk_class=risk,
        prompt="analyze",
        evidence=refs,
    )


def response(bindings=True, evidence_id="e1"):
    refs = (
        EvidenceBinding("c1", (evidence_id,), "supported by observation"),
    ) if bindings else ()
    return InferenceResponse(
        request_id="r1",
        engine_id="engine",
        engine_version="1",
        output={"verdict": "review"},
        evidence_bindings=refs,
    )


def test_high_risk_requires_evidence():
    with pytest.raises(InferenceValidationError):
        FailClosedResponseValidator().validate(request(evidence=False), response())


def test_high_risk_requires_bindings():
    with pytest.raises(InferenceValidationError):
        FailClosedResponseValidator().validate(request(), response(bindings=False))


def test_unknown_evidence_fails_closed():
    with pytest.raises(InferenceValidationError):
        FailClosedResponseValidator().validate(request(), response(evidence_id="missing"))


def test_valid_bound_response_passes():
    FailClosedResponseValidator().validate(request(), response())


def test_refusal_cannot_carry_trusted_output():
    refused = InferenceResponse(
        request_id="r1",
        engine_id="engine",
        engine_version="1",
        output={"verdict": "allow"},
        refusal="declined",
    )
    with pytest.raises(InferenceValidationError):
        FailClosedResponseValidator(
            require_bindings_for_high_risk=False
        ).validate(request(), refused)
