from __future__ import annotations

from dataclasses import dataclass

from .inference_contract import InferenceRequest, InferenceResponse


class InferenceValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class FailClosedResponseValidator:
    """Validates Sentinel-owned invariants before reasoning output is trusted."""

    require_evidence_for_high_risk: bool = True
    require_bindings_for_high_risk: bool = True

    def validate(self, request: InferenceRequest, response: InferenceResponse) -> None:
        if not response.engine_id.strip() or not response.engine_version.strip():
            raise InferenceValidationError("missing reasoning engine identity")

        evidence_ids = {item.evidence_id for item in request.evidence}
        if len(evidence_ids) != len(request.evidence):
            raise InferenceValidationError("duplicate evidence_id in request")

        high_risk = request.risk_class.value in {"high", "critical"}
        if high_risk and self.require_evidence_for_high_risk and not request.evidence:
            raise InferenceValidationError("high-risk reasoning requires evidence")

        if high_risk and self.require_bindings_for_high_risk and not response.evidence_bindings:
            raise InferenceValidationError("high-risk reasoning requires evidence bindings")

        seen_claims: set[str] = set()
        for binding in response.evidence_bindings:
            if not binding.claim_id.strip():
                raise InferenceValidationError("empty claim_id")
            if binding.claim_id in seen_claims:
                raise InferenceValidationError("duplicate claim_id")
            seen_claims.add(binding.claim_id)

            if not binding.evidence_ids:
                raise InferenceValidationError("claim has no evidence binding")
            unknown = set(binding.evidence_ids) - evidence_ids
            if unknown:
                raise InferenceValidationError(
                    f"claim references unknown evidence: {sorted(unknown)!r}"
                )
            if not binding.rationale.strip():
                raise InferenceValidationError("claim binding requires rationale")

        if response.refusal is not None and response.output:
            raise InferenceValidationError("refusal response must not carry trusted output")
