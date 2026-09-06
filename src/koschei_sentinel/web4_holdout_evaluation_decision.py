from __future__ import annotations

import base64
import hashlib
from enum import StrEnum
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.promotion import public_key_fingerprint
from koschei_sentinel.training import canonical_json
from koschei_sentinel.web4_holdout_evaluation import Web4HoldoutEvaluationEvidence

_DIGEST = r"^[a-f0-9]{64}$"
_DECISION_ID = r"^[a-z0-9][a-z0-9._-]{2,127}$"
_APPROVER_ID = r"^[A-Za-z0-9][A-Za-z0-9._@-]{0,127}$"
_SIGNATURE_CONTEXT = b"koschei-sentinel-web4-holdout-evaluation-decision-v1\0"


class Web4EvaluationDecision(StrEnum):
    ACCEPT = "ACCEPT_RESEARCH_EVIDENCE"
    REJECT = "REJECT_RESEARCH_EVIDENCE"
    HOLD = "HOLD_RESEARCH_EVIDENCE"


class Web4HoldoutEvaluationDecisionSpec(StrictModel):
    schema_version: Literal["sentinel.web4-holdout-evaluation-decision-spec.v1"] = (
        "sentinel.web4-holdout-evaluation-decision-spec.v1"
    )
    decision_id: str = Field(pattern=_DECISION_ID)
    approver_id: str = Field(pattern=_APPROVER_ID)
    decision: Web4EvaluationDecision
    rationale: str = Field(min_length=8, max_length=8000)


class Web4HoldoutEvaluationDecision(StrictModel):
    schema_version: Literal["sentinel.web4-holdout-evaluation-decision.v1"] = (
        "sentinel.web4-holdout-evaluation-decision.v1"
    )
    decision_id: str = Field(pattern=_DECISION_ID)
    state: Literal["owner_signed_research_evaluation_decision"] = (
        "owner_signed_research_evaluation_decision"
    )
    approver_id: str = Field(pattern=_APPROVER_ID)
    decision: Web4EvaluationDecision
    rationale: str = Field(min_length=8, max_length=8000)
    evidence_sha256: str = Field(pattern=_DIGEST)
    report_sha256: str = Field(pattern=_DIGEST)
    release_id: str
    release_sha256: str = Field(pattern=_DIGEST)
    release_artifact_sha256: str = Field(pattern=_DIGEST)
    inference_pack_sha256: str = Field(pattern=_DIGEST)
    prediction_set_sha256: str = Field(pattern=_DIGEST)
    answer_key_bundle_sha256: str = Field(pattern=_DIGEST)
    evaluation_policy_sha256: str = Field(pattern=_DIGEST)
    model_ref: str
    model_revision: str
    model_artifact_sha256: str = Field(pattern=_DIGEST)
    case_count: int = Field(gt=0)
    prediction_count: int = Field(ge=0)
    evaluation_passed: bool
    complete_case_accounting: bool
    exact_match_rate: float = Field(ge=0.0, le=1.0)
    expected_field_accuracy: float = Field(ge=0.0, le=1.0)
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    research_evaluation_evidence_accepted: bool
    research_comparison_eligible: bool
    research_evaluation_only: Literal[True] = True
    model_execution_authorized: Literal[False] = False
    training_authorization: Literal[False] = False
    promotion_eligible: Literal[False] = False
    production_activation_allowed: Literal[False] = False
    decision_sha256: str = Field(pattern=_DIGEST)
    signature_algorithm: Literal["ed25519"] = "ed25519"
    owner_signature_base64: str = Field(min_length=80, max_length=128)
    owner_signature_verified: Literal[True] = True
    artifact_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def decision_contract_verifies(self) -> Web4HoldoutEvaluationDecision:
        accepted = self.decision is Web4EvaluationDecision.ACCEPT
        if self.research_evaluation_evidence_accepted is not accepted:
            raise ValueError("Web4 evaluation decision accepted state is inconsistent")
        if self.research_comparison_eligible is not accepted:
            raise ValueError("Web4 evaluation decision research comparison state is inconsistent")
        if accepted and not self.evaluation_passed:
            raise ValueError("Web4 evaluation evidence cannot be accepted when evaluation failed")
        if accepted and not self.complete_case_accounting:
            raise ValueError("Web4 evaluation evidence cannot be accepted with incomplete accounting")

        payload = self.model_dump(mode="json")
        observed_artifact = str(payload.pop("artifact_sha256"))
        if _digest(payload) != observed_artifact:
            raise ValueError("Web4 evaluation decision artifact self-hash does not verify")

        semantic = dict(payload)
        semantic.pop("signature_algorithm")
        semantic.pop("owner_signature_base64")
        semantic.pop("owner_signature_verified")
        observed_decision = str(semantic.pop("decision_sha256"))
        if _digest(semantic) != observed_decision:
            raise ValueError("Web4 evaluation decision digest does not verify")
        return self


def _digest(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _signature_message(decision_sha256: str) -> bytes:
    return _SIGNATURE_CONTEXT + decision_sha256.encode("ascii")


def _semantic_payload(
    *,
    evidence: Web4HoldoutEvaluationEvidence,
    spec: Web4HoldoutEvaluationDecisionSpec,
    owner_key_fingerprint: str,
) -> dict[str, object]:
    accepted = spec.decision is Web4EvaluationDecision.ACCEPT
    return {
        "schema_version": "sentinel.web4-holdout-evaluation-decision.v1",
        "decision_id": spec.decision_id,
        "state": "owner_signed_research_evaluation_decision",
        "approver_id": spec.approver_id,
        "decision": spec.decision.value,
        "rationale": spec.rationale,
        "evidence_sha256": evidence.evidence_sha256,
        "report_sha256": evidence.report.report_sha256,
        "release_id": evidence.release_id,
        "release_sha256": evidence.release_sha256,
        "release_artifact_sha256": evidence.release_artifact_sha256,
        "inference_pack_sha256": evidence.inference_pack_sha256,
        "prediction_set_sha256": evidence.prediction_set_sha256,
        "answer_key_bundle_sha256": evidence.answer_key_bundle_sha256,
        "evaluation_policy_sha256": evidence.evaluation_policy_sha256,
        "model_ref": evidence.model_ref,
        "model_revision": evidence.model_revision,
        "model_artifact_sha256": evidence.model_artifact_sha256,
        "case_count": evidence.case_count,
        "prediction_count": evidence.prediction_count,
        "evaluation_passed": evidence.passed,
        "complete_case_accounting": evidence.complete_case_accounting,
        "exact_match_rate": evidence.report.exact_match_rate,
        "expected_field_accuracy": evidence.report.expected_field_accuracy,
        "owner_key_fingerprint": owner_key_fingerprint,
        "research_evaluation_evidence_accepted": accepted,
        "research_comparison_eligible": accepted,
        "research_evaluation_only": True,
        "model_execution_authorized": False,
        "training_authorization": False,
        "promotion_eligible": False,
        "production_activation_allowed": False,
    }


def sign_web4_holdout_evaluation_decision(
    *,
    evidence: Web4HoldoutEvaluationEvidence,
    spec: Web4HoldoutEvaluationDecisionSpec,
    owner_private_key: Ed25519PrivateKey,
) -> Web4HoldoutEvaluationDecision:
    evidence = Web4HoldoutEvaluationEvidence.model_validate(evidence.model_dump(mode="json"))
    spec = Web4HoldoutEvaluationDecisionSpec.model_validate(spec.model_dump(mode="json"))
    owner_public_key = owner_private_key.public_key()
    fingerprint = public_key_fingerprint(owner_public_key)
    if fingerprint != evidence.release_owner_key_fingerprint:
        raise ValueError("Web4 evaluation decision owner key differs from release owner")
    if spec.decision is Web4EvaluationDecision.ACCEPT:
        if not evidence.passed:
            raise ValueError("cannot accept failed Web4 HOLDOUT evaluation evidence")
        if not evidence.complete_case_accounting:
            raise ValueError("cannot accept Web4 HOLDOUT evidence with incomplete accounting")

    semantic = _semantic_payload(
        evidence=evidence,
        spec=spec,
        owner_key_fingerprint=fingerprint,
    )
    decision_sha256 = _digest(semantic)
    signature = owner_private_key.sign(_signature_message(decision_sha256))
    owner_public_key.verify(signature, _signature_message(decision_sha256))
    signed: dict[str, object] = {
        **semantic,
        "decision_sha256": decision_sha256,
        "signature_algorithm": "ed25519",
        "owner_signature_base64": base64.b64encode(signature).decode("ascii"),
        "owner_signature_verified": True,
    }
    signed["artifact_sha256"] = _digest(signed)
    decision = Web4HoldoutEvaluationDecision.model_validate(signed)
    return verify_web4_holdout_evaluation_decision(
        evidence=evidence,
        decision=decision,
        owner_public_key=owner_public_key,
    )


def verify_web4_holdout_evaluation_decision(
    *,
    evidence: Web4HoldoutEvaluationEvidence,
    decision: Web4HoldoutEvaluationDecision,
    owner_public_key: Ed25519PublicKey,
) -> Web4HoldoutEvaluationDecision:
    evidence = Web4HoldoutEvaluationEvidence.model_validate(evidence.model_dump(mode="json"))
    decision = Web4HoldoutEvaluationDecision.model_validate(decision.model_dump(mode="json"))
    fingerprint = public_key_fingerprint(owner_public_key)
    if evidence.release_owner_key_fingerprint != fingerprint:
        raise ValueError("Web4 evaluation evidence belongs to a different release owner")
    if decision.owner_key_fingerprint != fingerprint:
        raise ValueError("Web4 evaluation decision belongs to a different owner")

    bindings = {
        "evidence_sha256": evidence.evidence_sha256,
        "report_sha256": evidence.report.report_sha256,
        "release_id": evidence.release_id,
        "release_sha256": evidence.release_sha256,
        "release_artifact_sha256": evidence.release_artifact_sha256,
        "inference_pack_sha256": evidence.inference_pack_sha256,
        "prediction_set_sha256": evidence.prediction_set_sha256,
        "answer_key_bundle_sha256": evidence.answer_key_bundle_sha256,
        "evaluation_policy_sha256": evidence.evaluation_policy_sha256,
        "model_ref": evidence.model_ref,
        "model_revision": evidence.model_revision,
        "model_artifact_sha256": evidence.model_artifact_sha256,
        "case_count": evidence.case_count,
        "prediction_count": evidence.prediction_count,
        "evaluation_passed": evidence.passed,
        "complete_case_accounting": evidence.complete_case_accounting,
        "exact_match_rate": evidence.report.exact_match_rate,
        "expected_field_accuracy": evidence.report.expected_field_accuracy,
    }
    for field_name, expected in bindings.items():
        if getattr(decision, field_name) != expected:
            raise ValueError(f"Web4 evaluation decision {field_name} binding differs from evidence")

    if decision.decision is Web4EvaluationDecision.ACCEPT:
        if not evidence.passed or not evidence.complete_case_accounting:
            raise ValueError("accepted Web4 evaluation decision requires passing complete evidence")

    try:
        signature = base64.b64decode(decision.owner_signature_base64, validate=True)
        owner_public_key.verify(signature, _signature_message(decision.decision_sha256))
    except (InvalidSignature, ValueError) as exc:
        raise ValueError("Web4 evaluation decision owner signature verification failed") from exc
    return decision
