from __future__ import annotations

import base64
import hashlib
from enum import StrEnum
from pathlib import Path
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import canonical_json
from koschei_sentinel.web4_benchmark_intake import (
    Web4BenchmarkAnswerKey,
    Web4BenchmarkIntakePacket,
    Web4BenchmarkSplit,
)
from koschei_sentinel.web4_reviewer_trust import (
    Web4ReviewerTrustPolicy,
    Web4ReviewRole,
    delegate_public_key_fingerprint,
    verify_web4_reviewer_trust_policy,
)

_DIGEST = r"^[a-f0-9]{64}$"
_PRIMARY_SIGNATURE_CONTEXT = b"koschei-sentinel-web4-human-review-v1\0"
_ADJUDICATION_SIGNATURE_CONTEXT = b"koschei-sentinel-web4-adjudication-v1\0"


class Web4HumanReviewDecision(StrEnum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"


class Web4AdjudicationDecision(StrEnum):
    CONFIRM_PRIMARY = "CONFIRM_PRIMARY"
    DISPUTE_PRIMARY = "DISPUTE_PRIMARY"


class Web4BenchmarkHumanReviewSpec(StrictModel):
    schema_version: Literal["sentinel.web4-benchmark-human-review-spec.v1"] = (
        "sentinel.web4-benchmark-human-review-spec.v1"
    )
    reviewer_id: str = Field(min_length=3, max_length=128)
    decision: Web4HumanReviewDecision
    rationale: str = Field(min_length=16, max_length=8000)
    source_refs_verified: bool = False
    source_revision_status_verified: bool = False
    answer_key_verified: bool = False
    authority_status_checked: bool = False
    benchmark_family_checked: bool = False

    @model_validator(mode="after")
    def approval_requires_complete_checks(self) -> Web4BenchmarkHumanReviewSpec:
        if self.decision is Web4HumanReviewDecision.APPROVE:
            checks = (
                self.source_refs_verified,
                self.source_revision_status_verified,
                self.answer_key_verified,
                self.authority_status_checked,
                self.benchmark_family_checked,
            )
            if not all(checks):
                raise ValueError("approved Web4 human review requires all verification checks")
        return self


class Web4BenchmarkHumanReview(StrictModel):
    schema_version: Literal["sentinel.web4-benchmark-human-review.v1"] = (
        "sentinel.web4-benchmark-human-review.v1"
    )
    case_id: str = Field(min_length=3, max_length=256)
    family: str = Field(min_length=3, max_length=256)
    split: Web4BenchmarkSplit
    packet_sha256: str = Field(pattern=_DIGEST)
    model_input_sha256: str = Field(pattern=_DIGEST)
    answer_key_sha256: str = Field(pattern=_DIGEST)
    source_registry_sha256: str = Field(pattern=_DIGEST)
    reviewer_id: str = Field(min_length=3, max_length=128)
    reviewer_key_fingerprint: str = Field(pattern=_DIGEST)
    reviewer_trust_policy_digest: str = Field(pattern=_DIGEST)
    decision: Web4HumanReviewDecision
    rationale: str = Field(min_length=16, max_length=8000)
    source_refs_verified: bool
    source_revision_status_verified: bool
    answer_key_verified: bool
    authority_status_checked: bool
    benchmark_family_checked: bool
    review_status: Literal["HUMAN_REVIEWED"] = "HUMAN_REVIEWED"
    human_reviewed: Literal[True] = True
    contains_answer_key: Literal[False] = False
    training_authorization: Literal[False] = False
    evaluation_authorization: Literal[False] = False
    promotion_eligible: Literal[False] = False
    review_sha256: str = Field(pattern=_DIGEST)
    signature_algorithm: Literal["ed25519"] = "ed25519"
    signature_base64: str = Field(min_length=80, max_length=128)
    signature_verified: Literal[True] = True
    artifact_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def artifact_self_hash_verifies(self) -> Web4BenchmarkHumanReview:
        payload = self.model_dump(mode="json")
        observed = str(payload.pop("artifact_sha256"))
        if _digest(payload) != observed:
            raise ValueError("Web4 human review artifact self-hash does not verify")
        if self.decision is Web4HumanReviewDecision.APPROVE:
            checks = (
                self.source_refs_verified,
                self.source_revision_status_verified,
                self.answer_key_verified,
                self.authority_status_checked,
                self.benchmark_family_checked,
            )
            if not all(checks):
                raise ValueError("approved Web4 human review lacks required verification checks")
        return self


class Web4BenchmarkAdjudicationSpec(StrictModel):
    schema_version: Literal["sentinel.web4-benchmark-adjudication-spec.v1"] = (
        "sentinel.web4-benchmark-adjudication-spec.v1"
    )
    adjudicator_id: str = Field(min_length=3, max_length=128)
    decision: Web4AdjudicationDecision
    rationale: str = Field(min_length=16, max_length=8000)
    independently_checked_answer_key: bool = False
    independently_checked_source_bindings: bool = False
    independently_checked_review_reasoning: bool = False

    @model_validator(mode="after")
    def confirmation_requires_independent_checks(self) -> Web4BenchmarkAdjudicationSpec:
        if self.decision is Web4AdjudicationDecision.CONFIRM_PRIMARY:
            checks = (
                self.independently_checked_answer_key,
                self.independently_checked_source_bindings,
                self.independently_checked_review_reasoning,
            )
            if not all(checks):
                raise ValueError("confirmed Web4 adjudication requires all independent checks")
        return self


class Web4BenchmarkAdjudication(StrictModel):
    schema_version: Literal["sentinel.web4-benchmark-adjudication.v1"] = (
        "sentinel.web4-benchmark-adjudication.v1"
    )
    case_id: str = Field(min_length=3, max_length=256)
    family: str = Field(min_length=3, max_length=256)
    split: Web4BenchmarkSplit
    packet_sha256: str = Field(pattern=_DIGEST)
    review_sha256: str = Field(pattern=_DIGEST)
    review_artifact_sha256: str = Field(pattern=_DIGEST)
    primary_reviewer_id: str = Field(min_length=3, max_length=128)
    primary_reviewer_key_fingerprint: str = Field(pattern=_DIGEST)
    primary_review_decision: Web4HumanReviewDecision
    adjudicator_id: str = Field(min_length=3, max_length=128)
    adjudicator_key_fingerprint: str = Field(pattern=_DIGEST)
    adjudicator_trust_policy_digest: str = Field(pattern=_DIGEST)
    decision: Web4AdjudicationDecision
    rationale: str = Field(min_length=16, max_length=8000)
    independently_checked_answer_key: bool
    independently_checked_source_bindings: bool
    independently_checked_review_reasoning: bool
    independent_adjudication: Literal[True] = True
    final_review_approved: bool
    eligible_for_signed_holdout_release: bool
    contains_answer_key: Literal[False] = False
    training_authorization: Literal[False] = False
    evaluation_authorization: Literal[False] = False
    promotion_eligible: Literal[False] = False
    adjudication_sha256: str = Field(pattern=_DIGEST)
    signature_algorithm: Literal["ed25519"] = "ed25519"
    signature_base64: str = Field(min_length=80, max_length=128)
    signature_verified: Literal[True] = True
    artifact_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def artifact_contract_verifies(self) -> Web4BenchmarkAdjudication:
        payload = self.model_dump(mode="json")
        observed = str(payload.pop("artifact_sha256"))
        if _digest(payload) != observed:
            raise ValueError("Web4 adjudication artifact self-hash does not verify")
        if self.primary_reviewer_id == self.adjudicator_id:
            raise ValueError("Web4 adjudicator must be independent from primary reviewer")
        if self.primary_reviewer_key_fingerprint == self.adjudicator_key_fingerprint:
            raise ValueError("Web4 adjudicator must use an independent key")
        expected_final = (
            self.primary_review_decision is Web4HumanReviewDecision.APPROVE
            and self.decision is Web4AdjudicationDecision.CONFIRM_PRIMARY
        )
        if self.final_review_approved is not expected_final:
            raise ValueError("Web4 adjudication final approval state is inconsistent")
        expected_release = expected_final and self.split is Web4BenchmarkSplit.HOLDOUT
        if self.eligible_for_signed_holdout_release is not expected_release:
            raise ValueError("Web4 signed HOLDOUT release eligibility is inconsistent")
        return self


def _digest(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _revalidate_packet(packet: Web4BenchmarkIntakePacket) -> Web4BenchmarkIntakePacket:
    return Web4BenchmarkIntakePacket.model_validate(packet.model_dump(mode="json"))


def _read_answer_key(
    path: str | Path,
    packet: Web4BenchmarkIntakePacket,
) -> Web4BenchmarkAnswerKey:
    answer_path = Path(path)
    if answer_path.is_symlink() or not answer_path.is_file():
        raise ValueError("Web4 benchmark answer key must be a regular non-symlink file")
    raw = answer_path.read_bytes()
    observed_sha = hashlib.sha256(raw).hexdigest()
    if observed_sha != packet.answer_key_sha256:
        raise ValueError("Web4 benchmark answer key drifted after intake")
    try:
        answer_key = Web4BenchmarkAnswerKey.model_validate_json(raw)
    except ValueError as exc:
        raise ValueError("invalid Web4 benchmark answer key") from exc
    if answer_key.case_id != packet.case_id:
        raise ValueError("Web4 benchmark answer key case_id differs from intake packet")
    if set(answer_key.source_refs) != set(packet.source_refs):
        raise ValueError("Web4 benchmark answer key source refs differ from intake packet")
    return answer_key


def _review_binding_payload(
    packet: Web4BenchmarkIntakePacket,
    spec: Web4BenchmarkHumanReviewSpec,
    reviewer_key_fingerprint: str,
    trust_policy: Web4ReviewerTrustPolicy,
) -> dict[str, object]:
    return {
        "schema_version": "sentinel.web4-benchmark-human-review.v1",
        "case_id": packet.case_id,
        "family": packet.family,
        "split": packet.split.value,
        "packet_sha256": packet.packet_sha256,
        "model_input_sha256": packet.model_input_sha256,
        "answer_key_sha256": packet.answer_key_sha256,
        "source_registry_sha256": packet.source_registry_sha256,
        "reviewer_id": spec.reviewer_id,
        "reviewer_key_fingerprint": reviewer_key_fingerprint,
        "reviewer_trust_policy_digest": trust_policy.policy_digest,
        "decision": spec.decision.value,
        "rationale": spec.rationale,
        "source_refs_verified": spec.source_refs_verified,
        "source_revision_status_verified": spec.source_revision_status_verified,
        "answer_key_verified": spec.answer_key_verified,
        "authority_status_checked": spec.authority_status_checked,
        "benchmark_family_checked": spec.benchmark_family_checked,
        "review_status": "HUMAN_REVIEWED",
        "human_reviewed": True,
        "contains_answer_key": False,
        "training_authorization": False,
        "evaluation_authorization": False,
        "promotion_eligible": False,
    }


def _review_signature_message(review_sha256: str) -> bytes:
    return _PRIMARY_SIGNATURE_CONTEXT + review_sha256.encode("ascii")


def sign_web4_benchmark_human_review(
    *,
    packet: Web4BenchmarkIntakePacket,
    answer_key_path: str | Path,
    spec: Web4BenchmarkHumanReviewSpec,
    reviewer_private_key: Ed25519PrivateKey,
    trust_policy: Web4ReviewerTrustPolicy,
    owner_public_key: Ed25519PublicKey,
) -> Web4BenchmarkHumanReview:
    packet = _revalidate_packet(packet)
    _read_answer_key(answer_key_path, packet)
    if trust_policy.role is not Web4ReviewRole.PRIMARY_REVIEWER:
        raise ValueError("Web4 primary review requires PRIMARY_REVIEWER trust policy")
    if trust_policy.delegate_id != spec.reviewer_id:
        raise ValueError("Web4 reviewer identity does not match trust policy")
    reviewer_public_key = reviewer_private_key.public_key()
    verify_web4_reviewer_trust_policy(trust_policy, reviewer_public_key, owner_public_key)

    fingerprint = delegate_public_key_fingerprint(reviewer_public_key)
    binding = _review_binding_payload(packet, spec, fingerprint, trust_policy)
    review_sha = _digest(binding)
    signature = reviewer_private_key.sign(_review_signature_message(review_sha))
    reviewer_public_key.verify(signature, _review_signature_message(review_sha))
    payload: dict[str, object] = {
        **binding,
        "review_sha256": review_sha,
        "signature_algorithm": "ed25519",
        "signature_base64": base64.b64encode(signature).decode("ascii"),
        "signature_verified": True,
    }
    payload["artifact_sha256"] = _digest(payload)
    return Web4BenchmarkHumanReview.model_validate(payload)


def verify_web4_benchmark_human_review(
    *,
    review: Web4BenchmarkHumanReview,
    packet: Web4BenchmarkIntakePacket,
    answer_key_path: str | Path,
    reviewer_public_key: Ed25519PublicKey,
    trust_policy: Web4ReviewerTrustPolicy,
    owner_public_key: Ed25519PublicKey,
) -> Web4BenchmarkHumanReview:
    packet = _revalidate_packet(packet)
    review = Web4BenchmarkHumanReview.model_validate(review.model_dump(mode="json"))
    _read_answer_key(answer_key_path, packet)
    if trust_policy.role is not Web4ReviewRole.PRIMARY_REVIEWER:
        raise ValueError("Web4 human review trust policy role is not PRIMARY_REVIEWER")
    verify_web4_reviewer_trust_policy(trust_policy, reviewer_public_key, owner_public_key)
    fingerprint = delegate_public_key_fingerprint(reviewer_public_key)
    if review.reviewer_key_fingerprint != fingerprint:
        raise ValueError("Web4 human review uses an untrusted reviewer key")
    if review.reviewer_id != trust_policy.delegate_id:
        raise ValueError("Web4 human review reviewer_id differs from trust policy")
    if review.reviewer_trust_policy_digest != trust_policy.policy_digest:
        raise ValueError("Web4 human review does not bind the reviewer trust policy")

    expected_binding = {
        "schema_version": "sentinel.web4-benchmark-human-review.v1",
        "case_id": packet.case_id,
        "family": packet.family,
        "split": packet.split.value,
        "packet_sha256": packet.packet_sha256,
        "model_input_sha256": packet.model_input_sha256,
        "answer_key_sha256": packet.answer_key_sha256,
        "source_registry_sha256": packet.source_registry_sha256,
        "reviewer_id": review.reviewer_id,
        "reviewer_key_fingerprint": review.reviewer_key_fingerprint,
        "reviewer_trust_policy_digest": review.reviewer_trust_policy_digest,
        "decision": review.decision.value,
        "rationale": review.rationale,
        "source_refs_verified": review.source_refs_verified,
        "source_revision_status_verified": review.source_revision_status_verified,
        "answer_key_verified": review.answer_key_verified,
        "authority_status_checked": review.authority_status_checked,
        "benchmark_family_checked": review.benchmark_family_checked,
        "review_status": "HUMAN_REVIEWED",
        "human_reviewed": True,
        "contains_answer_key": False,
        "training_authorization": False,
        "evaluation_authorization": False,
        "promotion_eligible": False,
    }
    if _digest(expected_binding) != review.review_sha256:
        raise ValueError("Web4 human review digest does not bind the intake packet")
    try:
        signature = base64.b64decode(review.signature_base64, validate=True)
        reviewer_public_key.verify(signature, _review_signature_message(review.review_sha256))
    except (InvalidSignature, ValueError) as exc:
        raise ValueError("Web4 human reviewer Ed25519 signature verification failed") from exc
    return review


def _adjudication_binding_payload(
    *,
    review: Web4BenchmarkHumanReview,
    spec: Web4BenchmarkAdjudicationSpec,
    adjudicator_key_fingerprint: str,
    trust_policy: Web4ReviewerTrustPolicy,
) -> dict[str, object]:
    final_review_approved = (
        review.decision is Web4HumanReviewDecision.APPROVE
        and spec.decision is Web4AdjudicationDecision.CONFIRM_PRIMARY
    )
    return {
        "schema_version": "sentinel.web4-benchmark-adjudication.v1",
        "case_id": review.case_id,
        "family": review.family,
        "split": review.split.value,
        "packet_sha256": review.packet_sha256,
        "review_sha256": review.review_sha256,
        "review_artifact_sha256": review.artifact_sha256,
        "primary_reviewer_id": review.reviewer_id,
        "primary_reviewer_key_fingerprint": review.reviewer_key_fingerprint,
        "primary_review_decision": review.decision.value,
        "adjudicator_id": spec.adjudicator_id,
        "adjudicator_key_fingerprint": adjudicator_key_fingerprint,
        "adjudicator_trust_policy_digest": trust_policy.policy_digest,
        "decision": spec.decision.value,
        "rationale": spec.rationale,
        "independently_checked_answer_key": spec.independently_checked_answer_key,
        "independently_checked_source_bindings": spec.independently_checked_source_bindings,
        "independently_checked_review_reasoning": spec.independently_checked_review_reasoning,
        "independent_adjudication": True,
        "final_review_approved": final_review_approved,
        "eligible_for_signed_holdout_release": (
            final_review_approved and review.split is Web4BenchmarkSplit.HOLDOUT
        ),
        "contains_answer_key": False,
        "training_authorization": False,
        "evaluation_authorization": False,
        "promotion_eligible": False,
    }


def _adjudication_signature_message(adjudication_sha256: str) -> bytes:
    return _ADJUDICATION_SIGNATURE_CONTEXT + adjudication_sha256.encode("ascii")


def adjudicate_web4_benchmark_review(
    *,
    review: Web4BenchmarkHumanReview,
    packet: Web4BenchmarkIntakePacket,
    answer_key_path: str | Path,
    reviewer_public_key: Ed25519PublicKey,
    reviewer_trust_policy: Web4ReviewerTrustPolicy,
    adjudication_spec: Web4BenchmarkAdjudicationSpec,
    adjudicator_private_key: Ed25519PrivateKey,
    adjudicator_trust_policy: Web4ReviewerTrustPolicy,
    owner_public_key: Ed25519PublicKey,
) -> Web4BenchmarkAdjudication:
    review = verify_web4_benchmark_human_review(
        review=review,
        packet=packet,
        answer_key_path=answer_key_path,
        reviewer_public_key=reviewer_public_key,
        trust_policy=reviewer_trust_policy,
        owner_public_key=owner_public_key,
    )
    if adjudicator_trust_policy.role is not Web4ReviewRole.ADJUDICATOR:
        raise ValueError("Web4 adjudication requires ADJUDICATOR trust policy")
    if adjudicator_trust_policy.delegate_id != adjudication_spec.adjudicator_id:
        raise ValueError("Web4 adjudicator identity does not match trust policy")

    adjudicator_public_key = adjudicator_private_key.public_key()
    verify_web4_reviewer_trust_policy(
        adjudicator_trust_policy,
        adjudicator_public_key,
        owner_public_key,
    )
    adjudicator_fingerprint = delegate_public_key_fingerprint(adjudicator_public_key)
    if review.reviewer_id == adjudication_spec.adjudicator_id:
        raise ValueError("Web4 adjudicator must be a different identity from primary reviewer")
    if review.reviewer_key_fingerprint == adjudicator_fingerprint:
        raise ValueError("Web4 adjudicator must use a different key from primary reviewer")

    binding = _adjudication_binding_payload(
        review=review,
        spec=adjudication_spec,
        adjudicator_key_fingerprint=adjudicator_fingerprint,
        trust_policy=adjudicator_trust_policy,
    )
    adjudication_sha = _digest(binding)
    signature = adjudicator_private_key.sign(
        _adjudication_signature_message(adjudication_sha)
    )
    adjudicator_public_key.verify(
        signature,
        _adjudication_signature_message(adjudication_sha),
    )
    payload: dict[str, object] = {
        **binding,
        "adjudication_sha256": adjudication_sha,
        "signature_algorithm": "ed25519",
        "signature_base64": base64.b64encode(signature).decode("ascii"),
        "signature_verified": True,
    }
    payload["artifact_sha256"] = _digest(payload)
    return Web4BenchmarkAdjudication.model_validate(payload)


def verify_web4_benchmark_adjudication(
    *,
    adjudication: Web4BenchmarkAdjudication,
    review: Web4BenchmarkHumanReview,
    packet: Web4BenchmarkIntakePacket,
    answer_key_path: str | Path,
    reviewer_public_key: Ed25519PublicKey,
    reviewer_trust_policy: Web4ReviewerTrustPolicy,
    adjudicator_public_key: Ed25519PublicKey,
    adjudicator_trust_policy: Web4ReviewerTrustPolicy,
    owner_public_key: Ed25519PublicKey,
) -> Web4BenchmarkAdjudication:
    review = verify_web4_benchmark_human_review(
        review=review,
        packet=packet,
        answer_key_path=answer_key_path,
        reviewer_public_key=reviewer_public_key,
        trust_policy=reviewer_trust_policy,
        owner_public_key=owner_public_key,
    )
    adjudication = Web4BenchmarkAdjudication.model_validate(
        adjudication.model_dump(mode="json")
    )
    if adjudicator_trust_policy.role is not Web4ReviewRole.ADJUDICATOR:
        raise ValueError("Web4 adjudication trust policy role is not ADJUDICATOR")
    verify_web4_reviewer_trust_policy(
        adjudicator_trust_policy,
        adjudicator_public_key,
        owner_public_key,
    )
    adjudicator_fingerprint = delegate_public_key_fingerprint(adjudicator_public_key)
    if adjudication.adjudicator_key_fingerprint != adjudicator_fingerprint:
        raise ValueError("Web4 adjudication uses an untrusted adjudicator key")
    if adjudication.adjudicator_id != adjudicator_trust_policy.delegate_id:
        raise ValueError("Web4 adjudication adjudicator_id differs from trust policy")
    if adjudication.adjudicator_trust_policy_digest != adjudicator_trust_policy.policy_digest:
        raise ValueError("Web4 adjudication does not bind adjudicator trust policy")
    if adjudication.review_sha256 != review.review_sha256:
        raise ValueError("Web4 adjudication does not bind supplied human review")
    if adjudication.review_artifact_sha256 != review.artifact_sha256:
        raise ValueError("Web4 adjudication does not bind supplied review artifact")
    if adjudication.packet_sha256 != packet.packet_sha256:
        raise ValueError("Web4 adjudication does not bind supplied intake packet")

    spec = Web4BenchmarkAdjudicationSpec(
        adjudicator_id=adjudication.adjudicator_id,
        decision=adjudication.decision,
        rationale=adjudication.rationale,
        independently_checked_answer_key=adjudication.independently_checked_answer_key,
        independently_checked_source_bindings=adjudication.independently_checked_source_bindings,
        independently_checked_review_reasoning=adjudication.independently_checked_review_reasoning,
    )
    expected_binding = _adjudication_binding_payload(
        review=review,
        spec=spec,
        adjudicator_key_fingerprint=adjudication.adjudicator_key_fingerprint,
        trust_policy=adjudicator_trust_policy,
    )
    if _digest(expected_binding) != adjudication.adjudication_sha256:
        raise ValueError("Web4 adjudication digest does not bind review and policy")
    try:
        signature = base64.b64decode(adjudication.signature_base64, validate=True)
        adjudicator_public_key.verify(
            signature,
            _adjudication_signature_message(adjudication.adjudication_sha256),
        )
    except (InvalidSignature, ValueError) as exc:
        raise ValueError("Web4 adjudicator Ed25519 signature verification failed") from exc
    return adjudication
