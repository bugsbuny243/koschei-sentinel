from __future__ import annotations

import base64
import hashlib
import json
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import Field, model_validator

from koschei_sentinel.gold_holdout_evaluation import GoldHoldoutEvaluationReport
from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_checkpoint_manifest import RankCheckpointManifest
from koschei_sentinel.production_checkpoint_trust import (
    ApprovedRankCheckpointReceipt,
    verify_approved_rank_checkpoint_load,
)
from koschei_sentinel.production_mcore_training_checkpoint import MCoreTrainingState
from koschei_sentinel.production_mcore_training_loop import MCoreTrainingLoopResult
from koschei_sentinel.production_target_binding import ProductionTargetBinding
from koschei_sentinel.production_target_provenance import (
    ProductionTargetProvenance,
    owner_key_fingerprint,
)

_DIGEST = r"^[a-f0-9]{64}$"
_RUN_ID = r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$"
_TRAINING_DOMAIN = b"koschei-sentinel-training-lineage-v1\0"
_HOLDOUT_DOMAIN = b"koschei-sentinel-independent-holdout-lineage-v1\0"
_MIN_HOLDOUT_CASES = 50


def _digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unsigned(value: StrictModel, signature_field: str) -> dict[str, object]:
    return value.model_dump(mode="json", exclude={"attestation_sha256", signature_field})


def _verify_signature(
    attestation: StrictModel,
    *,
    signature_field: str,
    signature_domain: bytes,
    signer: Ed25519PublicKey,
) -> None:
    claimed_digest = getattr(attestation, "attestation_sha256")
    if _digest(_unsigned(attestation, signature_field)) != claimed_digest:
        raise ValueError("lineage attestation self-hash mismatch")
    if getattr(attestation, "signer_key_fingerprint") != owner_key_fingerprint(signer):
        raise ValueError("lineage attestation signer fingerprint mismatch")
    try:
        signature = base64.b64decode(
            getattr(attestation, signature_field), validate=True
        )
    except ValueError as exc:
        raise ValueError("lineage attestation signature is invalid base64") from exc
    try:
        signer.verify(signature, signature_domain + claimed_digest.encode("ascii"))
    except InvalidSignature as exc:
        raise ValueError("lineage attestation signature verification failed") from exc


class ProductionTrainingLineage(StrictModel):
    schema_version: Literal["sentinel.production-training-lineage.v1"] = (
        "sentinel.production-training-lineage.v1"
    )
    model_ref: str = Field(min_length=1, max_length=512)
    model_revision: str = Field(pattern=_DIGEST)
    checkpoint_approval_sha256: str = Field(pattern=_DIGEST)
    checkpoint_file_sha256: str = Field(pattern=_DIGEST)
    training_run_id: str = Field(pattern=_RUN_ID)
    training_state_sha256: str = Field(pattern=_DIGEST)
    training_loop_sha256: str = Field(pattern=_DIGEST)
    dataset_manifest_sha256: str = Field(pattern=_DIGEST)
    final_step: int = Field(gt=0)
    signer_key_fingerprint: str = Field(pattern=_DIGEST)
    attestation_sha256: str = Field(pattern=_DIGEST)
    operator_signature_b64: str = Field(min_length=1)

    @model_validator(mode="after")
    def self_hash_verifies(self) -> ProductionTrainingLineage:
        if _digest(_unsigned(self, "operator_signature_b64")) != self.attestation_sha256:
            raise ValueError("training lineage attestation self-hash mismatch")
        return self


class IndependentHoldoutLineage(StrictModel):
    schema_version: Literal["sentinel.independent-holdout-lineage.v1"] = (
        "sentinel.independent-holdout-lineage.v1"
    )
    model_ref: str = Field(min_length=1, max_length=512)
    model_revision: str = Field(pattern=_DIGEST)
    checkpoint_approval_sha256: str = Field(pattern=_DIGEST)
    checkpoint_file_sha256: str = Field(pattern=_DIGEST)
    training_attestation_sha256: str = Field(pattern=_DIGEST)
    holdout_report_sha256: str = Field(pattern=_DIGEST)
    evaluation_dataset_sha256: str = Field(pattern=_DIGEST)
    evaluation_policy_sha256: str = Field(pattern=_DIGEST)
    case_count: int = Field(ge=_MIN_HOLDOUT_CASES)
    signer_key_fingerprint: str = Field(pattern=_DIGEST)
    attestation_sha256: str = Field(pattern=_DIGEST)
    reviewer_signature_b64: str = Field(min_length=1)

    @model_validator(mode="after")
    def self_hash_verifies(self) -> IndependentHoldoutLineage:
        if _digest(_unsigned(self, "reviewer_signature_b64")) != self.attestation_sha256:
            raise ValueError("HOLDOUT lineage attestation self-hash mismatch")
        return self


class ProductionCheckpointLineageVerification(StrictModel):
    schema_version: Literal["sentinel.production-checkpoint-lineage-verification.v1"] = (
        "sentinel.production-checkpoint-lineage-verification.v1"
    )
    status: Literal["signed_lineage_claims_verified"] = "signed_lineage_claims_verified"
    model_ref: str
    model_revision: str
    checkpoint_approval_sha256: str = Field(pattern=_DIGEST)
    checkpoint_file_sha256: str = Field(pattern=_DIGEST)
    training_run_id: str
    training_attestation_sha256: str = Field(pattern=_DIGEST)
    independent_holdout_attestation_sha256: str = Field(pattern=_DIGEST)
    training_execution_independently_verified: Literal[False] = False
    holdout_execution_independently_verified: Literal[False] = False
    production_deployment_authorized: Literal[False] = False


def verify_production_checkpoint_lineage(
    manifest: RankCheckpointManifest,
    *,
    checkpoint_path: str,
    approved_receipt: ApprovedRankCheckpointReceipt,
    binding: ProductionTargetBinding,
    provenance: ProductionTargetProvenance,
    trusted_owner_key: Ed25519PublicKey,
    training_state: MCoreTrainingState,
    training_loop: MCoreTrainingLoopResult,
    training_lineage: ProductionTrainingLineage,
    trusted_training_operator_key: Ed25519PublicKey,
    holdout_report: GoldHoldoutEvaluationReport,
    holdout_lineage: IndependentHoldoutLineage,
    trusted_reviewer_key: Ed25519PublicKey,
) -> ProductionCheckpointLineageVerification:
    """Audit signatures and cross-document identities; do not assert execution truth.

    Signed records remain claims by their issuers. External runtime receipts and
    independent evaluation are required before any release-readiness judgment.
    """
    owner_fingerprint = owner_key_fingerprint(trusted_owner_key)
    operator_fingerprint = owner_key_fingerprint(trusted_training_operator_key)
    reviewer_fingerprint = owner_key_fingerprint(trusted_reviewer_key)
    if reviewer_fingerprint in (owner_fingerprint, operator_fingerprint):
        raise ValueError("HOLDOUT reviewer must have an independent trust root")

    for record in (training_lineage, holdout_lineage):
        if (record.model_ref, record.model_revision) != (
            binding.model_ref,
            binding.model_revision,
        ):
            raise ValueError("lineage model identity does not match production target")
        if record.checkpoint_approval_sha256 != approved_receipt.receipt_sha256:
            raise ValueError("lineage checkpoint approval identity mismatch")
        if record.checkpoint_file_sha256 != approved_receipt.checkpoint_file_sha256:
            raise ValueError("lineage checkpoint file digest mismatch")

    _verify_signature(
        training_lineage,
        signature_field="operator_signature_b64",
        signature_domain=_TRAINING_DOMAIN,
        signer=trusted_training_operator_key,
    )
    _verify_signature(
        holdout_lineage,
        signature_field="reviewer_signature_b64",
        signature_domain=_HOLDOUT_DOMAIN,
        signer=trusted_reviewer_key,
    )
    if holdout_lineage.training_attestation_sha256 != training_lineage.attestation_sha256:
        raise ValueError("HOLDOUT evidence does not bind the training attestation")

    if training_loop.status != "training_loop_completed":
        raise ValueError("training loop completion evidence is required")
    if (
        training_loop.final_step != training_state.global_step
        or training_lineage.final_step != training_state.global_step
        or training_loop.step_results[-1].global_step != training_state.global_step
        or training_loop.step_results[-1].consumed_microbatches
        != training_state.consumed_microbatches
    ):
        raise ValueError("training state and loop final step do not agree")
    if training_loop.final_checkpoint_dir is None:
        raise ValueError("training loop result lacks a final checkpoint reference")
    if training_lineage.training_state_sha256 != _digest(
        training_state.model_dump(mode="json")
    ):
        raise ValueError("training state digest mismatch")
    if training_lineage.training_loop_sha256 != _digest(
        training_loop.model_dump(mode="json")
    ):
        raise ValueError("training loop digest mismatch")

    report_payload = holdout_report.model_dump(mode="json", exclude={"report_sha256"})
    if _digest(report_payload) != holdout_report.report_sha256:
        raise ValueError("HOLDOUT report self-hash mismatch")
    if (holdout_report.model_ref, holdout_report.model_revision) != (
        binding.model_ref, binding.model_revision
    ):
        raise ValueError("HOLDOUT report model identity mismatch")
    if holdout_lineage.holdout_report_sha256 != holdout_report.report_sha256:
        raise ValueError("HOLDOUT report does not match reviewer attestation")
    if (
        holdout_report.case_count < _MIN_HOLDOUT_CASES
        or holdout_report.case_count != holdout_lineage.case_count
        or holdout_report.prediction_count != holdout_report.case_count
        or len(holdout_report.case_results) != holdout_report.case_count
        or len({case.case_id for case in holdout_report.case_results})
        != holdout_report.case_count
        or holdout_report.missing_case_ids
        or holdout_report.extra_case_ids
        or any(case.violations for case in holdout_report.case_results)
        or holdout_report.violations
        or not holdout_report.passed
    ):
        raise ValueError("HOLDOUT case accounting or pass claim is incomplete")

    # Only the final gate may read checkpoint bytes; all signature and
    # cross-document checks above fail closed before deserialization.
    verify_approved_rank_checkpoint_load(
        manifest,
        checkpoint_path=checkpoint_path,
        receipt=approved_receipt,
        target_binding=binding,
        target_provenance=provenance,
        trusted_owner_public_key=trusted_owner_key,
    )
    return ProductionCheckpointLineageVerification(
        model_ref=binding.model_ref,
        model_revision=binding.model_revision,
        checkpoint_approval_sha256=approved_receipt.receipt_sha256,
        checkpoint_file_sha256=approved_receipt.checkpoint_file_sha256,
        training_run_id=training_lineage.training_run_id,
        training_attestation_sha256=training_lineage.attestation_sha256,
        independent_holdout_attestation_sha256=holdout_lineage.attestation_sha256,
    )
