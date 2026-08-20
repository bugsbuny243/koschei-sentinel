from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Literal

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import Field, model_validator

from koschei_sentinel.cyber_sft_candidate_snapshot import (
    snapshot_verified_cyber_sft_export,
)
from koschei_sentinel.defense_reflex_gold_release_audit import audit_gold_defense_release
from koschei_sentinel.gold_candidate_training_binding import (
    verify_gold_candidate_training_binding,
)
from koschei_sentinel.gold_holdout_evaluation import (
    GoldHoldoutEvaluationPolicy,
    GoldHoldoutEvaluationReport,
    GoldHoldoutInferenceManifest,
    GoldHoldoutPrediction,
    evaluate_gold_holdout_predictions,
)
from koschei_sentinel.gold_holdout_inference_runner import (
    GoldHoldoutInferencePlan,
    GoldHoldoutInferenceRunReceipt,
)
from koschei_sentinel.gold_holdout_inference_verify import (
    GoldHoldoutInferenceVerification,
    verify_gold_holdout_inference_output,
)
from koschei_sentinel.gold_holdout_output_snapshot import (
    snapshot_verified_gold_holdout_inference_output,
)
from koschei_sentinel.gold_holdout_pack_admission import (
    GoldHoldoutPackAdmission,
    snapshot_admitted_gold_holdout_pack,
    verify_admitted_gold_holdout_pack,
)
from koschei_sentinel.gold_holdout_pack_preflight import (
    preflight_gold_holdout_inference_pack,
)
from koschei_sentinel.gold_holdout_pack_signing import GoldHoldoutPackSignatureProof
from koschei_sentinel.gold_holdout_zero_prediction import (
    build_zero_prediction_gold_report,
)
from koschei_sentinel.gold_release_snapshot import snapshot_verified_gold_release
from koschei_sentinel.gold_reviewer_trust import (
    GoldReviewerTrustPolicy,
    verify_gold_reviewer_trust_policy,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import canonical_json


class GoldHoldoutEvaluationEvidence(StrictModel):
    schema_version: Literal["sentinel.gold-holdout-evaluation-evidence.v1"] = (
        "sentinel.gold-holdout-evaluation-evidence.v1"
    )
    model_ref: str
    model_revision: str = Field(pattern=r"^[a-f0-9]{64}$")
    adapter_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_gold_audit_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    review_signature_audit_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    candidate_training_binding_verification_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    inference_pack_signature_proof_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    reviewer_trust_policy_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    owner_key_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    inference_inputs_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    inference_plan_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    inference_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    inference_verification_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    generation_policy_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    evaluation_policy_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    case_count: int = Field(gt=0)
    prediction_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    inference_verification_valid: Literal[True] = True
    complete_case_accounting: Literal[True] = True
    report: GoldHoldoutEvaluationReport
    passed: bool
    evidence_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def evidence_is_fail_closed(self) -> "GoldHoldoutEvaluationEvidence":
        if self.model_revision != self.adapter_digest:
            raise ValueError("Gold HOLDOUT model revision must equal the verified adapter digest")
        if self.report.model_ref != self.model_ref:
            raise ValueError("Gold HOLDOUT evidence report model_ref mismatch")
        if self.report.model_revision != self.model_revision:
            raise ValueError("Gold HOLDOUT evidence report model_revision mismatch")
        if self.report.adapter_digest != self.adapter_digest:
            raise ValueError("Gold HOLDOUT evidence report adapter digest mismatch")
        if self.report.case_count != self.case_count:
            raise ValueError("Gold HOLDOUT evidence report case count mismatch")
        if self.report.prediction_count != self.prediction_count:
            raise ValueError("Gold HOLDOUT evidence report prediction count mismatch")
        if (self.reviewer_trust_policy_sha256 is None) != (
            self.owner_key_fingerprint is None
        ):
            raise ValueError(
                "Gold HOLDOUT reviewer trust policy and owner fingerprint must be bound together"
            )
        expected = self.report.passed and self.failure_count == 0
        if self.passed != expected:
            raise ValueError(
                "Gold HOLDOUT evidence passes only when evaluation passes with zero inference failures"
            )
        return self


def _sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _digest_without(payload: dict[str, object], field_name: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field_name, None)
    if field_name == "evidence_sha256":
        # Preserve prior v1 digests for research/legacy evidence that predates
        # production signature, training, pack, or owner-trust binding.
        for optional_field in (
            "review_signature_audit_sha256",
            "candidate_training_binding_verification_sha256",
            "inference_pack_signature_proof_sha256",
            "reviewer_trust_policy_sha256",
            "owner_key_fingerprint",
        ):
            if unsigned.get(optional_field) is None:
                unsigned.pop(optional_field, None)
    return _sha256_text(canonical_json(unsigned))


def _policy_sha256(policy: GoldHoldoutEvaluationPolicy) -> str:
    return _sha256_text(canonical_json(policy.model_dump(mode="json")))


def _load_predictions(path: Path) -> list[GoldHoldoutPrediction]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read Gold HOLDOUT predictions: {exc}") from exc
    rows: list[GoldHoldoutPrediction] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(GoldHoldoutPrediction.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(
                f"invalid Gold HOLDOUT prediction at line {line_number}"
            ) from exc
    return rows


def _verify_evaluation_report_digest(report: GoldHoldoutEvaluationReport) -> None:
    expected = _digest_without(report.model_dump(mode="json"), "report_sha256")
    if expected != report.report_sha256:
        raise ValueError("Gold HOLDOUT evaluation report self-hash does not verify")


def _load_pack_evaluation_state(
    *,
    pack: str | Path,
    inference_output_dir: str | Path,
    candidate_export_dir: str | Path,
    source_gold_audit_sha256: str,
) -> tuple[GoldHoldoutInferenceManifest, GoldHoldoutInferenceVerification]:
    pack_path = Path(pack)
    preflight_gold_holdout_inference_pack(pack_path)
    inference_manifest = GoldHoldoutInferenceManifest.model_validate_json(
        (pack_path / "manifest.json").read_bytes()
    )
    if inference_manifest.source_gold_audit_sha256 != source_gold_audit_sha256:
        raise ValueError("Gold HOLDOUT inference pack belongs to a different release audit")
    verification = verify_gold_holdout_inference_output(
        inference_output_dir,
        pack_path,
        candidate_export_dir,
    )
    if not verification.valid:
        raise ValueError("Gold HOLDOUT inference output is not valid")
    return inference_manifest, verification


def _build_report(
    *,
    release_dir: str | Path,
    output_dir: str | Path,
    policy: GoldHoldoutEvaluationPolicy,
) -> tuple[GoldHoldoutInferencePlan, GoldHoldoutInferenceRunReceipt, GoldHoldoutEvaluationReport]:
    output = Path(output_dir)
    plan = GoldHoldoutInferencePlan.model_validate_json((output / "plan.json").read_bytes())
    receipt = GoldHoldoutInferenceRunReceipt.model_validate_json(
        (output / "receipt.json").read_bytes()
    )
    predictions = _load_predictions(output / "predictions.jsonl")
    if predictions:
        report = evaluate_gold_holdout_predictions(
            release_dir,
            predictions,
            policy=policy,
        )
    else:
        report = build_zero_prediction_gold_report(
            release_dir,
            model_ref=receipt.model_ref,
            model_revision=receipt.model_revision,
            adapter_digest=receipt.adapter_digest,
            policy=policy,
        )
    _verify_evaluation_report_digest(report)
    return plan, receipt, report


def build_gold_holdout_evaluation_evidence(
    *,
    release_dir: str | Path,
    inference_pack_dir: str | Path,
    inference_output_dir: str | Path,
    candidate_export_dir: str | Path,
    policy: GoldHoldoutEvaluationPolicy | None = None,
    reviewer_public_key: Ed25519PublicKey | None = None,
    inference_pack_signature_proof: GoldHoldoutPackSignatureProof | None = None,
) -> GoldHoldoutEvaluationEvidence:
    selected_policy = policy or GoldHoldoutEvaluationPolicy()
    review_signature_audit_sha: str | None = None
    candidate_training_binding_sha: str | None = None
    inference_pack_signature_sha: str | None = None
    pack = Path(inference_pack_dir)

    if reviewer_public_key is not None:
        if inference_pack_signature_proof is None:
            raise ValueError(
                "production Gold HOLDOUT evidence requires a signed inference-pack proof"
            )
        admission = GoldHoldoutPackAdmission(
            proof=inference_pack_signature_proof,
            reviewer_public_key=reviewer_public_key,
        )
        verify_admitted_gold_holdout_pack(admission, pack)

        with tempfile.TemporaryDirectory(prefix="gold-holdout-evidence-snapshot-") as temp_dir:
            snapshot_root = Path(temp_dir)
            release_snapshot, release_verification = snapshot_verified_gold_release(
                release_dir,
                snapshot_root / "release",
                reviewer_public_key=reviewer_public_key,
                expected_release_audit_sha256=(
                    inference_pack_signature_proof.source_gold_audit_sha256
                ),
                expected_review_signature_audit_sha256=(
                    inference_pack_signature_proof.review_signature_audit_sha256
                ),
            )
            release_audit = release_verification.release_audit
            signature_audit = release_verification.review_signature_audit
            review_signature_audit_sha = signature_audit.audit_sha256
            inference_pack_signature_sha = inference_pack_signature_proof.proof_sha256

            inference_snapshot = snapshot_admitted_gold_holdout_pack(
                admission,
                pack,
                snapshot_root / "pack",
            )
            candidate_snapshot = snapshot_verified_cyber_sft_export(
                candidate_export_dir,
                snapshot_root / "candidate-export",
            )
            output_snapshot, verification = snapshot_verified_gold_holdout_inference_output(
                inference_output_dir,
                snapshot_root / "output",
                inference_pack_dir=inference_snapshot,
                candidate_export_dir=candidate_snapshot,
            )
            preflight_gold_holdout_inference_pack(inference_snapshot)
            inference_manifest = GoldHoldoutInferenceManifest.model_validate_json(
                (inference_snapshot / "manifest.json").read_bytes()
            )
            if inference_manifest.source_gold_audit_sha256 != release_audit.audit_sha256:
                raise ValueError("Gold HOLDOUT inference pack belongs to a different release audit")

            candidate_binding = verify_gold_candidate_training_binding(
                release_snapshot,
                candidate_snapshot,
            )
            if not candidate_binding.valid:
                detail = "; ".join(candidate_binding.violations[:5])
                raise ValueError(
                    "Gold HOLDOUT candidate training binding failed"
                    + (f": {detail}" if detail else "")
                )
            candidate_training_binding_sha = candidate_binding.verification_sha256
            plan, receipt, report = _build_report(
                release_dir=release_snapshot,
                output_dir=output_snapshot,
                policy=selected_policy,
            )
    else:
        release_audit = audit_gold_defense_release(release_dir)
        if not release_audit.valid:
            raise ValueError("cannot build Gold HOLDOUT evidence from an invalid Gold release")
        with tempfile.TemporaryDirectory(prefix="gold-holdout-evidence-snapshot-") as temp_dir:
            candidate_snapshot = snapshot_verified_cyber_sft_export(
                candidate_export_dir,
                Path(temp_dir) / "candidate-export",
            )
            inference_manifest, verification = _load_pack_evaluation_state(
                pack=pack,
                inference_output_dir=inference_output_dir,
                candidate_export_dir=candidate_snapshot,
                source_gold_audit_sha256=release_audit.audit_sha256,
            )
        plan, receipt, report = _build_report(
            release_dir=release_dir,
            output_dir=inference_output_dir,
            policy=selected_policy,
        )

    identity = (report.model_ref, report.model_revision, report.adapter_digest)
    receipt_identity = (receipt.model_ref, receipt.model_revision, receipt.adapter_digest)
    if identity != receipt_identity:
        raise ValueError("Gold HOLDOUT report identity differs from inference receipt")
    if report.case_count != verification.case_count:
        raise ValueError("Gold HOLDOUT report case count differs from inference verification")
    if report.prediction_count != verification.prediction_count:
        raise ValueError("Gold HOLDOUT report prediction count differs from inference verification")

    passed = report.passed and verification.failure_count == 0
    payload: dict[str, object] = {
        "schema_version": "sentinel.gold-holdout-evaluation-evidence.v1",
        "model_ref": report.model_ref,
        "model_revision": report.model_revision,
        "adapter_digest": report.adapter_digest,
        "source_gold_audit_sha256": release_audit.audit_sha256,
        "review_signature_audit_sha256": review_signature_audit_sha,
        "candidate_training_binding_verification_sha256": candidate_training_binding_sha,
        "inference_pack_signature_proof_sha256": inference_pack_signature_sha,
        "inference_inputs_sha256": inference_manifest.inputs_sha256,
        "inference_plan_sha256": plan.plan_sha256,
        "inference_receipt_sha256": receipt.receipt_sha256,
        "inference_verification_sha256": verification.verification_sha256,
        "generation_policy_sha256": plan.generation_policy_sha256,
        "evaluation_policy_sha256": _policy_sha256(selected_policy),
        "case_count": verification.case_count,
        "prediction_count": verification.prediction_count,
        "failure_count": verification.failure_count,
        "inference_verification_valid": True,
        "complete_case_accounting": verification.complete_case_accounting,
        "report": report.model_dump(mode="json"),
        "passed": passed,
    }
    payload["evidence_sha256"] = _digest_without(payload, "evidence_sha256")
    return GoldHoldoutEvaluationEvidence.model_validate(payload)


def build_owner_trusted_gold_holdout_evaluation_evidence(
    *,
    release_dir: str | Path,
    inference_pack_dir: str | Path,
    inference_output_dir: str | Path,
    candidate_export_dir: str | Path,
    policy: GoldHoldoutEvaluationPolicy,
    reviewer_public_key: Ed25519PublicKey,
    reviewer_trust_policy: GoldReviewerTrustPolicy,
    owner_public_key: Ed25519PublicKey,
    inference_pack_signature_proof: GoldHoldoutPackSignatureProof,
) -> GoldHoldoutEvaluationEvidence:
    """Production evidence path anchored to the external owner public-key trust root."""
    verify_gold_reviewer_trust_policy(
        reviewer_trust_policy,
        reviewer_public_key,
        owner_public_key,
    )
    evidence = build_gold_holdout_evaluation_evidence(
        release_dir=release_dir,
        inference_pack_dir=inference_pack_dir,
        inference_output_dir=inference_output_dir,
        candidate_export_dir=candidate_export_dir,
        policy=policy,
        reviewer_public_key=reviewer_public_key,
        inference_pack_signature_proof=inference_pack_signature_proof,
    )
    payload = evidence.model_dump(mode="json")
    payload.pop("evidence_sha256")
    payload["reviewer_trust_policy_sha256"] = reviewer_trust_policy.policy_digest
    payload["owner_key_fingerprint"] = reviewer_trust_policy.owner_key_fingerprint
    payload["evidence_sha256"] = _digest_without(payload, "evidence_sha256")
    return GoldHoldoutEvaluationEvidence.model_validate(payload)


def verify_gold_holdout_evaluation_evidence(
    evidence: GoldHoldoutEvaluationEvidence,
) -> None:
    expected = _digest_without(evidence.model_dump(mode="json"), "evidence_sha256")
    if expected != evidence.evidence_sha256:
        raise ValueError("Gold HOLDOUT evaluation evidence self-hash does not verify")
    _verify_evaluation_report_digest(evidence.report)
