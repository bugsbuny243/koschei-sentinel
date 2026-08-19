from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.cyber_sft_export_verify import CyberSFTExportVerification
from koschei_sentinel.cyber_sft_run_attestation import (
    CyberSFTRunAttestation,
    _attestation_digest,
    _config_sha256,
)
from koschei_sentinel.cyber_sft_training import CyberSFTConfig
from koschei_sentinel.gold_holdout_evaluation import GoldHoldoutPrediction
from koschei_sentinel.gold_holdout_inference_runner import (
    GoldHoldoutGenerationPolicy,
    GoldHoldoutInferenceFailure,
    GoldHoldoutInferencePlan,
    GoldHoldoutInferenceRunReceipt,
    _digest_without,
    _export_verification_sha256,
    _load_candidate_identity,
    _load_inference_pack,
    _policy_sha256,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import canonical_json


class GoldHoldoutInferenceVerification(StrictModel):
    schema_version: Literal["sentinel.gold-holdout-inference-verification.v3"] = (
        "sentinel.gold-holdout-inference-verification.v3"
    )
    output_dir: str
    case_count: int = Field(ge=0)
    prediction_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    plan_verified: bool
    receipt_verified: bool
    input_binding_verified: bool
    generation_policy_verified: bool
    training_config_verified: bool
    run_attestation_verified: bool
    candidate_export_verification_verified: bool
    prediction_hashes_verified: bool
    identity_verified: bool
    complete_case_accounting: bool
    valid: bool
    violations: list[str]
    verification_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(payload: str) -> str:
    return _sha256_bytes(payload.encode("utf-8"))


def _verification_digest(payload: dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("verification_sha256", None)
    return _sha256_text(canonical_json(unsigned))


def _prediction_digest(prediction: GoldHoldoutPrediction) -> str:
    return _digest_without(prediction.model_dump(mode="json"), "prediction_sha256")


def _load_jsonl(path: Path, model_type, label: str):
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8") from exc
    rows = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(model_type.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"invalid {label} at line {line_number}") from exc
    return rows, raw


def verify_gold_holdout_inference_output(
    output_dir: str | Path,
    inference_pack_dir: str | Path,
    candidate_export_dir: str | Path,
) -> GoldHoldoutInferenceVerification:
    output = Path(output_dir)
    violations: list[str] = []
    case_count = 0
    prediction_count = 0
    failure_count = 0
    plan_verified = False
    receipt_verified = False
    input_binding_verified = False
    generation_policy_verified = False
    training_config_verified = False
    run_attestation_verified = False
    candidate_export_verification_verified = False
    prediction_hashes_verified = False
    identity_verified = False
    complete_case_accounting = False

    try:
        cases, inference_manifest, inference_manifest_raw = _load_inference_pack(
            inference_pack_dir
        )
        case_count = len(cases)
        case_by_id = {row.case_id: row for row in cases}
        plan = GoldHoldoutInferencePlan.model_validate_json(
            (output / "plan.json").read_bytes()
        )
        receipt = GoldHoldoutInferenceRunReceipt.model_validate_json(
            (output / "receipt.json").read_bytes()
        )
        generation_policy = GoldHoldoutGenerationPolicy.model_validate_json(
            (output / "generation-policy.json").read_bytes()
        )
        training_config = CyberSFTConfig.model_validate_json(
            (output / "training-config.json").read_bytes()
        )
        run_attestation = CyberSFTRunAttestation.model_validate_json(
            (output / "run-attestation.json").read_bytes()
        )
        candidate_export_verification = CyberSFTExportVerification.model_validate_json(
            (output / "candidate-export-verification.json").read_bytes()
        )
        (
            candidate_config,
            candidate_manifest,
            _candidate_adapter_path,
            candidate_attestation,
            fresh_export_verification,
        ) = _load_candidate_identity(candidate_export_dir=candidate_export_dir)
        predictions, prediction_raw = _load_jsonl(
            output / "predictions.jsonl",
            GoldHoldoutPrediction,
            "Gold HOLDOUT predictions",
        )
        failures, failure_raw = _load_jsonl(
            output / "failures.jsonl",
            GoldHoldoutInferenceFailure,
            "Gold HOLDOUT inference failures",
        )
        prediction_count = len(predictions)
        failure_count = len(failures)

        expected_plan_sha = _digest_without(
            plan.model_dump(mode="json"),
            "plan_sha256",
        )
        plan_verified = expected_plan_sha == plan.plan_sha256
        if not plan_verified:
            violations.append("Gold HOLDOUT inference plan self-hash does not verify")

        expected_receipt_sha = _digest_without(
            receipt.model_dump(mode="json"),
            "receipt_sha256",
        )
        receipt_verified = expected_receipt_sha == receipt.receipt_sha256
        if not receipt_verified:
            violations.append("Gold HOLDOUT inference receipt self-hash does not verify")

        generation_policy_verified = (
            _policy_sha256(generation_policy) == plan.generation_policy_sha256
        )
        if not generation_policy_verified:
            violations.append(
                "Gold HOLDOUT generation policy SHA differs from inference plan"
            )

        persisted_config_sha = _config_sha256(training_config)
        candidate_config_sha = _config_sha256(candidate_config)
        training_config_verified = (
            persisted_config_sha == plan.training_config_sha256
            and candidate_config_sha == plan.training_config_sha256
            and persisted_config_sha == run_attestation.config_sha256
            and candidate_config_sha == candidate_attestation.config_sha256
        )
        if not training_config_verified:
            violations.append(
                "Gold HOLDOUT training config SHA differs from "
                "plan/run attestation/candidate export"
            )

        persisted_attestation_self_hash = (
            _attestation_digest(run_attestation.model_dump(mode="json"))
            == run_attestation.attestation_sha256
        )
        persisted_attestation_identity = (
            run_attestation.attestation_sha256 == plan.run_attestation_sha256
            and run_attestation.run_id == plan.run_id
            and run_attestation.base_model == plan.base_model
            and run_attestation.base_revision == plan.base_revision
            and run_attestation.adapter_digest == plan.adapter_digest
            and run_attestation.promotion_eligible is True
            and run_attestation.smoke_only is False
        )
        candidate_attestation_identity = (
            candidate_attestation.attestation_sha256 == plan.run_attestation_sha256
            and candidate_attestation.run_id == plan.run_id
            and candidate_attestation.base_model == plan.base_model
            and candidate_attestation.base_revision == plan.base_revision
            and candidate_attestation.adapter_digest == plan.adapter_digest
            and candidate_attestation.promotion_eligible is True
            and candidate_attestation.smoke_only is False
        )
        run_attestation_verified = (
            persisted_attestation_self_hash
            and persisted_attestation_identity
            and candidate_attestation_identity
        )
        if not run_attestation_verified:
            violations.append(
                "Gold HOLDOUT run attestation does not bind the promoted candidate export"
            )

        persisted_export_sha = _export_verification_sha256(
            candidate_export_verification
        )
        fresh_export_sha = _export_verification_sha256(fresh_export_verification)
        candidate_identity_verified = (
            candidate_config.run_id == plan.run_id
            and candidate_config.base_model == plan.base_model
            and candidate_config.base_revision == plan.base_revision
            and candidate_manifest.adapter_digest == plan.adapter_digest
        )
        candidate_export_verification_verified = (
            candidate_export_verification.valid
            and not candidate_export_verification.violations
            and fresh_export_verification.valid
            and not fresh_export_verification.violations
            and candidate_export_verification.run_id == plan.run_id
            and fresh_export_verification.run_id == plan.run_id
            and persisted_export_sha == plan.candidate_export_verification_sha256
            and fresh_export_sha == plan.candidate_export_verification_sha256
            and candidate_identity_verified
        )
        if not candidate_export_verification_verified:
            violations.append(
                "Gold HOLDOUT candidate export does not independently verify "
                "against inference plan"
            )

        input_checks = (
            ("case_count", plan.case_count, case_count),
            ("inputs_sha256", plan.inputs_sha256, inference_manifest.inputs_sha256),
            (
                "inference_manifest_sha256",
                plan.inference_manifest_sha256,
                _sha256_bytes(inference_manifest_raw),
            ),
        )
        input_mismatches = [
            label for label, observed, expected in input_checks if observed != expected
        ]
        if input_mismatches:
            violations.append(
                "Gold HOLDOUT inference plan differs from input pack: "
                + ", ".join(input_mismatches)
            )
        else:
            input_binding_verified = True

        receipt_checks = (
            ("plan_sha256", receipt.plan_sha256, plan.plan_sha256),
            ("run_id", receipt.run_id, plan.run_id),
            ("model_ref", receipt.model_ref, plan.model_ref),
            ("model_revision", receipt.model_revision, plan.model_revision),
            ("adapter_digest", receipt.adapter_digest, plan.adapter_digest),
            ("base_model", receipt.base_model, plan.base_model),
            ("base_revision", receipt.base_revision, plan.base_revision),
            (
                "training_config_sha256",
                receipt.training_config_sha256,
                plan.training_config_sha256,
            ),
            (
                "run_attestation_sha256",
                receipt.run_attestation_sha256,
                plan.run_attestation_sha256,
            ),
            (
                "candidate_export_verification_sha256",
                receipt.candidate_export_verification_sha256,
                plan.candidate_export_verification_sha256,
            ),
            ("case_count", receipt.case_count, case_count),
            ("prediction_count", receipt.prediction_count, prediction_count),
            ("failure_count", receipt.failure_count, failure_count),
            (
                "predictions_sha256",
                receipt.predictions_sha256,
                _sha256_bytes(prediction_raw),
            ),
            (
                "failures_sha256",
                receipt.failures_sha256,
                _sha256_bytes(failure_raw),
            ),
        )
        receipt_mismatches = [
            label for label, observed, expected in receipt_checks if observed != expected
        ]
        if receipt_mismatches:
            violations.append(
                "Gold HOLDOUT inference receipt binding mismatch: "
                + ", ".join(receipt_mismatches)
            )

        prediction_ids: set[str] = set()
        prediction_hash_failures: list[str] = []
        identity_failures: list[str] = []
        context_failures: list[str] = []
        for prediction in predictions:
            if prediction.case_id in prediction_ids:
                violations.append(
                    f"duplicate Gold HOLDOUT prediction: {prediction.case_id}"
                )
            prediction_ids.add(prediction.case_id)
            if _prediction_digest(prediction) != prediction.prediction_sha256:
                prediction_hash_failures.append(prediction.case_id)
            if (
                prediction.model_ref != plan.model_ref
                or prediction.model_revision != plan.model_revision
                or prediction.adapter_digest != plan.adapter_digest
            ):
                identity_failures.append(prediction.case_id)
            case = case_by_id.get(prediction.case_id)
            if case is None or prediction.scenario_id != case.scenario_id:
                context_failures.append(prediction.case_id)
            elif prediction.input_context_sha256 != case.input_context_sha256:
                context_failures.append(prediction.case_id)

        if prediction_hash_failures:
            violations.append(
                "prediction self-hash failures: "
                + ", ".join(sorted(prediction_hash_failures))
            )
        else:
            prediction_hashes_verified = True
        if identity_failures:
            violations.append(
                "prediction model/adapter identity mismatches: "
                + ", ".join(sorted(identity_failures))
            )
        else:
            identity_verified = True
        if context_failures:
            violations.append(
                "prediction input-context binding mismatches: "
                + ", ".join(sorted(context_failures))
            )

        failure_ids: set[str] = set()
        failure_context_mismatches: list[str] = []
        for failure in failures:
            if failure.case_id in failure_ids:
                violations.append(
                    f"duplicate Gold HOLDOUT inference failure: {failure.case_id}"
                )
            failure_ids.add(failure.case_id)
            case = case_by_id.get(failure.case_id)
            if case is None or failure.scenario_id != case.scenario_id:
                failure_context_mismatches.append(failure.case_id)
            elif failure.input_context_sha256 != case.input_context_sha256:
                failure_context_mismatches.append(failure.case_id)
        if failure_context_mismatches:
            violations.append(
                "failure input-context binding mismatches: "
                + ", ".join(sorted(failure_context_mismatches))
            )

        overlap = sorted(prediction_ids & failure_ids)
        missing = sorted(set(case_by_id) - prediction_ids - failure_ids)
        extra = sorted((prediction_ids | failure_ids) - set(case_by_id))
        if overlap:
            violations.append(
                "cases appear in both predictions and failures: " + ", ".join(overlap)
            )
        if missing:
            violations.append("inference output omits cases: " + ", ".join(missing))
        if extra:
            violations.append(
                "inference output contains unknown cases: " + ", ".join(extra)
            )
        expected_failed_ids = sorted(failure_ids)
        if receipt.failed_case_ids != expected_failed_ids:
            violations.append("receipt failed_case_ids differs from failures.jsonl")
        complete_case_accounting = not overlap and not missing and not extra
    except (OSError, TypeError, ValueError) as exc:
        violations.append(str(exc))

    valid = (
        not violations
        and plan_verified
        and receipt_verified
        and input_binding_verified
        and generation_policy_verified
        and training_config_verified
        and run_attestation_verified
        and candidate_export_verification_verified
        and prediction_hashes_verified
        and identity_verified
        and complete_case_accounting
        and case_count == prediction_count + failure_count
    )
    payload: dict[str, object] = {
        "schema_version": "sentinel.gold-holdout-inference-verification.v3",
        "output_dir": str(output_dir),
        "case_count": case_count,
        "prediction_count": prediction_count,
        "failure_count": failure_count,
        "plan_verified": plan_verified,
        "receipt_verified": receipt_verified,
        "input_binding_verified": input_binding_verified,
        "generation_policy_verified": generation_policy_verified,
        "training_config_verified": training_config_verified,
        "run_attestation_verified": run_attestation_verified,
        "candidate_export_verification_verified": candidate_export_verification_verified,
        "prediction_hashes_verified": prediction_hashes_verified,
        "identity_verified": identity_verified,
        "complete_case_accounting": complete_case_accounting,
        "valid": valid,
        "violations": violations,
    }
    payload["verification_sha256"] = _verification_digest(payload)
    return GoldHoldoutInferenceVerification.model_validate(payload)
