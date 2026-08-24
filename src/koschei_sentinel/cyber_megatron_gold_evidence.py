from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.cyber_megatron_candidate import CyberMegatronCandidateManifest
from koschei_sentinel.cyber_megatron_holdout import CyberMegatronHoldoutPlan
from koschei_sentinel.cyber_megatron_holdout_worker import (
    EXECUTION_STATE_FILENAME,
    PREDICTIONS_FILENAME,
    RECEIPT_FILENAME,
    WORKER_PLAN_FILENAME,
    CyberMegatronHoldoutExecutionState,
    CyberMegatronHoldoutRunReceipt,
    CyberMegatronHoldoutWorkerPlan,
    verify_cyber_megatron_holdout_output,
)
from koschei_sentinel.cyber_megatron_training import (
    QWEN35_397B_MODEL,
    QWEN35_397B_REVISION,
)
from koschei_sentinel.defense_reflex_gold_release_audit import audit_gold_defense_release
from koschei_sentinel.gold_holdout_evaluation import (
    GoldHoldoutEvaluationPolicy,
    GoldHoldoutEvaluationReport,
    GoldHoldoutPrediction,
    evaluate_gold_holdout_predictions,
)
from koschei_sentinel.gold_holdout_zero_prediction import build_zero_prediction_gold_report
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import canonical_json

_DIGEST = r"^[a-f0-9]{64}$"
_PRODUCTION_MINIMUM_HOLDOUT_CASES = 50


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_canonical(payload: object) -> str:
    return _sha256_bytes(canonical_json(payload).encode("utf-8"))


def _digest_without(payload: dict[str, object], field_name: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field_name, None)
    return _sha256_canonical(unsigned)


def _read_regular_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {label}") from exc


def _load_canonical_model(path: Path, model_type, label: str):
    raw = _read_regular_bytes(path, label)
    try:
        model = model_type.model_validate_json(raw)
    except ValueError as exc:
        raise ValueError(f"{label} cannot be parsed") from exc
    expected = json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    if raw != expected.encode("utf-8"):
        raise ValueError(f"{label} is not canonical byte-for-byte")
    return model, raw


def _load_predictions(path: Path) -> list[GoldHoldoutPrediction]:
    raw = _read_regular_bytes(path, "397B HOLDOUT predictions")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("397B HOLDOUT predictions are not valid UTF-8") from exc
    rows: list[GoldHoldoutPrediction] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            prediction = GoldHoldoutPrediction.model_validate_json(line)
        except ValueError as exc:
            raise ValueError(
                f"invalid 397B HOLDOUT prediction at line {line_number}"
            ) from exc
        if prediction.case_id in seen:
            raise ValueError(f"duplicate 397B HOLDOUT prediction: {prediction.case_id}")
        seen.add(prediction.case_id)
        if prediction.model_ref != QWEN35_397B_MODEL:
            raise ValueError("397B HOLDOUT prediction model_ref differs from active target")
        if prediction.model_revision != QWEN35_397B_REVISION:
            raise ValueError("397B HOLDOUT prediction revision differs from pinned target")
        expected_prediction_sha = _digest_without(
            prediction.model_dump(mode="json"),
            "prediction_sha256",
        )
        if prediction.prediction_sha256 != expected_prediction_sha:
            raise ValueError(f"397B HOLDOUT prediction self-hash failed: {prediction.case_id}")
        rows.append(prediction)
    return sorted(rows, key=lambda row: row.case_id)


def _verify_report_digest(report: GoldHoldoutEvaluationReport) -> None:
    expected = _digest_without(report.model_dump(mode="json"), "report_sha256")
    if report.report_sha256 != expected:
        raise ValueError("397B Gold evaluation report self-hash does not verify")


class CyberMegatronGoldEvaluationEvidence(StrictModel):
    schema_version: Literal["sentinel.cyber-megatron-gold-evaluation-evidence.v1"] = (
        "sentinel.cyber-megatron-gold-evaluation-evidence.v1"
    )
    backend: Literal["megatron-swift-vllm"] = "megatron-swift-vllm"
    model: Literal["Qwen/Qwen3.5-397B-A17B"]
    model_revision: Literal["8472618112abcbd45acbcdc58436aff4233c23f7"]
    run_id: str
    candidate_manifest_sha256: str = Field(pattern=_DIGEST)
    candidate_sha256: str = Field(pattern=_DIGEST)
    checkpoint_tree_sha256: str = Field(pattern=_DIGEST)
    source_gold_audit_sha256: str = Field(pattern=_DIGEST)
    review_signature_audit_sha256: str = Field(pattern=_DIGEST)
    pack_signature_file_sha256: str = Field(pattern=_DIGEST)
    pack_proof_sha256: str = Field(pattern=_DIGEST)
    reviewer_trust_policy_file_sha256: str = Field(pattern=_DIGEST)
    reviewer_trust_policy_digest: str = Field(pattern=_DIGEST)
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    holdout_plan_file_sha256: str = Field(pattern=_DIGEST)
    holdout_plan_sha256: str = Field(pattern=_DIGEST)
    worker_plan_sha256: str = Field(pattern=_DIGEST)
    execution_state_sha256: str = Field(pattern=_DIGEST)
    inference_receipt_sha256: str = Field(pattern=_DIGEST)
    output_verification_sha256: str = Field(pattern=_DIGEST)
    generation_policy_sha256: str = Field(pattern=_DIGEST)
    evaluation_policy_sha256: str = Field(pattern=_DIGEST)
    minimum_case_count: int = Field(ge=1)
    case_count: int = Field(gt=0)
    prediction_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    output_verification_valid: Literal[True] = True
    complete_case_accounting: Literal[True] = True
    report: GoldHoldoutEvaluationReport
    passed: bool
    evidence_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def evidence_contract_verifies(self) -> CyberMegatronGoldEvaluationEvidence:
        if self.case_count < self.minimum_case_count:
            raise ValueError("397B Gold evidence case count is below its production floor")
        if self.prediction_count + self.failure_count != self.case_count:
            raise ValueError("397B Gold evidence does not account for every HOLDOUT case")
        if self.report.model_ref != self.model:
            raise ValueError("397B Gold report model_ref differs from evidence")
        if self.report.model_revision != self.model_revision:
            raise ValueError("397B Gold report model revision differs from evidence")
        if self.report.adapter_digest != self.checkpoint_tree_sha256:
            raise ValueError("397B Gold report identity must equal checkpoint tree SHA")
        if self.report.case_count != self.case_count:
            raise ValueError("397B Gold report case count differs from evidence")
        if self.report.prediction_count != self.prediction_count:
            raise ValueError("397B Gold report prediction count differs from evidence")
        expected_pass = self.report.passed and self.failure_count == 0
        if self.passed != expected_pass:
            raise ValueError(
                "397B Gold evidence passes only with a passing report and zero failures"
            )
        unsigned = self.model_dump(mode="json")
        observed = str(unsigned.pop("evidence_sha256"))
        if _sha256_canonical(unsigned) != observed:
            raise ValueError("397B Gold evidence self-hash does not verify")
        return self


def verify_cyber_megatron_gold_evidence(
    evidence: CyberMegatronGoldEvaluationEvidence,
) -> None:
    expected = _digest_without(evidence.model_dump(mode="json"), "evidence_sha256")
    if expected != evidence.evidence_sha256:
        raise ValueError("397B Gold evidence self-hash does not verify")
    _verify_report_digest(evidence.report)


def build_cyber_megatron_gold_evidence(
    *,
    release_dir: str | Path,
    worker_dir: str | Path,
    holdout_plan_path: str | Path,
    candidate_manifest_path: str | Path,
    candidate_config_path: str | Path,
    candidate_training_plan_path: str | Path,
    checkpoint_dir: str | Path,
    inference_pack: str | Path,
    signature_path: str | Path,
    reviewer_public_key_path: str | Path,
    reviewer_trust_policy_path: str | Path,
    owner_public_key_path: str | Path,
    policy: GoldHoldoutEvaluationPolicy | None = None,
    minimum_case_count: int = _PRODUCTION_MINIMUM_HOLDOUT_CASES,
    root: str | Path = ".",
) -> CyberMegatronGoldEvaluationEvidence:
    if minimum_case_count < _PRODUCTION_MINIMUM_HOLDOUT_CASES:
        raise ValueError("397B production Gold evidence requires at least 50 HOLDOUT cases")
    selected_policy = policy or GoldHoldoutEvaluationPolicy(
        minimum_case_count=minimum_case_count
    )
    if selected_policy.minimum_case_count < minimum_case_count:
        raise ValueError("397B evaluation policy cannot weaken the bound HOLDOUT case floor")

    root_path = Path(root).resolve()
    release = Path(release_dir)
    release_audit = audit_gold_defense_release(release)
    if not release_audit.valid:
        raise ValueError("397B Gold evidence requires a valid Gold release")

    verification = verify_cyber_megatron_holdout_output(
        root=root_path,
        worker_dir=worker_dir,
        holdout_plan_path=holdout_plan_path,
        candidate_manifest_path=candidate_manifest_path,
        candidate_config_path=candidate_config_path,
        candidate_training_plan_path=candidate_training_plan_path,
        checkpoint_dir=checkpoint_dir,
        inference_pack=inference_pack,
        signature_path=signature_path,
        reviewer_public_key_path=reviewer_public_key_path,
        reviewer_trust_policy_path=reviewer_trust_policy_path,
        owner_public_key_path=owner_public_key_path,
        minimum_case_count=minimum_case_count,
    )
    if not verification.valid:
        detail = "; ".join(verification.violations[:5])
        raise ValueError(
            "397B Gold evidence requires freshly verified HOLDOUT output"
            + (f": {detail}" if detail else "")
        )
    if not verification.complete_case_accounting:
        raise ValueError("397B Gold evidence requires complete case accounting")

    worker_root = (root_path / Path(worker_dir)).resolve()
    holdout_path = (root_path / Path(holdout_plan_path)).resolve()
    candidate_path = (root_path / Path(candidate_manifest_path)).resolve()
    holdout_plan, holdout_raw = _load_canonical_model(
        holdout_path,
        CyberMegatronHoldoutPlan,
        "397B HOLDOUT plan",
    )
    candidate, candidate_raw = _load_canonical_model(
        candidate_path,
        CyberMegatronCandidateManifest,
        "397B candidate manifest",
    )
    worker_plan, _worker_raw = _load_canonical_model(
        worker_root / WORKER_PLAN_FILENAME,
        CyberMegatronHoldoutWorkerPlan,
        "397B HOLDOUT worker plan",
    )
    execution, _execution_raw = _load_canonical_model(
        worker_root / EXECUTION_STATE_FILENAME,
        CyberMegatronHoldoutExecutionState,
        "397B HOLDOUT execution state",
    )
    receipt, _receipt_raw = _load_canonical_model(
        worker_root / RECEIPT_FILENAME,
        CyberMegatronHoldoutRunReceipt,
        "397B HOLDOUT receipt",
    )

    if holdout_plan.gold_release_audit_sha256 != release_audit.audit_sha256:
        raise ValueError("397B HOLDOUT plan and Gold release audit identities differ")
    if candidate.gold_release_audit_sha256 != release_audit.audit_sha256:
        raise ValueError("397B candidate and Gold release audit identities differ")
    if worker_plan.holdout_plan_sha256 != holdout_plan.plan_sha256:
        raise ValueError("397B worker plan does not bind the supplied HOLDOUT plan")
    if worker_plan.candidate_sha256 != candidate.candidate_sha256:
        raise ValueError("397B worker plan does not bind the supplied candidate")
    if receipt.checkpoint_tree_sha256 != candidate.checkpoint_tree_sha256:
        raise ValueError("397B receipt checkpoint identity differs from candidate")
    if execution.state_sha256 != receipt.execution_state_sha256:
        raise ValueError("397B receipt does not bind the supplied execution state")

    predictions = _load_predictions(worker_root / PREDICTIONS_FILENAME)
    if predictions:
        report = evaluate_gold_holdout_predictions(
            release,
            predictions,
            policy=selected_policy,
        )
    else:
        report = build_zero_prediction_gold_report(
            release,
            model_ref=QWEN35_397B_MODEL,
            model_revision=QWEN35_397B_REVISION,
            adapter_digest=candidate.checkpoint_tree_sha256,
            policy=selected_policy,
        )
    _verify_report_digest(report)

    if report.model_ref != QWEN35_397B_MODEL:
        raise ValueError("397B evaluation report has the wrong model_ref")
    if report.model_revision != QWEN35_397B_REVISION:
        raise ValueError("397B evaluation report has the wrong model revision")
    if report.adapter_digest != candidate.checkpoint_tree_sha256:
        raise ValueError("397B evaluation report has the wrong checkpoint identity")
    if report.case_count != verification.request_count:
        raise ValueError("397B evaluation report case count differs from verified worker output")
    if report.prediction_count != verification.prediction_count:
        raise ValueError(
            "397B evaluation report prediction count differs from verified worker output"
        )

    passed = report.passed and verification.failure_count == 0
    unsigned: dict[str, object] = {
        "schema_version": "sentinel.cyber-megatron-gold-evaluation-evidence.v1",
        "backend": "megatron-swift-vllm",
        "model": QWEN35_397B_MODEL,
        "model_revision": QWEN35_397B_REVISION,
        "run_id": holdout_plan.run_id,
        "candidate_manifest_sha256": _sha256_bytes(candidate_raw),
        "candidate_sha256": candidate.candidate_sha256,
        "checkpoint_tree_sha256": candidate.checkpoint_tree_sha256,
        "source_gold_audit_sha256": release_audit.audit_sha256,
        "review_signature_audit_sha256": holdout_plan.review_signature_audit_sha256,
        "pack_signature_file_sha256": holdout_plan.pack_signature_file_sha256,
        "pack_proof_sha256": holdout_plan.pack_proof_sha256,
        "reviewer_trust_policy_file_sha256": (
            holdout_plan.reviewer_trust_policy_file_sha256
        ),
        "reviewer_trust_policy_digest": holdout_plan.reviewer_trust_policy_digest,
        "owner_key_fingerprint": holdout_plan.owner_key_fingerprint,
        "holdout_plan_file_sha256": _sha256_bytes(holdout_raw),
        "holdout_plan_sha256": holdout_plan.plan_sha256,
        "worker_plan_sha256": worker_plan.plan_sha256,
        "execution_state_sha256": execution.state_sha256,
        "inference_receipt_sha256": receipt.receipt_sha256,
        "output_verification_sha256": verification.verification_sha256,
        "generation_policy_sha256": holdout_plan.generation_policy_sha256,
        "evaluation_policy_sha256": _sha256_canonical(
            selected_policy.model_dump(mode="json")
        ),
        "minimum_case_count": minimum_case_count,
        "case_count": verification.request_count,
        "prediction_count": verification.prediction_count,
        "failure_count": verification.failure_count,
        "output_verification_valid": True,
        "complete_case_accounting": True,
        "report": report.model_dump(mode="json"),
        "passed": passed,
    }
    evidence = CyberMegatronGoldEvaluationEvidence(
        **unsigned,
        evidence_sha256=_sha256_canonical(unsigned),
    )
    verify_cyber_megatron_gold_evidence(evidence)
    return evidence
