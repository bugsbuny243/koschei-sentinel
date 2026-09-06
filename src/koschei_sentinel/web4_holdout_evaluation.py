from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import canonical_json
from koschei_sentinel.web4_benchmark_intake import Web4BenchmarkAnswerKey
from koschei_sentinel.web4_holdout_release import (
    Web4HoldoutRelease,
    Web4HoldoutReleaseCase,
    verify_web4_holdout_release,
)

_DIGEST = r"^[a-f0-9]{64}$"


class Web4HoldoutInferenceCase(StrictModel):
    schema_version: Literal["sentinel.web4-holdout-inference-case.v1"] = (
        "sentinel.web4-holdout-inference-case.v1"
    )
    case_id: str = Field(min_length=3, max_length=256)
    family: str = Field(min_length=3, max_length=256)
    release_case_sha256: str = Field(pattern=_DIGEST)
    model_input: dict[str, object]
    model_input_sha256: str = Field(pattern=_DIGEST)


class Web4HoldoutInferencePack(StrictModel):
    schema_version: Literal["sentinel.web4-holdout-inference-pack.v1"] = (
        "sentinel.web4-holdout-inference-pack.v1"
    )
    release_id: str
    release_sha256: str = Field(pattern=_DIGEST)
    release_artifact_sha256: str = Field(pattern=_DIGEST)
    benchmark_policy_sha256: str = Field(pattern=_DIGEST)
    case_count: int = Field(gt=0)
    case_ids: list[str] = Field(min_length=1)
    cases: list[Web4HoldoutInferenceCase] = Field(min_length=1)
    answer_key_excluded: Literal[True] = True
    research_evaluation_authorization: Literal[True] = True
    training_authorization: Literal[False] = False
    promotion_eligible: Literal[False] = False
    production_activation_allowed: Literal[False] = False
    network_access_required: Literal[False] = False
    gpu_required: Literal[False] = False
    pack_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def pack_contract_verifies(self) -> Web4HoldoutInferencePack:
        if self.case_count != len(self.cases):
            raise ValueError("Web4 inference pack case_count differs from cases")
        if self.case_ids != sorted(self.case_ids):
            raise ValueError("Web4 inference pack case_ids must be sorted")
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("Web4 inference pack contains duplicate case IDs")
        if self.case_ids != [case.case_id for case in self.cases]:
            raise ValueError("Web4 inference pack case_ids differ from case records")
        payload = self.model_dump(mode="json")
        observed = str(payload.pop("pack_sha256"))
        if _digest(payload) != observed:
            raise ValueError("Web4 inference pack self-hash does not verify")
        return self


class Web4HoldoutPrediction(StrictModel):
    schema_version: Literal["sentinel.web4-holdout-prediction.v1"] = (
        "sentinel.web4-holdout-prediction.v1"
    )
    case_id: str = Field(min_length=3, max_length=256)
    release_sha256: str = Field(pattern=_DIGEST)
    release_case_sha256: str = Field(pattern=_DIGEST)
    model_input_sha256: str = Field(pattern=_DIGEST)
    model_ref: str = Field(min_length=3, max_length=512)
    model_revision: str = Field(min_length=1, max_length=256)
    model_artifact_sha256: str = Field(pattern=_DIGEST)
    answer: dict[str, object]
    prediction_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def prediction_self_hash_verifies(self) -> Web4HoldoutPrediction:
        if not self.answer:
            raise ValueError("Web4 HOLDOUT prediction answer must not be empty")
        payload = self.model_dump(mode="json")
        observed = str(payload.pop("prediction_sha256"))
        if _digest(payload) != observed:
            raise ValueError("Web4 HOLDOUT prediction self-hash does not verify")
        return self


class Web4HoldoutPredictionSet(StrictModel):
    schema_version: Literal["sentinel.web4-holdout-prediction-set.v1"] = (
        "sentinel.web4-holdout-prediction-set.v1"
    )
    release_sha256: str = Field(pattern=_DIGEST)
    release_artifact_sha256: str = Field(pattern=_DIGEST)
    inference_pack_sha256: str = Field(pattern=_DIGEST)
    model_ref: str = Field(min_length=3, max_length=512)
    model_revision: str = Field(min_length=1, max_length=256)
    model_artifact_sha256: str = Field(pattern=_DIGEST)
    prediction_count: int = Field(ge=0)
    case_ids: list[str]
    predictions: list[Web4HoldoutPrediction]
    offline_replay: Literal[True] = True
    network_access_required: Literal[False] = False
    gpu_required: Literal[False] = False
    prediction_set_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def prediction_set_contract_verifies(self) -> Web4HoldoutPredictionSet:
        if self.prediction_count != len(self.predictions):
            raise ValueError("Web4 prediction-set count differs from predictions")
        if self.case_ids != sorted(self.case_ids):
            raise ValueError("Web4 prediction-set case_ids must be sorted")
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("Web4 prediction set contains duplicate case IDs")
        if self.case_ids != [row.case_id for row in self.predictions]:
            raise ValueError("Web4 prediction-set case_ids differ from predictions")
        identity = {
            (row.model_ref, row.model_revision, row.model_artifact_sha256)
            for row in self.predictions
        }
        if self.predictions and identity != {
            (self.model_ref, self.model_revision, self.model_artifact_sha256)
        }:
            raise ValueError("Web4 predictions do not share one model identity")
        if any(row.release_sha256 != self.release_sha256 for row in self.predictions):
            raise ValueError("Web4 prediction belongs to a different HOLDOUT release")
        payload = self.model_dump(mode="json")
        observed = str(payload.pop("prediction_set_sha256"))
        if _digest(payload) != observed:
            raise ValueError("Web4 prediction-set self-hash does not verify")
        return self


class Web4HoldoutEvaluationPolicy(StrictModel):
    schema_version: Literal["sentinel.web4-holdout-evaluation-policy.v1"] = (
        "sentinel.web4-holdout-evaluation-policy.v1"
    )
    minimum_case_count: int = Field(default=1, ge=1, le=100000)
    minimum_exact_match_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    minimum_expected_field_accuracy: float = Field(default=1.0, ge=0.0, le=1.0)
    require_complete_case_accounting: Literal[True] = True
    reject_unexpected_fields: Literal[True] = True
    answer_key_values_may_be_embedded_in_report: Literal[False] = False


class Web4HoldoutCaseResult(StrictModel):
    case_id: str
    family: str
    exact_match: bool
    expected_field_count: int = Field(ge=0)
    matched_field_count: int = Field(ge=0)
    missing_field_paths: list[str]
    mismatched_field_paths: list[str]
    unexpected_field_paths: list[str]


class Web4HoldoutEvaluationReport(StrictModel):
    schema_version: Literal["sentinel.web4-holdout-evaluation-report.v1"] = (
        "sentinel.web4-holdout-evaluation-report.v1"
    )
    release_sha256: str = Field(pattern=_DIGEST)
    release_artifact_sha256: str = Field(pattern=_DIGEST)
    inference_pack_sha256: str = Field(pattern=_DIGEST)
    prediction_set_sha256: str = Field(pattern=_DIGEST)
    evaluation_policy_sha256: str = Field(pattern=_DIGEST)
    model_ref: str
    model_revision: str
    model_artifact_sha256: str = Field(pattern=_DIGEST)
    case_count: int = Field(gt=0)
    prediction_count: int = Field(ge=0)
    answer_key_count: int = Field(ge=0)
    missing_case_ids: list[str]
    extra_case_ids: list[str]
    exact_match_cases: int = Field(ge=0)
    exact_match_rate: float = Field(ge=0.0, le=1.0)
    expected_field_count: int = Field(ge=0)
    matched_field_count: int = Field(ge=0)
    expected_field_accuracy: float = Field(ge=0.0, le=1.0)
    case_results: list[Web4HoldoutCaseResult]
    answer_key_values_embedded: Literal[False] = False
    complete_case_accounting: bool
    passed: bool
    violations: list[str]
    report_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def report_self_hash_verifies(self) -> Web4HoldoutEvaluationReport:
        payload = self.model_dump(mode="json")
        observed = str(payload.pop("report_sha256"))
        if _digest(payload) != observed:
            raise ValueError("Web4 HOLDOUT evaluation report self-hash does not verify")
        expected_accounting = not self.missing_case_ids and not self.extra_case_ids
        if self.complete_case_accounting is not expected_accounting:
            raise ValueError("Web4 HOLDOUT complete-case accounting is inconsistent")
        return self


class Web4HoldoutEvaluationEvidence(StrictModel):
    schema_version: Literal["sentinel.web4-holdout-evaluation-evidence.v1"] = (
        "sentinel.web4-holdout-evaluation-evidence.v1"
    )
    release_id: str
    release_sha256: str = Field(pattern=_DIGEST)
    release_artifact_sha256: str = Field(pattern=_DIGEST)
    release_owner_key_fingerprint: str = Field(pattern=_DIGEST)
    inference_pack_sha256: str = Field(pattern=_DIGEST)
    prediction_set_sha256: str = Field(pattern=_DIGEST)
    answer_key_bundle_sha256: str = Field(pattern=_DIGEST)
    evaluation_policy_sha256: str = Field(pattern=_DIGEST)
    model_ref: str
    model_revision: str
    model_artifact_sha256: str = Field(pattern=_DIGEST)
    case_count: int = Field(gt=0)
    prediction_count: int = Field(ge=0)
    answer_key_count: int = Field(ge=0)
    complete_case_accounting: bool
    answer_key_values_embedded: Literal[False] = False
    offline_replay: Literal[True] = True
    network_access_required: Literal[False] = False
    gpu_required: Literal[False] = False
    research_evaluation_only: Literal[True] = True
    training_authorization: Literal[False] = False
    promotion_eligible: Literal[False] = False
    production_activation_allowed: Literal[False] = False
    report: Web4HoldoutEvaluationReport
    passed: bool
    evidence_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def evidence_contract_verifies(self) -> Web4HoldoutEvaluationEvidence:
        if self.report.release_sha256 != self.release_sha256:
            raise ValueError("Web4 evaluation evidence report release mismatch")
        if self.report.release_artifact_sha256 != self.release_artifact_sha256:
            raise ValueError("Web4 evaluation evidence report release artifact mismatch")
        if self.report.inference_pack_sha256 != self.inference_pack_sha256:
            raise ValueError("Web4 evaluation evidence inference-pack mismatch")
        if self.report.prediction_set_sha256 != self.prediction_set_sha256:
            raise ValueError("Web4 evaluation evidence prediction-set mismatch")
        if self.report.evaluation_policy_sha256 != self.evaluation_policy_sha256:
            raise ValueError("Web4 evaluation evidence policy mismatch")
        if self.report.case_count != self.case_count:
            raise ValueError("Web4 evaluation evidence case count mismatch")
        if self.report.prediction_count != self.prediction_count:
            raise ValueError("Web4 evaluation evidence prediction count mismatch")
        if self.report.answer_key_count != self.answer_key_count:
            raise ValueError("Web4 evaluation evidence answer-key count mismatch")
        if self.complete_case_accounting is not self.report.complete_case_accounting:
            raise ValueError("Web4 evaluation evidence accounting mismatch")
        if self.passed is not self.report.passed:
            raise ValueError("Web4 evaluation evidence pass state differs from report")
        payload = self.model_dump(mode="json")
        observed = str(payload.pop("evidence_sha256"))
        if _digest(payload) != observed:
            raise ValueError("Web4 HOLDOUT evaluation evidence self-hash does not verify")
        return self


def _digest(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _policy_sha256(policy: Web4HoldoutEvaluationPolicy) -> str:
    return _digest(policy.model_dump(mode="json"))


def build_web4_holdout_inference_pack(
    *,
    release: Web4HoldoutRelease,
    owner_public_key: Ed25519PublicKey,
    benchmark_policy_path: str | Path,
) -> Web4HoldoutInferencePack:
    verified = verify_web4_holdout_release(
        release,
        owner_public_key=owner_public_key,
        benchmark_policy_path=benchmark_policy_path,
    )
    cases = [
        Web4HoldoutInferenceCase(
            case_id=case.case_id,
            family=case.family,
            release_case_sha256=case.case_sha256,
            model_input=case.model_input,
            model_input_sha256=case.model_input_sha256,
        )
        for case in verified.cases
    ]
    cases.sort(key=lambda row: row.case_id)
    payload: dict[str, object] = {
        "schema_version": "sentinel.web4-holdout-inference-pack.v1",
        "release_id": verified.release_id,
        "release_sha256": verified.release_sha256,
        "release_artifact_sha256": verified.artifact_sha256,
        "benchmark_policy_sha256": verified.benchmark_policy_sha256,
        "case_count": len(cases),
        "case_ids": [case.case_id for case in cases],
        "cases": [case.model_dump(mode="json") for case in cases],
        "answer_key_excluded": True,
        "research_evaluation_authorization": True,
        "training_authorization": False,
        "promotion_eligible": False,
        "production_activation_allowed": False,
        "network_access_required": False,
        "gpu_required": False,
    }
    payload["pack_sha256"] = _digest(payload)
    return Web4HoldoutInferencePack.model_validate(payload)


def build_web4_holdout_prediction(
    *,
    inference_case: Web4HoldoutInferenceCase,
    release_sha256: str,
    model_ref: str,
    model_revision: str,
    model_artifact_sha256: str,
    answer: dict[str, object],
) -> Web4HoldoutPrediction:
    payload: dict[str, object] = {
        "schema_version": "sentinel.web4-holdout-prediction.v1",
        "case_id": inference_case.case_id,
        "release_sha256": release_sha256,
        "release_case_sha256": inference_case.release_case_sha256,
        "model_input_sha256": inference_case.model_input_sha256,
        "model_ref": model_ref,
        "model_revision": model_revision,
        "model_artifact_sha256": model_artifact_sha256,
        "answer": answer,
    }
    payload["prediction_sha256"] = _digest(payload)
    return Web4HoldoutPrediction.model_validate(payload)


def build_web4_holdout_prediction_set(
    *,
    inference_pack: Web4HoldoutInferencePack,
    predictions: list[Web4HoldoutPrediction],
    model_ref: str,
    model_revision: str,
    model_artifact_sha256: str,
) -> Web4HoldoutPredictionSet:
    pack = Web4HoldoutInferencePack.model_validate(inference_pack.model_dump(mode="json"))
    rows = sorted(predictions, key=lambda row: row.case_id)
    if len({row.case_id for row in rows}) != len(rows):
        raise ValueError("Web4 prediction set contains duplicate case IDs")
    pack_cases = {row.case_id: row for row in pack.cases}
    for row in rows:
        if row.case_id in pack_cases:
            expected = pack_cases[row.case_id]
            if row.release_case_sha256 != expected.release_case_sha256:
                raise ValueError("Web4 prediction release-case digest differs from inference pack")
            if row.model_input_sha256 != expected.model_input_sha256:
                raise ValueError("Web4 prediction input digest differs from inference pack")
        if row.release_sha256 != pack.release_sha256:
            raise ValueError("Web4 prediction belongs to a different release")
        identity = (row.model_ref, row.model_revision, row.model_artifact_sha256)
        if identity != (model_ref, model_revision, model_artifact_sha256):
            raise ValueError("Web4 prediction model identity differs from prediction set")
    payload: dict[str, object] = {
        "schema_version": "sentinel.web4-holdout-prediction-set.v1",
        "release_sha256": pack.release_sha256,
        "release_artifact_sha256": pack.release_artifact_sha256,
        "inference_pack_sha256": pack.pack_sha256,
        "model_ref": model_ref,
        "model_revision": model_revision,
        "model_artifact_sha256": model_artifact_sha256,
        "prediction_count": len(rows),
        "case_ids": [row.case_id for row in rows],
        "predictions": [row.model_dump(mode="json") for row in rows],
        "offline_replay": True,
        "network_access_required": False,
        "gpu_required": False,
    }
    payload["prediction_set_sha256"] = _digest(payload)
    return Web4HoldoutPredictionSet.model_validate(payload)


def _load_answer_keys(
    answer_key_dir: str | Path,
    release: Web4HoldoutRelease,
) -> tuple[dict[str, Web4BenchmarkAnswerKey], str]:
    root = Path(answer_key_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Web4 answer-key root must be a regular non-symlink directory")
    expected_names = {f"{case.case_id}.json" for case in release.cases}
    observed_paths = [path for path in root.iterdir() if path.name != ".gitkeep"]
    if any(path.is_symlink() or not path.is_file() for path in observed_paths):
        raise ValueError("Web4 answer-key directory contains a non-regular file")
    observed_names = {path.name for path in observed_paths}
    if observed_names != expected_names:
        raise ValueError("Web4 answer-key directory file set differs from signed release")

    release_cases = {case.case_id: case for case in release.cases}
    rows: dict[str, Web4BenchmarkAnswerKey] = {}
    identities: dict[str, str] = {}
    for case_id in sorted(release_cases):
        path = root / f"{case_id}.json"
        raw = path.read_bytes()
        observed_sha = hashlib.sha256(raw).hexdigest()
        signed_case = release_cases[case_id]
        if observed_sha != signed_case.answer_key_sha256:
            raise ValueError(f"Web4 answer key drifted after signed release: {case_id}")
        try:
            answer_key = Web4BenchmarkAnswerKey.model_validate_json(raw)
        except ValueError as exc:
            raise ValueError(f"invalid Web4 answer key: {case_id}") from exc
        if answer_key.case_id != case_id:
            raise ValueError("Web4 answer-key case_id differs from file identity")
        if set(answer_key.source_refs) != set(signed_case.source_refs):
            raise ValueError("Web4 answer-key source refs differ from signed release")
        if not answer_key.expected:
            raise ValueError("Web4 answer-key expected object must not be empty")
        rows[case_id] = answer_key
        identities[case_id] = observed_sha
    return rows, _digest(identities)


def _flatten(value: object, path: str = "$") -> dict[str, object]:
    leaves: dict[str, object] = {}
    if isinstance(value, dict):
        if not value:
            leaves[path] = {}
        for key in sorted(value):
            leaves.update(_flatten(value[key], f"{path}.{key}"))
        return leaves
    if isinstance(value, list):
        if not value:
            leaves[path] = []
        for index, item in enumerate(value):
            leaves.update(_flatten(item, f"{path}[{index}]"))
        return leaves
    leaves[path] = value
    return leaves


def _case_result(
    *,
    release_case: Web4HoldoutReleaseCase,
    prediction: Web4HoldoutPrediction,
    answer_key: Web4BenchmarkAnswerKey,
) -> Web4HoldoutCaseResult:
    expected = _flatten(answer_key.expected)
    predicted = _flatten(prediction.answer)
    expected_paths = set(expected)
    predicted_paths = set(predicted)
    missing = sorted(expected_paths - predicted_paths)
    unexpected = sorted(predicted_paths - expected_paths)
    common = expected_paths & predicted_paths
    mismatched = sorted(path for path in common if predicted[path] != expected[path])
    matched_count = len(common) - len(mismatched)
    exact = not missing and not unexpected and not mismatched
    return Web4HoldoutCaseResult(
        case_id=release_case.case_id,
        family=release_case.family,
        exact_match=exact,
        expected_field_count=len(expected),
        matched_field_count=matched_count,
        missing_field_paths=missing,
        mismatched_field_paths=mismatched,
        unexpected_field_paths=unexpected,
    )


def evaluate_web4_holdout_predictions(
    *,
    release: Web4HoldoutRelease,
    inference_pack: Web4HoldoutInferencePack,
    prediction_set: Web4HoldoutPredictionSet,
    answer_key_dir: str | Path,
    owner_public_key: Ed25519PublicKey,
    benchmark_policy_path: str | Path,
    policy: Web4HoldoutEvaluationPolicy | None = None,
) -> tuple[Web4HoldoutEvaluationReport, str]:
    signed_release = verify_web4_holdout_release(
        release,
        owner_public_key=owner_public_key,
        benchmark_policy_path=benchmark_policy_path,
    )
    pack = Web4HoldoutInferencePack.model_validate(inference_pack.model_dump(mode="json"))
    predictions = Web4HoldoutPredictionSet.model_validate(
        prediction_set.model_dump(mode="json")
    )
    selected_policy = policy or Web4HoldoutEvaluationPolicy()
    policy_sha = _policy_sha256(selected_policy)

    if pack.release_sha256 != signed_release.release_sha256:
        raise ValueError("Web4 inference pack belongs to a different signed release")
    if pack.release_artifact_sha256 != signed_release.artifact_sha256:
        raise ValueError("Web4 inference pack release artifact differs")
    if pack.benchmark_policy_sha256 != signed_release.benchmark_policy_sha256:
        raise ValueError("Web4 inference pack benchmark policy differs")
    if predictions.release_sha256 != signed_release.release_sha256:
        raise ValueError("Web4 prediction set belongs to a different signed release")
    if predictions.release_artifact_sha256 != signed_release.artifact_sha256:
        raise ValueError("Web4 prediction-set release artifact differs")
    if predictions.inference_pack_sha256 != pack.pack_sha256:
        raise ValueError("Web4 prediction set belongs to a different inference pack")

    answer_keys, answer_bundle_sha = _load_answer_keys(answer_key_dir, signed_release)
    release_cases = {case.case_id: case for case in signed_release.cases}
    pack_cases = {case.case_id: case for case in pack.cases}
    prediction_by_id = {row.case_id: row for row in predictions.predictions}
    if len(prediction_by_id) != len(predictions.predictions):
        raise ValueError("Web4 prediction set contains duplicate case IDs")

    expected_ids = set(release_cases)
    predicted_ids = set(prediction_by_id)
    missing_cases = sorted(expected_ids - predicted_ids)
    extra_cases = sorted(predicted_ids - expected_ids)
    results: list[Web4HoldoutCaseResult] = []
    for case_id in sorted(expected_ids & predicted_ids):
        release_case = release_cases[case_id]
        pack_case = pack_cases.get(case_id)
        if pack_case is None:
            raise ValueError("Web4 inference pack is missing a signed release case")
        prediction = prediction_by_id[case_id]
        if prediction.release_case_sha256 != release_case.case_sha256:
            raise ValueError("Web4 prediction does not bind signed release case")
        if prediction.model_input_sha256 != release_case.model_input_sha256:
            raise ValueError("Web4 prediction does not bind signed model input")
        if pack_case.release_case_sha256 != release_case.case_sha256:
            raise ValueError("Web4 inference-pack case does not bind signed release case")
        if pack_case.model_input_sha256 != release_case.model_input_sha256:
            raise ValueError("Web4 inference-pack case does not bind signed model input")
        results.append(
            _case_result(
                release_case=release_case,
                prediction=prediction,
                answer_key=answer_keys[case_id],
            )
        )

    exact_count = sum(row.exact_match for row in results)
    expected_fields = sum(row.expected_field_count for row in results)
    matched_fields = sum(row.matched_field_count for row in results)
    exact_rate = exact_count / len(signed_release.cases)
    field_accuracy = matched_fields / expected_fields if expected_fields else 0.0
    complete_accounting = not missing_cases and not extra_cases
    violations: list[str] = []
    if len(signed_release.cases) < selected_policy.minimum_case_count:
        violations.append("signed HOLDOUT case count is below evaluation policy minimum")
    if selected_policy.require_complete_case_accounting and not complete_accounting:
        violations.append("prediction set does not have complete HOLDOUT case accounting")
    if exact_rate < selected_policy.minimum_exact_match_rate:
        violations.append("exact-match rate is below evaluation policy minimum")
    if field_accuracy < selected_policy.minimum_expected_field_accuracy:
        violations.append("expected-field accuracy is below evaluation policy minimum")
    if selected_policy.reject_unexpected_fields and any(
        row.unexpected_field_paths for row in results
    ):
        violations.append("prediction set contains unexpected answer fields")

    passed = not violations
    payload: dict[str, object] = {
        "schema_version": "sentinel.web4-holdout-evaluation-report.v1",
        "release_sha256": signed_release.release_sha256,
        "release_artifact_sha256": signed_release.artifact_sha256,
        "inference_pack_sha256": pack.pack_sha256,
        "prediction_set_sha256": predictions.prediction_set_sha256,
        "evaluation_policy_sha256": policy_sha,
        "model_ref": predictions.model_ref,
        "model_revision": predictions.model_revision,
        "model_artifact_sha256": predictions.model_artifact_sha256,
        "case_count": len(signed_release.cases),
        "prediction_count": len(predictions.predictions),
        "answer_key_count": len(answer_keys),
        "missing_case_ids": missing_cases,
        "extra_case_ids": extra_cases,
        "exact_match_cases": exact_count,
        "exact_match_rate": exact_rate,
        "expected_field_count": expected_fields,
        "matched_field_count": matched_fields,
        "expected_field_accuracy": field_accuracy,
        "case_results": [row.model_dump(mode="json") for row in results],
        "answer_key_values_embedded": False,
        "complete_case_accounting": complete_accounting,
        "passed": passed,
        "violations": violations,
    }
    payload["report_sha256"] = _digest(payload)
    return Web4HoldoutEvaluationReport.model_validate(payload), answer_bundle_sha


def build_web4_holdout_evaluation_evidence(
    *,
    release: Web4HoldoutRelease,
    inference_pack: Web4HoldoutInferencePack,
    prediction_set: Web4HoldoutPredictionSet,
    answer_key_dir: str | Path,
    owner_public_key: Ed25519PublicKey,
    benchmark_policy_path: str | Path,
    policy: Web4HoldoutEvaluationPolicy | None = None,
) -> Web4HoldoutEvaluationEvidence:
    signed_release = verify_web4_holdout_release(
        release,
        owner_public_key=owner_public_key,
        benchmark_policy_path=benchmark_policy_path,
    )
    selected_policy = policy or Web4HoldoutEvaluationPolicy()
    report, answer_bundle_sha = evaluate_web4_holdout_predictions(
        release=signed_release,
        inference_pack=inference_pack,
        prediction_set=prediction_set,
        answer_key_dir=answer_key_dir,
        owner_public_key=owner_public_key,
        benchmark_policy_path=benchmark_policy_path,
        policy=selected_policy,
    )
    payload: dict[str, object] = {
        "schema_version": "sentinel.web4-holdout-evaluation-evidence.v1",
        "release_id": signed_release.release_id,
        "release_sha256": signed_release.release_sha256,
        "release_artifact_sha256": signed_release.artifact_sha256,
        "release_owner_key_fingerprint": signed_release.owner_key_fingerprint,
        "inference_pack_sha256": inference_pack.pack_sha256,
        "prediction_set_sha256": prediction_set.prediction_set_sha256,
        "answer_key_bundle_sha256": answer_bundle_sha,
        "evaluation_policy_sha256": _policy_sha256(selected_policy),
        "model_ref": prediction_set.model_ref,
        "model_revision": prediction_set.model_revision,
        "model_artifact_sha256": prediction_set.model_artifact_sha256,
        "case_count": report.case_count,
        "prediction_count": report.prediction_count,
        "answer_key_count": report.answer_key_count,
        "complete_case_accounting": report.complete_case_accounting,
        "answer_key_values_embedded": False,
        "offline_replay": True,
        "network_access_required": False,
        "gpu_required": False,
        "research_evaluation_only": True,
        "training_authorization": False,
        "promotion_eligible": False,
        "production_activation_allowed": False,
        "report": report.model_dump(mode="json"),
        "passed": report.passed,
    }
    payload["evidence_sha256"] = _digest(payload)
    return Web4HoldoutEvaluationEvidence.model_validate(payload)


def write_web4_holdout_evaluation_evidence(
    evidence: Web4HoldoutEvaluationEvidence,
    path: str | Path,
) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Web4 HOLDOUT evaluation evidence already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(evidence.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
