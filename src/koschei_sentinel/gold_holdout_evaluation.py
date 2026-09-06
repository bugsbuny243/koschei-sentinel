from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.defense_authority import DefenseActionType, DefenseMode
from koschei_sentinel.defense_reflex_gold_release import (
    GoldHoldoutEvaluationCase,
    GoldHoldoutManifest,
)
from koschei_sentinel.defense_reflex_gold_release_audit import (
    audit_gold_defense_release,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import canonical_json


class GoldHoldoutInferenceCase(StrictModel):
    schema_version: Literal["sentinel.gold-holdout-inference-case.v1"] = (
        "sentinel.gold-holdout-inference-case.v1"
    )
    case_id: str
    scenario_id: str
    input_context: dict[str, object]
    input_context_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class GoldHoldoutInferenceManifest(StrictModel):
    schema_version: Literal["sentinel.gold-holdout-inference-manifest.v1"] = (
        "sentinel.gold-holdout-inference-manifest.v1"
    )
    case_count: int = Field(gt=0)
    case_ids: list[str] = Field(min_length=1)
    inputs_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_gold_audit_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    answer_key_excluded: Literal[True] = True


class GoldHoldoutPredictedStep(StrictModel):
    sequence: int = Field(gt=0)
    expected_mode: DefenseMode
    action: DefenseActionType
    target_entity_id: str = Field(min_length=1, max_length=1024)
    rationale: str = Field(min_length=1, max_length=8000)
    supporting_evidence_ids: list[str] = Field(min_length=1, max_length=256)
    outcome_verification_required: Literal[True] = True


class GoldHoldoutPrediction(StrictModel):
    schema_version: Literal["sentinel.gold-holdout-prediction.v1"] = (
        "sentinel.gold-holdout-prediction.v1"
    )
    case_id: str
    scenario_id: str
    input_context_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_ref: str = Field(min_length=3, max_length=512)
    model_revision: str = Field(min_length=1, max_length=256)
    adapter_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    interpretation: str = Field(min_length=1, max_length=12000)
    defense_sequence: list[GoldHoldoutPredictedStep] = Field(min_length=1, max_length=64)
    prediction_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def sequence_is_contiguous(self) -> GoldHoldoutPrediction:
        sequences = [row.sequence for row in self.defense_sequence]
        if sequences != list(range(1, len(sequences) + 1)):
            raise ValueError("Gold holdout prediction steps must be contiguous from 1")
        return self


class GoldHoldoutEvaluationPolicy(StrictModel):
    schema_version: Literal["sentinel.gold-holdout-evaluation-policy.v1"] = (
        "sentinel.gold-holdout-evaluation-policy.v1"
    )
    minimum_case_count: int = Field(default=1, ge=1, le=100000)
    minimum_structural_exact_rate: float = Field(default=0.90, ge=0.0, le=1.0)
    minimum_mode_accuracy: float = Field(default=0.95, ge=0.0, le=1.0)
    minimum_action_accuracy: float = Field(default=0.95, ge=0.0, le=1.0)
    minimum_target_accuracy: float = Field(default=0.95, ge=0.0, le=1.0)
    minimum_evidence_selection_accuracy: float = Field(default=0.95, ge=0.0, le=1.0)
    minimum_evidence_grounding_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    minimum_target_grounding_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    minimum_outcome_verification_rate: float = Field(default=1.0, ge=0.0, le=1.0)


class GoldHoldoutCaseResult(StrictModel):
    case_id: str
    scenario_id: str
    structural_exact: bool
    expected_steps: int = Field(ge=0)
    predicted_steps: int = Field(ge=0)
    mode_matches: int = Field(ge=0)
    action_matches: int = Field(ge=0)
    target_matches: int = Field(ge=0)
    evidence_selection_matches: int = Field(ge=0)
    grounded_evidence_steps: int = Field(ge=0)
    grounded_target_steps: int = Field(ge=0)
    outcome_verification_steps: int = Field(ge=0)
    violations: list[str]


class GoldHoldoutEvaluationReport(StrictModel):
    schema_version: Literal["sentinel.gold-holdout-evaluation-report.v2"] = (
        "sentinel.gold-holdout-evaluation-report.v2"
    )
    model_ref: str
    model_revision: str
    adapter_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    case_count: int = Field(gt=0)
    prediction_count: int = Field(ge=0)
    missing_case_ids: list[str]
    extra_case_ids: list[str]
    structural_exact_cases: int = Field(ge=0)
    structural_exact_rate: float = Field(ge=0.0, le=1.0)
    compared_steps: int = Field(ge=0)
    predicted_steps: int = Field(ge=0)
    mode_accuracy: float = Field(ge=0.0, le=1.0)
    action_accuracy: float = Field(ge=0.0, le=1.0)
    target_accuracy: float = Field(ge=0.0, le=1.0)
    evidence_selection_accuracy: float = Field(ge=0.0, le=1.0)
    evidence_grounding_rate: float = Field(ge=0.0, le=1.0)
    target_grounding_rate: float = Field(ge=0.0, le=1.0)
    outcome_verification_rate: float = Field(ge=0.0, le=1.0)
    case_results: list[GoldHoldoutCaseResult]
    passed: bool
    violations: list[str]
    report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def _sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _digest_without(payload: dict[str, object], field_name: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field_name, None)
    return _sha256_text(canonical_json(unsigned))


def _input_context_sha256(context: dict[str, object]) -> str:
    return _sha256_text(canonical_json(context))


def _serialize_inputs(rows: list[GoldHoldoutInferenceCase]) -> str:
    return "".join(
        json.dumps(row.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in sorted(rows, key=lambda item: item.case_id)
    )


def _load_holdout_cases(release_dir: Path) -> list[GoldHoldoutEvaluationCase]:
    holdout_dir = release_dir / "holdout"
    manifest = GoldHoldoutManifest.model_validate_json(
        (holdout_dir / "manifest.json").read_bytes()
    )
    lines = (holdout_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    cases = [
        GoldHoldoutEvaluationCase.model_validate_json(line)
        for line in lines
        if line.strip()
    ]
    if len(cases) != manifest.case_count:
        raise ValueError("Gold holdout case count differs from manifest")
    return sorted(cases, key=lambda row: row.case_id)


def export_gold_holdout_inference_pack(
    release_dir: str | Path,
    output_dir: str | Path,
) -> GoldHoldoutInferenceManifest:
    release = Path(release_dir)
    audit = audit_gold_defense_release(release)
    if not audit.valid:
        raise ValueError("cannot export inference pack from invalid Gold release")
    cases = _load_holdout_cases(release)
    inference_rows = [
        GoldHoldoutInferenceCase(
            case_id=case.case_id,
            scenario_id=case.scenario_id,
            input_context=case.input_context,
            input_context_sha256=_input_context_sha256(case.input_context),
        )
        for case in cases
    ]
    payload = _serialize_inputs(inference_rows)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "inputs.jsonl").write_text(payload, encoding="utf-8")
    manifest = GoldHoldoutInferenceManifest(
        case_count=len(inference_rows),
        case_ids=sorted(row.case_id for row in inference_rows),
        inputs_sha256=_sha256_text(payload),
        source_gold_audit_sha256=audit.audit_sha256,
    )
    (destination / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_gold_holdout_prediction(
    *,
    inference_case: GoldHoldoutInferenceCase,
    model_ref: str,
    model_revision: str,
    adapter_digest: str,
    interpretation: str,
    defense_sequence: list[GoldHoldoutPredictedStep],
) -> GoldHoldoutPrediction:
    payload: dict[str, object] = {
        "schema_version": "sentinel.gold-holdout-prediction.v1",
        "case_id": inference_case.case_id,
        "scenario_id": inference_case.scenario_id,
        "input_context_sha256": inference_case.input_context_sha256,
        "model_ref": model_ref,
        "model_revision": model_revision,
        "adapter_digest": adapter_digest,
        "interpretation": interpretation,
        "defense_sequence": [row.model_dump(mode="json") for row in defense_sequence],
    }
    payload["prediction_sha256"] = _digest_without(payload, "prediction_sha256")
    return GoldHoldoutPrediction.model_validate(payload)


def _collect_visible_ids(context: dict[str, object]) -> tuple[set[str], set[str]]:
    entity_ids: set[str] = set()
    evidence_ids: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            entity = value.get("entity_id")
            evidence = value.get("evidence_id")
            if isinstance(entity, str):
                entity_ids.add(entity)
            if isinstance(evidence, str):
                evidence_ids.add(evidence)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(context)
    return entity_ids, evidence_ids


def _verify_prediction_digest(prediction: GoldHoldoutPrediction) -> bool:
    return (
        _digest_without(prediction.model_dump(mode="json"), "prediction_sha256")
        == prediction.prediction_sha256
    )


def _case_result(
    case: GoldHoldoutEvaluationCase,
    prediction: GoldHoldoutPrediction,
) -> GoldHoldoutCaseResult:
    violations: list[str] = []
    visible_entities, visible_evidence = _collect_visible_ids(case.input_context)
    expected = case.expected_sequence
    predicted = prediction.defense_sequence
    compared = min(len(expected), len(predicted))
    mode_matches = 0
    action_matches = 0
    target_matches = 0
    evidence_selection_matches = 0
    grounded_evidence = 0
    grounded_targets = 0
    outcome_verification = 0

    if prediction.scenario_id != case.scenario_id:
        violations.append("prediction scenario_id differs from Gold case")
    expected_context_sha = _input_context_sha256(case.input_context)
    if prediction.input_context_sha256 != expected_context_sha:
        violations.append("prediction input-context digest differs from Gold case")
    if not _verify_prediction_digest(prediction):
        violations.append("prediction self-hash does not verify")
    if len(expected) != len(predicted):
        violations.append("predicted defense sequence length differs from Gold sequence")

    for index in range(compared):
        gold_step = expected[index]
        predicted_step = predicted[index]
        if predicted_step.expected_mode.value == gold_step.get("expected_mode"):
            mode_matches += 1
        if predicted_step.action.value == gold_step.get("action"):
            action_matches += 1
        if predicted_step.target_entity_id == gold_step.get("target_entity_id"):
            target_matches += 1
        expected_evidence = set(gold_step.get("supporting_evidence_ids", []))
        predicted_evidence = set(predicted_step.supporting_evidence_ids)
        if predicted_evidence == expected_evidence:
            evidence_selection_matches += 1
        if predicted_step.target_entity_id in visible_entities:
            grounded_targets += 1
        else:
            violations.append(
                f"step {predicted_step.sequence} targets an entity absent from visible input"
            )
        if all(row in visible_evidence for row in predicted_step.supporting_evidence_ids):
            grounded_evidence += 1
        else:
            violations.append(
                f"step {predicted_step.sequence} cites evidence absent from visible input"
            )
        if predicted_step.outcome_verification_required:
            outcome_verification += 1

    structural_exact = (
        len(expected) == len(predicted)
        and compared > 0
        and mode_matches == compared
        and action_matches == compared
        and target_matches == compared
        and evidence_selection_matches == compared
        and grounded_evidence == compared
        and grounded_targets == compared
        and outcome_verification == compared
        and not violations
    )
    return GoldHoldoutCaseResult(
        case_id=case.case_id,
        scenario_id=case.scenario_id,
        structural_exact=structural_exact,
        expected_steps=len(expected),
        predicted_steps=len(predicted),
        mode_matches=mode_matches,
        action_matches=action_matches,
        target_matches=target_matches,
        evidence_selection_matches=evidence_selection_matches,
        grounded_evidence_steps=grounded_evidence,
        grounded_target_steps=grounded_targets,
        outcome_verification_steps=outcome_verification,
        violations=violations,
    )


def evaluate_gold_holdout_predictions(
    release_dir: str | Path,
    predictions: list[GoldHoldoutPrediction],
    *,
    policy: GoldHoldoutEvaluationPolicy | None = None,
) -> GoldHoldoutEvaluationReport:
    release = Path(release_dir)
    audit = audit_gold_defense_release(release)
    if not audit.valid:
        raise ValueError("cannot evaluate predictions against an invalid Gold release")
    cases = _load_holdout_cases(release)
    selected_policy = policy or GoldHoldoutEvaluationPolicy()
    case_by_id = {row.case_id: row for row in cases}
    prediction_by_id: dict[str, GoldHoldoutPrediction] = {}
    for prediction in predictions:
        if prediction.case_id in prediction_by_id:
            raise ValueError(f"duplicate Gold holdout prediction: {prediction.case_id}")
        prediction_by_id[prediction.case_id] = prediction

    model_identities = {
        (row.model_ref, row.model_revision, row.adapter_digest) for row in predictions
    }
    if len(model_identities) != 1:
        raise ValueError("Gold holdout predictions must come from one model/adapter identity")
    model_ref, model_revision, adapter_digest = next(iter(model_identities))

    expected_ids = set(case_by_id)
    predicted_ids = set(prediction_by_id)
    missing = sorted(expected_ids - predicted_ids)
    extra = sorted(predicted_ids - expected_ids)
    results = [
        _case_result(case_by_id[case_id], prediction_by_id[case_id])
        for case_id in sorted(expected_ids & predicted_ids)
    ]

    structural_exact_cases = sum(row.structural_exact for row in results)
    compared_steps = sum(min(row.expected_steps, row.predicted_steps) for row in results)
    predicted_steps = sum(row.predicted_steps for row in results)
    mode_matches = sum(row.mode_matches for row in results)
    action_matches = sum(row.action_matches for row in results)
    target_matches = sum(row.target_matches for row in results)
    evidence_selection_matches = sum(row.evidence_selection_matches for row in results)
    grounded_evidence = sum(row.grounded_evidence_steps for row in results)
    grounded_targets = sum(row.grounded_target_steps for row in results)
    verified_outcomes = sum(row.outcome_verification_steps for row in results)

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    structural_rate = ratio(structural_exact_cases, len(cases))
    mode_accuracy = ratio(mode_matches, compared_steps)
    action_accuracy = ratio(action_matches, compared_steps)
    target_accuracy = ratio(target_matches, compared_steps)
    evidence_selection_accuracy = ratio(evidence_selection_matches, compared_steps)
    evidence_grounding_rate = ratio(grounded_evidence, predicted_steps)
    target_grounding_rate = ratio(grounded_targets, predicted_steps)
    outcome_verification_rate = ratio(verified_outcomes, predicted_steps)

    violations: list[str] = []
    if len(cases) < selected_policy.minimum_case_count:
        violations.append(
            "Gold HOLDOUT case count below policy: "
            f"{len(cases)} < {selected_policy.minimum_case_count}"
        )
    if missing:
        violations.append("missing Gold HOLDOUT predictions")
    if extra:
        violations.append("predictions contain unknown Gold HOLDOUT case IDs")
    thresholds = (
        ("structural exact rate", structural_rate, selected_policy.minimum_structural_exact_rate),
        ("mode accuracy", mode_accuracy, selected_policy.minimum_mode_accuracy),
        ("action accuracy", action_accuracy, selected_policy.minimum_action_accuracy),
        ("target accuracy", target_accuracy, selected_policy.minimum_target_accuracy),
        (
            "evidence selection accuracy",
            evidence_selection_accuracy,
            selected_policy.minimum_evidence_selection_accuracy,
        ),
        (
            "evidence grounding rate",
            evidence_grounding_rate,
            selected_policy.minimum_evidence_grounding_rate,
        ),
        (
            "target grounding rate",
            target_grounding_rate,
            selected_policy.minimum_target_grounding_rate,
        ),
        (
            "outcome verification rate",
            outcome_verification_rate,
            selected_policy.minimum_outcome_verification_rate,
        ),
    )
    for label, observed, minimum in thresholds:
        if observed + 1e-12 < minimum:
            violations.append(f"{label} below policy: {observed:.6f} < {minimum:.6f}")
    for result in results:
        violations.extend(f"{result.case_id}: {row}" for row in result.violations)

    passed = not violations and len(predictions) == len(cases)
    payload: dict[str, object] = {
        "schema_version": "sentinel.gold-holdout-evaluation-report.v2",
        "model_ref": model_ref,
        "model_revision": model_revision,
        "adapter_digest": adapter_digest,
        "case_count": len(cases),
        "prediction_count": len(predictions),
        "missing_case_ids": missing,
        "extra_case_ids": extra,
        "structural_exact_cases": structural_exact_cases,
        "structural_exact_rate": structural_rate,
        "compared_steps": compared_steps,
        "predicted_steps": predicted_steps,
        "mode_accuracy": mode_accuracy,
        "action_accuracy": action_accuracy,
        "target_accuracy": target_accuracy,
        "evidence_selection_accuracy": evidence_selection_accuracy,
        "evidence_grounding_rate": evidence_grounding_rate,
        "target_grounding_rate": target_grounding_rate,
        "outcome_verification_rate": outcome_verification_rate,
        "case_results": [row.model_dump(mode="json") for row in results],
        "passed": passed,
        "violations": violations,
    }
    payload["report_sha256"] = _digest_without(payload, "report_sha256")
    return GoldHoldoutEvaluationReport.model_validate(payload)
