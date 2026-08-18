import json

from koschei_sentinel.defense_reflex_gold_release import (
    GoldHoldoutEvaluationCase,
    write_gold_defense_release,
)
from koschei_sentinel.gold_holdout_evaluation import (
    GoldHoldoutInferenceCase,
    GoldHoldoutPredictedStep,
    build_gold_holdout_prediction,
    evaluate_gold_holdout_predictions,
    export_gold_holdout_inference_pack,
)
from tests.test_defense_reflex_gold_release import _release_rows


def _fixture(tmp_path):
    _policy, rows = _release_rows()
    release = tmp_path / "gold-release"
    write_gold_defense_release(rows, release)
    inference = tmp_path / "gold-inference"
    export_gold_holdout_inference_pack(release, inference)

    input_line = (inference / "inputs.jsonl").read_text(encoding="utf-8").splitlines()[0]
    inference_case = GoldHoldoutInferenceCase.model_validate_json(input_line)
    case_line = (release / "holdout" / "cases.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    gold_case = GoldHoldoutEvaluationCase.model_validate_json(case_line)
    return release, inference, inference_case, gold_case


def _perfect_prediction(inference_case, gold_case):
    steps = [GoldHoldoutPredictedStep.model_validate(row) for row in gold_case.expected_sequence]
    return build_gold_holdout_prediction(
        inference_case=inference_case,
        model_ref="Qwen/Qwen3.5-9B-Base",
        model_revision="candidate:test",
        adapter_digest="d" * 64,
        interpretation="Evidence supports the reviewed defensive sequence.",
        defense_sequence=steps,
    )


def _all_keys(value):
    keys = []
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.append(str(key))
            keys.extend(_all_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.extend(_all_keys(nested))
    return keys


def test_holdout_inference_pack_contains_no_answer_key(tmp_path) -> None:
    _release, inference, _inference_case, _gold_case = _fixture(tmp_path)

    row = json.loads((inference / "inputs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    keys = set(_all_keys(row))

    assert "expected_interpretation" not in keys
    assert "expected_sequence" not in keys
    assert "scenario_truth" not in keys
    assert "range_report" not in keys
    assert set(row["input_context"]) == {
        "scenario_id",
        "critical_entity_ids",
        "graph_snapshots",
    }


def test_structurally_correct_grounded_holdout_prediction_passes(tmp_path) -> None:
    release, _inference, inference_case, gold_case = _fixture(tmp_path)
    prediction = _perfect_prediction(inference_case, gold_case)

    report = evaluate_gold_holdout_predictions(release, [prediction])

    assert report.passed is True
    assert report.structural_exact_rate == 1.0
    assert report.mode_accuracy == 1.0
    assert report.action_accuracy == 1.0
    assert report.target_accuracy == 1.0
    assert report.evidence_grounding_rate == 1.0
    assert report.target_grounding_rate == 1.0
    assert report.outcome_verification_rate == 1.0
    assert report.violations == []


def test_holdout_prediction_cannot_invent_evidence(tmp_path) -> None:
    release, _inference, inference_case, gold_case = _fixture(tmp_path)
    steps = [GoldHoldoutPredictedStep.model_validate(row) for row in gold_case.expected_sequence]
    first = steps[0].model_copy(update={"supporting_evidence_ids": ["evidence:not-visible"]})
    prediction = build_gold_holdout_prediction(
        inference_case=inference_case,
        model_ref="Qwen/Qwen3.5-9B-Base",
        model_revision="candidate:test",
        adapter_digest="d" * 64,
        interpretation="Attempted answer with invented evidence.",
        defense_sequence=[first, *steps[1:]],
    )

    report = evaluate_gold_holdout_predictions(release, [prediction])

    assert report.passed is False
    assert report.evidence_grounding_rate < 1.0
    assert any("evidence absent from visible input" in row for row in report.violations)
