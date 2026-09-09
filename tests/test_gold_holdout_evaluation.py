import json

from koschei_sentinel.cyber_seed_curriculum import build_seed_curriculum
from koschei_sentinel.defense_reflex_gold_queue import (
    GoldReviewSplit,
    GoldReviewSplitPolicy,
    _report_sha256,
    _split_basis,
    _split_for_basis,
    build_gold_review_packet,
)
from koschei_sentinel.defense_reflex_gold_release import (
    GoldHoldoutEvaluationCase,
    write_gold_defense_release,
)
from koschei_sentinel.gold_holdout_evaluation import (
    GoldHoldoutEvaluationPolicy,
    GoldHoldoutInferenceCase,
    GoldHoldoutPredictedStep,
    build_gold_holdout_prediction,
    evaluate_gold_holdout_predictions,
    export_gold_holdout_inference_pack,
)
from tests.test_defense_reflex_gold_release import _release_rows, _review


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


def _visible_evidence_ids(value):
    result = set()
    if isinstance(value, dict):
        evidence_id = value.get("evidence_id")
        if isinstance(evidence_id, str):
            result.add(evidence_id)
        for nested in value.values():
            result.update(_visible_evidence_ids(nested))
    elif isinstance(value, list):
        for nested in value:
            result.update(_visible_evidence_ids(nested))
    return result


def _fixture_with_alternative_visible_evidence(tmp_path):
    seed_rows = build_seed_curriculum()
    report_sha_by_id = {
        scenario.scenario_id: _report_sha256(scenario)[1]
        for _family, scenario, _lesson in seed_rows
    }

    for policy_index in range(10000):
        policy = GoldReviewSplitPolicy(seed=f"gold-evidence-selection-{policy_index}")
        train_row = None
        validation_row = None
        holdout_row = None
        alternative = None
        step_index = None

        for _family, scenario, lesson in seed_rows:
            report_sha = report_sha_by_id[scenario.scenario_id]
            basis = _split_basis(scenario.scenario_id, report_sha, policy)
            split = _split_for_basis(basis, policy)
            if split is GoldReviewSplit.TRAIN and train_row is None:
                train_row = (scenario, lesson)
            elif split is GoldReviewSplit.VALIDATION and validation_row is None:
                validation_row = (scenario, lesson)
            elif split is GoldReviewSplit.HOLDOUT and holdout_row is None:
                graph_payload = [
                    graph.model_dump(mode="json")
                    for graph in scenario.graph_snapshots
                ]
                visible = _visible_evidence_ids(graph_payload)
                for index, expected_step in enumerate(lesson.expected_steps):
                    expected = set(expected_step.supporting_evidence_ids)
                    candidates = sorted(visible - expected)
                    if candidates:
                        holdout_row = (scenario, lesson)
                        alternative = candidates[0]
                        step_index = index
                        break

        if train_row and validation_row and holdout_row and alternative is not None:
            selected = {
                GoldReviewSplit.TRAIN: train_row,
                GoldReviewSplit.VALIDATION: validation_row,
                GoldReviewSplit.HOLDOUT: holdout_row,
            }
            rows = []
            for split in (
                GoldReviewSplit.TRAIN,
                GoldReviewSplit.VALIDATION,
                GoldReviewSplit.HOLDOUT,
            ):
                scenario, lesson = selected[split]
                packet = build_gold_review_packet(scenario, policy=policy)
                assert packet.split is split
                rows.append((scenario, packet, _review(packet, scenario, lesson)))

            release = tmp_path / "gold-release-evidence-selection"
            write_gold_defense_release(rows, release)
            inference = tmp_path / "gold-inference-evidence-selection"
            export_gold_holdout_inference_pack(release, inference)
            inference_case = GoldHoldoutInferenceCase.model_validate_json(
                (inference / "inputs.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            gold_case = GoldHoldoutEvaluationCase.model_validate_json(
                (release / "holdout" / "cases.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            return release, inference_case, gold_case, step_index, alternative

    raise AssertionError(
        "could not construct a HOLDOUT case with an alternative visible evidence ID"
    )


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

    assert report.schema_version == "sentinel.gold-holdout-evaluation-report.v2"
    assert report.passed is True
    assert report.structural_exact_rate == 1.0
    assert report.mode_accuracy == 1.0
    assert report.action_accuracy == 1.0
    assert report.target_accuracy == 1.0
    assert report.evidence_selection_accuracy == 1.0
    assert report.evidence_grounding_rate == 1.0
    assert report.target_grounding_rate == 1.0
    assert report.outcome_verification_rate == 1.0
    assert report.violations == []


def test_extra_defense_step_cannot_preserve_perfect_step_accuracy(tmp_path) -> None:
    release, _inference, inference_case, gold_case = _fixture(tmp_path)
    steps = [GoldHoldoutPredictedStep.model_validate(row) for row in gold_case.expected_sequence]
    extra = steps[-1].model_copy(update={"sequence": len(steps) + 1})
    prediction = build_gold_holdout_prediction(
        inference_case=inference_case,
        model_ref="Qwen/Qwen3.5-9B-Base",
        model_revision="candidate:test",
        adapter_digest="d" * 64,
        interpretation="Matches Gold, then adds one unsupported extra defense step.",
        defense_sequence=[*steps, extra],
    )

    report = evaluate_gold_holdout_predictions(release, [prediction])
    expected_accuracy = len(steps) / (len(steps) + 1)

    assert report.passed is False
    assert report.compared_steps == len(steps) + 1
    assert report.mode_accuracy == expected_accuracy
    assert report.action_accuracy == expected_accuracy
    assert report.target_accuracy == expected_accuracy
    assert report.evidence_selection_accuracy == expected_accuracy
    assert report.structural_exact_rate == 0.0
    assert any(
        "predicted defense sequence length differs" in row
        for row in report.violations
    )


def test_perfect_single_case_still_fails_when_policy_requires_more_cases(tmp_path) -> None:
    release, _inference, inference_case, gold_case = _fixture(tmp_path)
    prediction = _perfect_prediction(inference_case, gold_case)
    policy = GoldHoldoutEvaluationPolicy(minimum_case_count=2)

    report = evaluate_gold_holdout_predictions(
        release,
        [prediction],
        policy=policy,
    )

    assert report.structural_exact_rate == 1.0
    assert report.evidence_selection_accuracy == 1.0
    assert report.passed is False
    assert "Gold HOLDOUT case count below policy: 1 < 2" in report.violations


def test_holdout_prediction_cannot_invent_evidence(tmp_path) -> None:
    release, _inference, inference_case, gold_case = _fixture(tmp_path)
    steps = [GoldHoldoutPredictedStep.model_validate(row) for row in gold_case.expected_sequence]
    first = steps[0].model_copy(
        update={"supporting_evidence_ids": ["evidence:not-visible"]}
    )
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


def test_visible_but_wrong_evidence_selection_fails_gold_exactness(tmp_path) -> None:
    release, inference_case, gold_case, step_index, alternative = (
        _fixture_with_alternative_visible_evidence(tmp_path)
    )
    steps = [GoldHoldoutPredictedStep.model_validate(row) for row in gold_case.expected_sequence]
    changed = steps[step_index].model_copy(
        update={"supporting_evidence_ids": [alternative]}
    )
    modified_steps = list(steps)
    modified_steps[step_index] = changed
    prediction = build_gold_holdout_prediction(
        inference_case=inference_case,
        model_ref="Qwen/Qwen3.5-9B-Base",
        model_revision="candidate:test",
        adapter_digest="d" * 64,
        interpretation="Uses visible evidence that is not the reviewed Gold support set.",
        defense_sequence=modified_steps,
    )

    report = evaluate_gold_holdout_predictions(release, [prediction])

    assert report.passed is False
    assert report.evidence_grounding_rate == 1.0
    assert report.evidence_selection_accuracy < 1.0
    assert report.structural_exact_rate < 1.0
    assert any("evidence selection accuracy below policy" in row for row in report.violations)
