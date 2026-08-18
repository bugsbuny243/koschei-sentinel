import json

import pytest

from koschei_sentinel.cyber_seed_curriculum import build_seed_curriculum
from koschei_sentinel.defense_reflex_corpus_v3 import DefenseReflexCorpusManifestV3
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
    GoldHoldoutManifest,
    write_gold_defense_release,
)
from koschei_sentinel.defense_reflex_gold_review import (
    GoldHumanReviewSpec,
    review_gold_packet,
)
from koschei_sentinel.defense_reflex_review import CorrectionReviewDecision


def _rows_by_split():
    seed_rows = build_seed_curriculum()
    report_sha_by_id = {
        scenario.scenario_id: _report_sha256(scenario)[1]
        for _family, scenario, _lesson in seed_rows
    }

    for index in range(10000):
        policy = GoldReviewSplitPolicy(seed=f"gold-release-policy-{index}")
        selected = {}
        for _family, scenario, lesson in seed_rows:
            report_sha = report_sha_by_id[scenario.scenario_id]
            basis = _split_basis(scenario.scenario_id, report_sha, policy)
            split = _split_for_basis(basis, policy)
            selected.setdefault(split, (scenario, lesson))
            if len(selected) == 3:
                return policy, selected
    raise AssertionError("could not find one split policy covering TRAIN/VALIDATION/HOLDOUT")


def _review(packet, scenario, lesson):
    if packet.split is GoldReviewSplit.HOLDOUT:
        training = False
        evaluation = True
    else:
        training = True
        evaluation = False
    spec = GoldHumanReviewSpec(
        reviewer_id="human-reviewer:gold-release",
        decision=CorrectionReviewDecision.APPROVE,
        interpretation=lesson.interpretation,
        expected_steps=lesson.expected_steps,
        review_evidence_ids=[f"human-review:{scenario.scenario_id}:case-record"],
        outcome_verified=True,
        authorize_for_training=training,
        authorize_for_evaluation=evaluation,
    )
    return review_gold_packet(packet, scenario, spec)


def _release_rows():
    policy, selected = _rows_by_split()
    rows = []
    for split in (
        GoldReviewSplit.TRAIN,
        GoldReviewSplit.VALIDATION,
        GoldReviewSplit.HOLDOUT,
    ):
        scenario, lesson = selected[split]
        packet = build_gold_review_packet(scenario, policy=policy)
        assert packet.split is split
        reviewed = _review(packet, scenario, lesson)
        rows.append((scenario, packet, reviewed))
    return policy, rows


def test_gold_release_physically_isolates_holdout_from_training(tmp_path) -> None:
    _policy, rows = _release_rows()
    output = tmp_path / "gold-release"

    manifest = write_gold_defense_release(rows, output)

    assert manifest.train_examples == 1
    assert manifest.validation_examples == 1
    assert manifest.holdout_cases == 1
    assert manifest.promotion_ready is True
    assert manifest.holdout_training_authorization is False
    assert manifest.split_policy_sha256 == rows[0][1].split_policy_sha256

    train_manifest = DefenseReflexCorpusManifestV3.model_validate_json(
        (output / "train" / "manifest.json").read_bytes()
    )
    validation_manifest = DefenseReflexCorpusManifestV3.model_validate_json(
        (output / "validation" / "manifest.json").read_bytes()
    )
    holdout_manifest = GoldHoldoutManifest.model_validate_json(
        (output / "holdout" / "manifest.json").read_bytes()
    )

    assert train_manifest.promotion_eligible is True
    assert validation_manifest.promotion_eligible is True
    assert train_manifest.human_reviewed_examples == 1
    assert validation_manifest.human_reviewed_examples == 1
    assert holdout_manifest.training_authorization is False
    assert holdout_manifest.evaluation_ready is True

    assert (output / "train" / "examples.jsonl").is_file()
    assert (output / "validation" / "examples.jsonl").is_file()
    assert (output / "holdout" / "cases.jsonl").is_file()
    assert not (output / "holdout" / "examples.jsonl").exists()

    holdout_lines = (output / "holdout" / "cases.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    holdout_case = GoldHoldoutEvaluationCase.model_validate_json(holdout_lines[0])
    assert holdout_case.training_authorization is False
    assert holdout_case.evaluation_authorization is True

    train_ids = set(train_manifest.source_report_sha256s)
    validation_ids = set(validation_manifest.source_report_sha256s)
    holdout_ids = set(holdout_manifest.scenario_ids)
    assert train_ids.isdisjoint(validation_ids)
    assert holdout_ids == {rows[2][0].scenario_id}


def test_gold_release_rejects_mixed_pre_review_split_policies(tmp_path) -> None:
    policy, rows = _release_rows()
    scenario, _packet, _reviewed = rows[0]
    lesson = next(
        lesson
        for _family, candidate, lesson in build_seed_curriculum()
        if candidate.scenario_id == scenario.scenario_id
    )
    different_policy = GoldReviewSplitPolicy(seed=policy.seed + "-different")
    different_packet = build_gold_review_packet(scenario, policy=different_policy)
    different_review = _review(different_packet, scenario, lesson)
    mixed_rows = [
        (scenario, different_packet, different_review),
        rows[1],
        rows[2],
    ]

    with pytest.raises(ValueError, match="cannot mix pre-review split policies"):
        write_gold_defense_release(mixed_rows, tmp_path / "mixed-release")


def test_gold_holdout_schema_cannot_parse_as_training_example(tmp_path) -> None:
    _policy, rows = _release_rows()
    output = tmp_path / "gold-release"
    write_gold_defense_release(rows, output)

    holdout_line = (output / "holdout" / "cases.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    payload = json.loads(holdout_line)

    assert payload["schema_version"] == "sentinel.gold-holdout-evaluation-case.v1"
    assert "graph_snapshots" not in payload
    assert payload["training_authorization"] is False
