import pytest

from koschei_sentinel.cyber_seed_curriculum import build_seed_curriculum
from koschei_sentinel.defense_reflex_gold_queue import (
    GoldReviewSplit,
    GoldReviewSplitPolicy,
    _split_basis,
    _split_for_basis,
    build_gold_review_packet,
)
from koschei_sentinel.defense_reflex_gold_review import (
    GoldHumanReviewSpec,
    review_gold_packet,
)
from koschei_sentinel.defense_reflex_review import CorrectionReviewDecision


def _seed_row():
    _family, scenario, lesson = build_seed_curriculum()[0]
    return scenario, lesson


def _packet_for_split(scenario, desired: GoldReviewSplit):
    first = build_gold_review_packet(scenario, policy=GoldReviewSplitPolicy())
    for index in range(20000):
        policy = GoldReviewSplitPolicy(seed=f"gold-review-test-{index}")
        basis = _split_basis(scenario.scenario_id, first.source_report_sha256, policy)
        if _split_for_basis(basis, policy) is desired:
            return build_gold_review_packet(scenario, policy=policy)
    raise AssertionError(f"could not assign scenario to {desired.value}")


def _approved_spec(lesson, *, training: bool, evaluation: bool) -> GoldHumanReviewSpec:
    return GoldHumanReviewSpec(
        reviewer_id="human-reviewer:test",
        decision=CorrectionReviewDecision.APPROVE,
        interpretation=lesson.interpretation,
        expected_steps=lesson.expected_steps,
        review_evidence_ids=["human-review:case-record:001"],
        outcome_verified=True,
        authorize_for_training=training,
        authorize_for_evaluation=evaluation,
    )


def test_human_train_review_can_be_promotion_eligible() -> None:
    scenario, lesson = _seed_row()
    packet = _packet_for_split(scenario, GoldReviewSplit.TRAIN)

    reviewed = review_gold_packet(
        packet,
        scenario,
        _approved_spec(lesson, training=True, evaluation=False),
    )

    assert reviewed.split is GoldReviewSplit.TRAIN
    assert reviewed.training_authorization is True
    assert reviewed.promotion_eligible is True
    assert reviewed.evaluation_authorization is False
    assert reviewed.outcome_verified is True


def test_holdout_review_is_evaluation_only() -> None:
    scenario, lesson = _seed_row()
    packet = _packet_for_split(scenario, GoldReviewSplit.HOLDOUT)

    reviewed = review_gold_packet(
        packet,
        scenario,
        _approved_spec(lesson, training=False, evaluation=True),
    )

    assert reviewed.split is GoldReviewSplit.HOLDOUT
    assert reviewed.training_authorization is False
    assert reviewed.promotion_eligible is False
    assert reviewed.evaluation_authorization is True


def test_holdout_review_rejects_training_authorization() -> None:
    scenario, lesson = _seed_row()
    packet = _packet_for_split(scenario, GoldReviewSplit.HOLDOUT)

    with pytest.raises(ValueError, match="HOLDOUT packet cannot be authorized for training"):
        review_gold_packet(
            packet,
            scenario,
            _approved_spec(lesson, training=True, evaluation=True),
        )


def test_gold_review_rejects_invented_supporting_evidence() -> None:
    scenario, lesson = _seed_row()
    packet = _packet_for_split(scenario, GoldReviewSplit.TRAIN)
    step = lesson.expected_steps[0].model_copy(
        update={"supporting_evidence_ids": ["evidence:not-in-graph"]}
    )
    spec = GoldHumanReviewSpec(
        reviewer_id="human-reviewer:test",
        decision=CorrectionReviewDecision.APPROVE,
        interpretation=lesson.interpretation,
        expected_steps=[step],
        review_evidence_ids=["human-review:case-record:002"],
        outcome_verified=True,
        authorize_for_training=True,
    )

    with pytest.raises(ValueError, match="supporting evidence absent"):
        review_gold_packet(packet, scenario, spec)


def test_gold_review_rejects_scenario_drift_after_split_assignment() -> None:
    scenario, lesson = _seed_row()
    packet = _packet_for_split(scenario, GoldReviewSplit.TRAIN)
    graph = scenario.graph_snapshots[0]
    relation = graph.relations[0].model_copy(update={"confidence": 0.51})
    changed_graph = graph.model_copy(
        update={"relations": [relation, *graph.relations[1:]]}
    )
    changed_scenario = scenario.model_copy(update={"graph_snapshots": [changed_graph]})

    with pytest.raises(ValueError, match="scenario drifted after queue assignment"):
        review_gold_packet(
            packet,
            changed_scenario,
            _approved_spec(lesson, training=True, evaluation=False),
        )
