from koschei_sentinel.cyber_defense_promotion import (
    build_cyber_defense_promotion_evidence,
)
from koschei_sentinel.gold_holdout_evaluation import GoldHoldoutEvaluationPolicy
from tests.test_cyber_defense_promotion import (
    ADAPTER_DIGEST,
    _bundle,
    _gold_evidence,
    _load,
    _multi,
    _single,
)


def test_promotion_rechecks_minimum_gold_holdout_case_count() -> None:
    policy = GoldHoldoutEvaluationPolicy(minimum_case_count=3)
    gold_evidence = _gold_evidence(policy=policy)
    assert gold_evidence.case_count == 2
    assert gold_evidence.passed is True

    promotion = build_cyber_defense_promotion_evidence(
        promotion_id="promotion:minimum-case-count",
        candidate_model_ref="sentinel:candidate",
        candidate_model_revision=ADAPTER_DIGEST,
        training_bundle=_bundle(),
        cyber_range_report=_single(),
        multi_incident_range_report=_multi(),
        defense_load_range_report=_load(),
        gold_holdout_evidence=gold_evidence,
        gold_holdout_policy=policy,
    )

    assert promotion.cyber_range_passed is True
    assert promotion.multi_incident_range_passed is True
    assert promotion.defense_load_range_passed is True
    assert promotion.gold_holdout_passed is False
    assert promotion.ready_for_promotion is False
