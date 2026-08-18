import json

import pytest

from koschei_sentinel.cyber_seed_curriculum import build_seed_curriculum
from koschei_sentinel.defense_reflex_gold_queue import (
    GoldReviewSplitPolicy,
    build_gold_review_queue,
    write_gold_review_queue,
)


def _scenarios():
    return [scenario for _family, scenario, _lesson in build_seed_curriculum()]


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


def test_gold_review_queue_is_deterministic_before_human_review() -> None:
    scenarios = _scenarios()
    first, policy = build_gold_review_queue(scenarios)
    second, second_policy = build_gold_review_queue(list(reversed(scenarios)))

    assert policy == second_policy
    assert [row.packet_id for row in first] == [row.packet_id for row in second]
    assert [row.split for row in first] == [row.split for row in second]
    assert [row.split_basis_sha256 for row in first] == [
        row.split_basis_sha256 for row in second
    ]


def test_model_visible_review_context_excludes_answer_key_fields() -> None:
    packets, _policy = build_gold_review_queue(_scenarios())
    forbidden_keys = {
        "truth",
        "scenario_truth",
        "expected_reroute_ticks",
        "simulated_action_outcomes",
        "failure_type",
        "attempted_actions",
        "succeeded_actions",
        "contained",
        "reroute_expected",
    }

    for packet in packets:
        visible_keys = set(_all_keys(packet.model_visible_context))
        assert forbidden_keys.isdisjoint(visible_keys)
        assert "scenario_truth" in packet.review_only_context
        assert "range_report" in packet.review_only_context
        assert packet.training_authorization is False
        assert packet.promotion_eligible is False


def test_gold_queue_writes_physically_separate_review_splits(tmp_path) -> None:
    output = tmp_path / "gold-review-queue"
    manifest = write_gold_review_queue(_scenarios(), output)

    assert manifest.packet_count == 32
    assert manifest.scenario_count == 32
    assert manifest.training_ready is False
    assert manifest.review_required is True
    assert manifest.train_packets + manifest.validation_packets + manifest.holdout_packets == 32

    train = (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
    validation = (output / "validation.jsonl").read_text(encoding="utf-8").splitlines()
    holdout = (output / "holdout.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(train) == manifest.train_packets
    assert len(validation) == manifest.validation_packets
    assert len(holdout) == manifest.holdout_packets

    packet_ids = {
        json.loads(line)["packet_id"]
        for line in train + validation + holdout
        if line.strip()
    }
    assert len(packet_ids) == 32


def test_invalid_gold_split_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="sum to 10000"):
        GoldReviewSplitPolicy(
            train_basis_points=8000,
            validation_basis_points=1500,
            holdout_basis_points=1000,
        )
