import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.defense_reflex_gold_capacity import (
    _evaluate_gold_review_capacity,
    build_gold_review_capacity_report,
)
from koschei_sentinel.defense_reflex_gold_queue import GoldReviewSplit
from koschei_sentinel.gold_reviewer_trust import (
    build_gold_reviewer_trust_policy,
    write_gold_reviewer_trust_policy,
)
from koschei_sentinel.gold_review_signing import sign_gold_reviewed_packet
from tests.test_defense_reflex_gold_release import _release_rows


def _public_key_bytes(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _trusted_rows():
    policy, rows = _release_rows()
    reviewer = Ed25519PrivateKey.generate()
    owner = Ed25519PrivateKey.generate()
    trust_policy = build_gold_reviewer_trust_policy(
        reviewer.public_key(),
        owner,
        policy_id="gold-capacity-reviewer-v1",
    )
    scenarios = {scenario.scenario_id: scenario for scenario, _packet, _review in rows}
    packets = {packet.scenario_id: packet for _scenario, packet, _review in rows}
    reviews = {review.scenario_id: review for _scenario, _packet, review in rows}
    signatures = {
        review.scenario_id: sign_gold_reviewed_packet(review, reviewer)
        for _scenario, _packet, review in rows
    }
    return (
        policy,
        rows,
        reviewer,
        owner,
        trust_policy,
        scenarios,
        packets,
        reviews,
        signatures,
    )


def _write_artifacts(tmp_path):
    (
        policy,
        rows,
        reviewer,
        owner,
        trust_policy,
        _scenarios,
        _packets,
        _reviews,
        signatures,
    ) = _trusted_rows()
    dirs = {
        "scenarios": tmp_path / "scenarios",
        "packets": tmp_path / "packets",
        "reviews": tmp_path / "reviews",
        "signatures": tmp_path / "signatures",
    }
    for path in dirs.values():
        path.mkdir()

    for scenario, packet, reviewed in rows:
        scenario_id = scenario.scenario_id
        (dirs["scenarios"] / f"{scenario_id}.json").write_text(
            json.dumps(scenario.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (dirs["packets"] / f"{scenario_id}.json").write_text(
            json.dumps(packet.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (dirs["reviews"] / f"{scenario_id}.json").write_text(
            json.dumps(reviewed.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        proof = signatures[scenario_id]
        (dirs["signatures"] / f"{scenario_id}.json").write_text(
            json.dumps(proof.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    policy_path = tmp_path / "split-policy.json"
    policy_path.write_text(
        json.dumps(policy.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reviewer_key = tmp_path / "reviewer-public.pem"
    reviewer_key.write_bytes(_public_key_bytes(reviewer))
    owner_key = tmp_path / "owner-public.pem"
    owner_key.write_bytes(_public_key_bytes(owner))
    trust_path = tmp_path / "reviewer-trust.json"
    write_gold_reviewer_trust_policy(trust_policy, trust_path)
    return dirs, policy_path, reviewer_key, trust_path, owner_key


def test_internal_capacity_verifies_signed_human_review_and_exact_splits() -> None:
    (
        policy,
        _rows,
        reviewer,
        owner,
        trust_policy,
        scenarios,
        packets,
        reviews,
        signatures,
    ) = _trusted_rows()

    report = _evaluate_gold_review_capacity(
        scenarios=scenarios,
        packets=packets,
        reviews=reviews,
        signatures=signatures,
        split_policy=policy,
        reviewer_public_key=reviewer.public_key(),
        reviewer_trust_policy=trust_policy,
        owner_public_key=owner.public_key(),
        minimum_train_cases=1,
        minimum_validation_cases=1,
        minimum_holdout_cases=1,
    )

    assert report.train_count == 1
    assert report.validation_count == 1
    assert report.holdout_count == 1
    assert report.holdout_shortfall == 0
    assert report.ready_for_release is True
    assert report.blockers == []


def test_public_capacity_reports_real_production_holdout_shortfall(tmp_path) -> None:
    dirs, policy, reviewer_key, trust, owner_key = _write_artifacts(tmp_path)

    report = build_gold_review_capacity_report(
        scenario_dir=dirs["scenarios"],
        packet_dir=dirs["packets"],
        review_dir=dirs["reviews"],
        signature_dir=dirs["signatures"],
        split_policy_path=policy,
        reviewer_public_key_path=reviewer_key,
        reviewer_trust_policy_path=trust,
        owner_public_key_path=owner_key,
    )

    assert report.train_count == 1
    assert report.validation_count == 1
    assert report.holdout_count == 1
    assert report.minimum_holdout_cases == 50
    assert report.holdout_shortfall == 49
    assert report.ready_for_release is False
    assert report.blockers == ["Gold HOLDOUT capacity below minimum: 1 < 50"]


def test_public_capacity_cannot_lower_production_holdout_floor(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least 50 HOLDOUT"):
        build_gold_review_capacity_report(
            scenario_dir=tmp_path / "scenarios",
            packet_dir=tmp_path / "packets",
            review_dir=tmp_path / "reviews",
            signature_dir=tmp_path / "signatures",
            split_policy_path=tmp_path / "split-policy.json",
            reviewer_public_key_path=tmp_path / "reviewer-public.pem",
            reviewer_trust_policy_path=tmp_path / "reviewer-trust.json",
            owner_public_key_path=tmp_path / "owner-public.pem",
            minimum_holdout_cases=49,
        )


def test_capacity_rejects_post_review_split_reassignment() -> None:
    (
        policy,
        _rows,
        reviewer,
        owner,
        trust_policy,
        scenarios,
        packets,
        reviews,
        signatures,
    ) = _trusted_rows()
    holdout_id = next(
        scenario_id
        for scenario_id, packet in packets.items()
        if packet.split is GoldReviewSplit.HOLDOUT
    )
    packets[holdout_id] = packets[holdout_id].model_copy(
        update={"split": GoldReviewSplit.TRAIN}
    )

    with pytest.raises(ValueError, match="pre-review split drift"):
        _evaluate_gold_review_capacity(
            scenarios=scenarios,
            packets=packets,
            reviews=reviews,
            signatures=signatures,
            split_policy=policy,
            reviewer_public_key=reviewer.public_key(),
            reviewer_trust_policy=trust_policy,
            owner_public_key=owner.public_key(),
            minimum_train_cases=1,
            minimum_validation_cases=1,
            minimum_holdout_cases=1,
        )


def test_capacity_rejects_missing_signature_scenario() -> None:
    (
        policy,
        _rows,
        reviewer,
        owner,
        trust_policy,
        scenarios,
        packets,
        reviews,
        signatures,
    ) = _trusted_rows()
    signatures.pop(next(iter(signatures)))

    with pytest.raises(ValueError, match="signatures scenario set differs"):
        _evaluate_gold_review_capacity(
            scenarios=scenarios,
            packets=packets,
            reviews=reviews,
            signatures=signatures,
            split_policy=policy,
            reviewer_public_key=reviewer.public_key(),
            reviewer_trust_policy=trust_policy,
            owner_public_key=owner.public_key(),
            minimum_train_cases=1,
            minimum_validation_cases=1,
            minimum_holdout_cases=1,
        )
