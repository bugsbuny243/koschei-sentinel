from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.promotion import public_key_fingerprint
from koschei_sentinel.training import canonical_json
from koschei_sentinel.web4_holdout_capacity import (
    Web4HoldoutCapacityReport,
    build_web4_holdout_capacity_report,
)
from koschei_sentinel.web4_holdout_release import (
    Web4HoldoutRelease,
    Web4HoldoutReleaseCase,
)

_ROOT = Path(__file__).resolve().parents[1]
_BENCHMARK = _ROOT / "evals/web4-security-benchmark.v1.json"
_SIGNATURE_CONTEXT = b"koschei-sentinel-web4-holdout-release-v1\0"


def _digest(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _policy() -> dict[str, object]:
    return json.loads(_BENCHMARK.read_text(encoding="utf-8"))


def _case(case_id: str, family: str) -> Web4HoldoutReleaseCase:
    source_ref = "fixture-source"
    payload: dict[str, object] = {
        "schema_version": "sentinel.web4-holdout-release-case.v1",
        "case_id": case_id,
        "family": family,
        "split": "HOLDOUT",
        "created_at": "2026-09-07T00:00:00+00:00",
        "packet_sha256": _digest([case_id, "packet"]),
        "model_input": {"messages": [{"role": "user", "content": case_id}]},
        "model_input_sha256": _digest([case_id, "model-input"]),
        "answer_key_sha256": _digest([case_id, "answer-key"]),
        "intake_policy_sha256": _digest("intake-policy"),
        "split_seed": "sentinel-web4-benchmark-split-v1",
        "split_material_sha256": _digest([case_id, "split-material"]),
        "source_refs": [source_ref],
        "source_revision_status": {source_ref: "PINNED_FIXTURE"},
        "source_snapshot_sha256s": {source_ref: _digest([case_id, "snapshot"])},
        "review_sha256": _digest([case_id, "review"]),
        "review_artifact_sha256": _digest([case_id, "review-artifact"]),
        "reviewer_id": "fixture-reviewer",
        "reviewer_key_fingerprint": _digest("reviewer-key"),
        "reviewer_trust_policy_digest": _digest("reviewer-policy"),
        "adjudication_sha256": _digest([case_id, "adjudication"]),
        "adjudication_artifact_sha256": _digest([case_id, "adjudication-artifact"]),
        "adjudicator_id": "fixture-adjudicator",
        "adjudicator_key_fingerprint": _digest("adjudicator-key"),
        "adjudicator_trust_policy_digest": _digest("adjudicator-policy"),
        "human_reviewed": True,
        "independent_adjudication": True,
        "final_review_approved": True,
        "contains_answer_key": False,
        "research_evaluation_authorization": True,
        "training_authorization": False,
        "promotion_eligible": False,
        "production_activation_allowed": False,
    }
    payload["case_sha256"] = _digest(payload)
    return Web4HoldoutReleaseCase.model_validate(payload)


def _signed_release(
    owner: Ed25519PrivateKey,
    family_counts: dict[str, int],
) -> Web4HoldoutRelease:
    cases: list[Web4HoldoutReleaseCase] = []
    serial = 0
    for family in sorted(family_counts):
        for _ in range(family_counts[family]):
            serial += 1
            cases.append(_case(f"capacity-{serial:04d}", family))
    cases.sort(key=lambda row: row.case_id)
    policy_sha = hashlib.sha256(_BENCHMARK.read_bytes()).hexdigest()
    payload: dict[str, object] = {
        "schema_version": "sentinel.web4-holdout-release.v1",
        "release_id": "web4-capacity-fixture-v1",
        "state": "owner_signed_research_holdout",
        "benchmark_policy_sha256": policy_sha,
        "source_registry_sha256": _digest("fixture-source-registry"),
        "case_ids": [case.case_id for case in cases],
        "cases": [case.model_dump(mode="json") for case in cases],
        "case_count": len(cases),
        "answer_keys_isolated": True,
        "all_cases_human_reviewed": True,
        "all_cases_independently_adjudicated": True,
        "deterministic_holdout_only": True,
        "research_evaluation_authorization": True,
        "training_authorization": False,
        "promotion_eligible": False,
        "production_activation_allowed": False,
        "owner_key_fingerprint": public_key_fingerprint(owner.public_key()),
    }
    release_sha = _digest(payload)
    signature = owner.sign(_SIGNATURE_CONTEXT + release_sha.encode("ascii"))
    signed: dict[str, object] = {
        **payload,
        "release_sha256": release_sha,
        "signature_algorithm": "ed25519",
        "owner_signature_base64": base64.b64encode(signature).decode("ascii"),
        "owner_signature_verified": True,
    }
    signed["artifact_sha256"] = _digest(signed)
    return Web4HoldoutRelease.model_validate(signed)


def test_capacity_reports_real_policy_shortfall_without_granting_authority() -> None:
    owner = Ed25519PrivateKey.generate()
    first_family = str(_policy()["required_families"][0])
    release = _signed_release(owner, {first_family: 1})
    report = build_web4_holdout_capacity_report(
        release=release,
        owner_public_key=owner.public_key(),
        benchmark_policy_path=_BENCHMARK,
    )

    assert report.release_case_count == 1
    assert report.desired_minimum_total_cases == 240
    assert report.total_case_shortfall == 239
    assert report.required_family_count == 25
    assert report.minimum_cases_per_required_family == 12
    assert report.family_counts[first_family] == 1
    assert report.family_shortfalls[first_family] == 11
    assert report.covered_family_count == 1
    assert len(report.missing_families) == 24
    assert report.research_benchmark_capacity_ready is False
    assert report.blockers
    assert report.model_execution_authorized is False
    assert report.training_authorization is False
    assert report.promotion_eligible is False
    assert report.production_activation_allowed is False


def test_capacity_ready_requires_every_family_minimum_and_total_target() -> None:
    owner = Ed25519PrivateKey.generate()
    families = [str(item) for item in _policy()["required_families"]]
    release = _signed_release(owner, {family: 12 for family in families})
    report = build_web4_holdout_capacity_report(
        release=release,
        owner_public_key=owner.public_key(),
        benchmark_policy_path=_BENCHMARK,
    )

    assert report.release_case_count == 300
    assert report.total_case_shortfall == 0
    assert report.covered_family_count == 25
    assert report.families_at_minimum_count == 25
    assert report.missing_families == []
    assert set(report.family_shortfalls.values()) == {0}
    assert report.blockers == []
    assert report.research_benchmark_capacity_ready is True
    assert report.training_authorization is False


def test_capacity_fails_family_minimum_even_when_total_target_is_met() -> None:
    owner = Ed25519PrivateKey.generate()
    families = [str(item) for item in _policy()["required_families"]]
    counts = {family: 12 for family in families}
    counts[families[0]] = 11
    counts[families[1]] = 13
    release = _signed_release(owner, counts)
    report = build_web4_holdout_capacity_report(
        release=release,
        owner_public_key=owner.public_key(),
        benchmark_policy_path=_BENCHMARK,
    )

    assert report.release_case_count == 300
    assert report.total_case_shortfall == 0
    assert report.family_shortfalls[families[0]] == 1
    assert report.families_at_minimum_count == 24
    assert report.research_benchmark_capacity_ready is False
    assert any("1 families" in blocker for blocker in report.blockers)


def test_capacity_rejects_owner_signed_unknown_family() -> None:
    owner = Ed25519PrivateKey.generate()
    release = _signed_release(owner, {"not-a-policy-family": 1})
    with pytest.raises(ValueError, match="unknown family"):
        build_web4_holdout_capacity_report(
            release=release,
            owner_public_key=owner.public_key(),
            benchmark_policy_path=_BENCHMARK,
        )


def test_capacity_rejects_wrong_owner_and_tampered_report() -> None:
    owner = Ed25519PrivateKey.generate()
    first_family = str(_policy()["required_families"][0])
    release = _signed_release(owner, {first_family: 1})
    with pytest.raises(ValueError, match="owner key fingerprint"):
        build_web4_holdout_capacity_report(
            release=release,
            owner_public_key=Ed25519PrivateKey.generate().public_key(),
            benchmark_policy_path=_BENCHMARK,
        )

    report = build_web4_holdout_capacity_report(
        release=release,
        owner_public_key=owner.public_key(),
        benchmark_policy_path=_BENCHMARK,
    )
    tampered = report.model_dump(mode="json")
    tampered["capacity_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="self-hash"):
        Web4HoldoutCapacityReport.model_validate(tampered)
