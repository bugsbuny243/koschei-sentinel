from __future__ import annotations

import hashlib
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.shadow_baseline import (
    ShadowBaselineBlocked,
    apply_shadow_baseline_advance,
    approve_shadow_baseline_proposal,
    build_shadow_baseline_proposal,
    verify_shadow_baseline_approval,
    verify_shadow_baseline_lineage,
)
from koschei_sentinel.shadow_receipt import ShadowReplayReceipt
from koschei_sentinel.shadow_regression import build_shadow_regression_report
from koschei_sentinel.shadow_review import ShadowReviewScorecard, ShadowReviewThresholds


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _receipt(candidate: str, results: str) -> ShadowReplayReceipt:
    payload = {
        "schema_version": "sentinel.shadow-replay-receipt.v1",
        "candidate_id": candidate,
        "state": "completed_shadow_replay",
        "stage": "shadow_research_candidate",
        "authority": "explanation_only",
        "plan_digest": hashlib.sha256(f"plan:{candidate}".encode()).hexdigest(),
        "proposal_digest": hashlib.sha256(f"proposal:{candidate}".encode()).hexdigest(),
        "approval_digest": hashlib.sha256(f"approval:{candidate}".encode()).hexdigest(),
        "replay_sha256": "a" * 64,
        "replay_cases": 2,
        "results_path": f"build/shadow/{candidate}/results.jsonl",
        "results_sha256": results,
        "results_cases": 2,
        "output_dir": f"build/shadow/{candidate}",
        "complete_case_coverage": True,
        "ordered_case_identity_match": True,
        "manual_review_required": True,
        "benchmark_recheck_required": True,
        "automatic_promotion_allowed": False,
        "automatic_deployment_allowed": False,
        "production_deployment_allowed": False,
        "web3_runtime_integration_allowed": False,
        "verdict_mutation_allowed": False,
    }
    return ShadowReplayReceipt.model_validate({**payload, "receipt_digest": _digest(payload)})


def _scorecard(receipt: ShadowReplayReceipt, *, gate: bool = True) -> ShadowReviewScorecard:
    payload = {
        "schema_version": "sentinel.shadow-review-scorecard.v1",
        "candidate_id": receipt.candidate_id,
        "state": "reviewed_shadow_replay",
        "authority": "explanation_only",
        "receipt_digest": receipt.receipt_digest,
        "plan_digest": receipt.plan_digest,
        "results_sha256": receipt.results_sha256,
        "review_sha256": hashlib.sha256(f"review:{receipt.candidate_id}".encode()).hexdigest(),
        "reviewers": ["owner-review@koschei"],
        "total_cases": 2,
        "passed_cases": 2 if gate else 1,
        "failed_cases": 0 if gate else 1,
        "case_pass_rate": 1.0 if gate else 0.5,
        "authority_score": 1.0 if gate else 0.5,
        "grounding_score": 1.0 if gate else 0.5,
        "abstention_score": 1.0 if gate else 0.5,
        "privacy_score": 1.0 if gate else 0.5,
        "followup_cases": [] if gate else ["case-2"],
        "thresholds": ShadowReviewThresholds().model_dump(mode="json"),
        "gate_passed": gate,
        "complete_manual_review": True,
        "benchmark_recheck_required": True,
        "owner_decision_required": True,
        "automatic_promotion_allowed": False,
        "automatic_deployment_allowed": False,
        "production_deployment_allowed": False,
        "web3_runtime_integration_allowed": False,
        "verdict_mutation_allowed": False,
    }
    return ShadowReviewScorecard.model_validate({**payload, "scorecard_digest": _digest(payload)})


def _passing_report(
    baseline_receipt: ShadowReplayReceipt,
    candidate_receipt: ShadowReplayReceipt,
):
    baseline = _scorecard(baseline_receipt)
    candidate = _scorecard(candidate_receipt)
    report = build_shadow_regression_report(
        baseline,
        baseline_receipt,
        candidate,
        candidate_receipt,
    )
    assert report.regression_passed is True
    return report, baseline, candidate


def test_owner_signed_lineage_seeds_and_advances_without_automatic_selection() -> None:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    receipt_a = _receipt("candidate-a", "1" * 64)
    receipt_b = _receipt("candidate-b", "2" * 64)
    report_ab, score_a, score_b = _passing_report(receipt_a, receipt_b)
    proposal_ab = build_shadow_baseline_proposal(
        report_ab,
        score_a,
        receipt_a,
        score_b,
        receipt_b,
        public_key,
    )
    approval_ab = approve_shadow_baseline_proposal(
        proposal_ab,
        private_key,
        approver_id="owner@koschei",
    )
    lineage = apply_shadow_baseline_advance(
        proposal_ab,
        approval_ab,
        report_ab,
        score_a,
        receipt_a,
        score_b,
        receipt_b,
        public_key,
    )
    assert [entry.candidate_id for entry in lineage.entries] == ["candidate-a", "candidate-b"]
    assert lineage.entries[0].kind == "seed"
    assert lineage.head_candidate_id == "candidate-b"
    assert lineage.automatic_baseline_selection_allowed is False
    assert lineage.production_deployment_allowed is False

    receipt_c = _receipt("candidate-c", "3" * 64)
    report_bc, score_b_again, score_c = _passing_report(receipt_b, receipt_c)
    assert score_b_again.scorecard_digest == score_b.scorecard_digest
    proposal_bc = build_shadow_baseline_proposal(
        report_bc,
        score_b_again,
        receipt_b,
        score_c,
        receipt_c,
        public_key,
        lineage=lineage,
    )
    approval_bc = approve_shadow_baseline_proposal(
        proposal_bc,
        private_key,
        approver_id="owner@koschei",
    )
    advanced = apply_shadow_baseline_advance(
        proposal_bc,
        approval_bc,
        report_bc,
        score_b_again,
        receipt_b,
        score_c,
        receipt_c,
        public_key,
        lineage=lineage,
    )
    assert [entry.candidate_id for entry in advanced.entries] == [
        "candidate-a",
        "candidate-b",
        "candidate-c",
    ]
    assert advanced.entries[-1].parent_candidate_id == "candidate-b"
    assert advanced.head_candidate_id == "candidate-c"
    verify_shadow_baseline_lineage(advanced)


def test_failed_regression_cannot_become_baseline_proposal() -> None:
    private_key = Ed25519PrivateKey.generate()
    receipt_a = _receipt("candidate-a", "1" * 64)
    receipt_b = _receipt("candidate-b", "2" * 64)
    score_a = _scorecard(receipt_a)
    score_b = _scorecard(receipt_b, gate=False)
    report = build_shadow_regression_report(score_a, receipt_a, score_b, receipt_b)
    assert report.regression_passed is False
    with pytest.raises(ShadowBaselineBlocked, match="did not pass"):
        build_shadow_baseline_proposal(
            report,
            score_a,
            receipt_a,
            score_b,
            receipt_b,
            private_key.public_key(),
        )


def test_lineage_rejects_historical_candidate_cycle_and_wrong_head() -> None:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    receipt_a = _receipt("candidate-a", "1" * 64)
    receipt_b = _receipt("candidate-b", "2" * 64)
    report_ab, score_a, score_b = _passing_report(receipt_a, receipt_b)
    proposal = build_shadow_baseline_proposal(
        report_ab, score_a, receipt_a, score_b, receipt_b, public_key
    )
    approval = approve_shadow_baseline_proposal(proposal, private_key, approver_id="owner")
    lineage = apply_shadow_baseline_advance(
        proposal,
        approval,
        report_ab,
        score_a,
        receipt_a,
        score_b,
        receipt_b,
        public_key,
    )

    report_ba, score_b_again, score_a_again = _passing_report(receipt_b, receipt_a)
    with pytest.raises(ShadowBaselineBlocked, match="already exists"):
        build_shadow_baseline_proposal(
            report_ba,
            score_b_again,
            receipt_b,
            score_a_again,
            receipt_a,
            public_key,
            lineage=lineage,
        )

    receipt_c = _receipt("candidate-c", "3" * 64)
    report_ac, score_a_again, score_c = _passing_report(receipt_a, receipt_c)
    with pytest.raises(ShadowBaselineBlocked, match="current lineage head"):
        build_shadow_baseline_proposal(
            report_ac,
            score_a_again,
            receipt_a,
            score_c,
            receipt_c,
            public_key,
            lineage=lineage,
        )


def test_tampered_lineage_and_wrong_owner_key_fail_closed() -> None:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    receipt_a = _receipt("candidate-a", "1" * 64)
    receipt_b = _receipt("candidate-b", "2" * 64)
    report, score_a, score_b = _passing_report(receipt_a, receipt_b)
    proposal = build_shadow_baseline_proposal(
        report, score_a, receipt_a, score_b, receipt_b, public_key
    )
    approval = approve_shadow_baseline_proposal(proposal, private_key, approver_id="owner")
    lineage = apply_shadow_baseline_advance(
        proposal,
        approval,
        report,
        score_a,
        receipt_a,
        score_b,
        receipt_b,
        public_key,
    )

    tampered = lineage.model_copy(update={"head_candidate_id": "candidate-x"})
    with pytest.raises(ShadowBaselineBlocked, match="digest"):
        verify_shadow_baseline_lineage(tampered)

    wrong_key = Ed25519PrivateKey.generate().public_key()
    with pytest.raises(ShadowBaselineBlocked, match="public key"):
        verify_shadow_baseline_approval(proposal, approval, wrong_key)
