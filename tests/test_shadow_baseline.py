from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.shadow_baseline import (
    ShadowBaselineApproval,
    ShadowBaselineBlocked,
    ShadowBaselineLineage,
    apply_shadow_baseline_advance,
    approve_shadow_baseline_proposal,
    build_shadow_baseline_proposal,
    verify_shadow_baseline_approval,
    verify_shadow_baseline_lineage,
)
from koschei_sentinel.shadow_baseline_claim import (
    build_shadow_baseline_successor_claim,
    claim_shadow_baseline_successor,
)
from koschei_sentinel.shadow_receipt import ShadowReplayReceipt
from koschei_sentinel.shadow_regression import (
    ShadowRegressionReport,
    build_shadow_regression_report,
)
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


def _scorecard(
    receipt: ShadowReplayReceipt,
    *,
    score: float = 1.0,
    thresholds: ShadowReviewThresholds | None = None,
    gate: bool | None = None,
) -> ShadowReviewScorecard:
    active = thresholds or ShadowReviewThresholds()
    passed = round(2 * score)
    derived_gate = (
        score >= active.min_case_pass_rate
        and score >= active.min_authority_score
        and score >= active.min_grounding_score
        and score >= active.min_abstention_score
        and score >= active.min_privacy_score
    )
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
        "passed_cases": passed,
        "failed_cases": 2 - passed,
        "case_pass_rate": score,
        "authority_score": score,
        "grounding_score": score,
        "abstention_score": score,
        "privacy_score": score,
        "followup_cases": [],
        "thresholds": active.model_dump(mode="json"),
        "gate_passed": derived_gate if gate is None else gate,
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


def _seed_lineage(
    private_key: Ed25519PrivateKey,
) -> tuple[
    ShadowBaselineLineage,
    ShadowReplayReceipt,
    ShadowReviewScorecard,
]:
    public_key = private_key.public_key()
    receipt_a = _receipt("candidate-a", "1" * 64)
    receipt_b = _receipt("candidate-b", "2" * 64)
    report_ab, score_a, score_b = _passing_report(receipt_a, receipt_b)
    proposal = build_shadow_baseline_proposal(
        report_ab,
        score_a,
        receipt_a,
        score_b,
        receipt_b,
        public_key,
    )
    approval = approve_shadow_baseline_proposal(
        proposal,
        private_key,
        approver_id="owner@koschei",
    )
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
    return lineage, receipt_b, score_b


def test_owner_signed_lineage_seeds_and_advances_without_automatic_selection() -> None:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    lineage, receipt_b, score_b = _seed_lineage(private_key)

    assert [entry.candidate_id for entry in lineage.entries] == ["candidate-a", "candidate-b"]
    assert lineage.entries[0].kind == "seed"
    assert lineage.entries[1].owner_signature_base64 is not None
    assert lineage.historical_signatures_verified is True
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
    verify_shadow_baseline_lineage(advanced, public_key)


def test_failed_review_cannot_become_baseline_proposal() -> None:
    private_key = Ed25519PrivateKey.generate()
    receipt_a = _receipt("candidate-a", "1" * 64)
    receipt_b = _receipt("candidate-b", "2" * 64)
    score_a = _scorecard(receipt_a)
    score_b = _scorecard(receipt_b, score=0.5, gate=False)
    report = build_shadow_regression_report(score_a, receipt_a, score_b, receipt_b)
    assert report.regression_passed is False
    with pytest.raises(ShadowBaselineBlocked, match="passing human review scorecards"):
        build_shadow_baseline_proposal(
            report,
            score_a,
            receipt_a,
            score_b,
            receipt_b,
            private_key.public_key(),
        )


def test_forged_regression_verdict_is_recomputed_from_scorecards() -> None:
    private_key = Ed25519PrivateKey.generate()
    thresholds = ShadowReviewThresholds(
        min_case_pass_rate=0.5,
        min_authority_score=0.5,
        min_grounding_score=0.5,
        min_abstention_score=0.5,
        min_privacy_score=0.5,
    )
    receipt_a = _receipt("candidate-a", "1" * 64)
    receipt_b = _receipt("candidate-b", "2" * 64)
    baseline = _scorecard(receipt_a, score=1.0, thresholds=thresholds)
    candidate = _scorecard(receipt_b, score=0.5, thresholds=thresholds)
    report = build_shadow_regression_report(
        baseline,
        receipt_a,
        candidate,
        receipt_b,
    )
    assert candidate.gate_passed is True
    assert report.regression_passed is False

    payload = report.model_dump(mode="json")
    payload.pop("report_digest")
    payload["regression_reasons"] = []
    payload["regression_passed"] = True
    forged = ShadowRegressionReport.model_validate(
        {**payload, "report_digest": _digest(payload)}
    )
    with pytest.raises(ShadowBaselineBlocked, match="does not match supplied scorecards"):
        build_shadow_baseline_proposal(
            forged,
            baseline,
            receipt_a,
            candidate,
            receipt_b,
            private_key.public_key(),
        )


def test_approver_identity_is_covered_by_owner_signature() -> None:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    receipt_a = _receipt("candidate-a", "1" * 64)
    receipt_b = _receipt("candidate-b", "2" * 64)
    report, score_a, score_b = _passing_report(receipt_a, receipt_b)
    proposal = build_shadow_baseline_proposal(
        report,
        score_a,
        receipt_a,
        score_b,
        receipt_b,
        public_key,
    )
    approval = approve_shadow_baseline_proposal(
        proposal,
        private_key,
        approver_id="owner-a",
    )
    payload = approval.model_dump(mode="json")
    payload.pop("approval_digest")
    payload["approver_id"] = "owner-b"
    forged = ShadowBaselineApproval.model_validate(
        {**payload, "approval_digest": _digest(payload)}
    )
    with pytest.raises(ShadowBaselineBlocked, match="signature verification"):
        verify_shadow_baseline_approval(proposal, forged, public_key)


def test_lineage_rejects_historical_candidate_cycle_and_wrong_head() -> None:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    lineage, receipt_b, _ = _seed_lineage(private_key)
    receipt_a = _receipt("candidate-a", "1" * 64)

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


def test_forged_historical_signature_fails_even_with_recomputed_lineage_digest() -> None:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    lineage, _, _ = _seed_lineage(private_key)

    payload = lineage.model_dump(mode="json")
    payload.pop("lineage_digest")
    payload["entries"][1]["owner_signature_base64"] = "A" * 88
    approval_payload = {
        "schema_version": "sentinel.shadow-baseline-approval.v1",
        "state": "owner_approved_baseline_advance",
        "authority": "explanation_only",
        "candidate_id": payload["entries"][1]["candidate_id"],
        "approver_id": payload["entries"][1]["approver_id"],
        "owner_key_fingerprint": payload["owner_key_fingerprint"],
        "proposal_digest": payload["entries"][1]["proposal_digest"],
        "signature_algorithm": "ed25519",
        "signature_base64": "A" * 88,
        "signature_verified": True,
        "automatic_baseline_selection_allowed": False,
        "automatic_promotion_allowed": False,
        "production_deployment_allowed": False,
        "web3_runtime_integration_allowed": False,
    }
    payload["entries"][1]["owner_approval_digest"] = _digest(approval_payload)
    forged = ShadowBaselineLineage.model_validate(
        {**payload, "lineage_digest": _digest(payload)}
    )
    with pytest.raises(ShadowBaselineBlocked, match="signature verification"):
        verify_shadow_baseline_lineage(forged, public_key)


def test_canonical_successor_claim_prevents_two_heads_from_same_predecessor(
    tmp_path: Path,
) -> None:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    lineage, receipt_b, _ = _seed_lineage(private_key)
    claims = tmp_path / "claims"

    successors = []
    for candidate_id, results in (("candidate-c", "3" * 64), ("candidate-d", "4" * 64)):
        receipt = _receipt(candidate_id, results)
        report, score_b, score_candidate = _passing_report(receipt_b, receipt)
        proposal = build_shadow_baseline_proposal(
            report,
            score_b,
            receipt_b,
            score_candidate,
            receipt,
            public_key,
            lineage=lineage,
        )
        approval = approve_shadow_baseline_proposal(
            proposal,
            private_key,
            approver_id="owner@koschei",
        )
        successor = apply_shadow_baseline_advance(
            proposal,
            approval,
            report,
            score_b,
            receipt_b,
            score_candidate,
            receipt,
            public_key,
            lineage=lineage,
        )
        successors.append((proposal, successor))

    first_claim = build_shadow_baseline_successor_claim(*successors[0])
    first_path = claim_shadow_baseline_successor(first_claim, claims)
    assert first_path.exists()
    assert claim_shadow_baseline_successor(first_claim, claims) == first_path

    second_claim = build_shadow_baseline_successor_claim(*successors[1])
    with pytest.raises(ShadowBaselineBlocked, match="different claimed successor"):
        claim_shadow_baseline_successor(second_claim, claims)


def test_tampered_lineage_and_wrong_owner_key_fail_closed() -> None:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    lineage, _, _ = _seed_lineage(private_key)

    tampered = lineage.model_copy(update={"head_candidate_id": "candidate-x"})
    with pytest.raises(ShadowBaselineBlocked, match="digest"):
        verify_shadow_baseline_lineage(tampered, public_key)

    wrong_key = Ed25519PrivateKey.generate().public_key()
    with pytest.raises(ShadowBaselineBlocked, match="public key"):
        verify_shadow_baseline_lineage(lineage, wrong_key)
