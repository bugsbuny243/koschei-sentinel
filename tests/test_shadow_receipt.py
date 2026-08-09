from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from koschei_sentinel.shadow_receipt import (
    ShadowReceiptBlocked,
    build_shadow_replay_receipt,
    load_shadow_replay_receipt,
    write_shadow_replay_receipt,
)
from koschei_sentinel.shadow_replay import ShadowReplayPlan


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def _plan(tmp_path: Path) -> ShadowReplayPlan:
    replay = tmp_path / "replay.jsonl"
    replay.write_text(
        '{"case_id":"case-1"}\n{"case_id":"case-2"}\n',
        encoding="utf-8",
    )
    (tmp_path / "output").mkdir(exist_ok=True)
    payload = {
        "schema_version": "sentinel.shadow-replay-plan.v1",
        "candidate_id": "sentinel-shadow-v1",
        "state": "sealed_shadow_replay",
        "stage": "shadow_research_candidate",
        "authority": "explanation_only",
        "proposal_digest": "1" * 64,
        "approval_digest": "2" * 64,
        "promotion_policy_digest": "0" * 64,
        "finalization_digest": "3" * 64,
        "adapter_digest": "4" * 64,
        "benchmark_suite_digest": "5" * 64,
        "benchmark_report_digest": "6" * 64,
        "registry_digest": "7" * 64,
        "owner_key_fingerprint": "8" * 64,
        "approver_id": "owner@koschei",
        "replay_path": "replay.jsonl",
        "replay_sha256": hashlib.sha256(replay.read_bytes()).hexdigest(),
        "replay_cases": 2,
        "output_dir": "output",
        "manual_dispatch_required": True,
        "network_access_allowed": False,
        "live_chain_reads_allowed": False,
        "live_customer_traffic_allowed": False,
        "verdict_mutation_allowed": False,
        "automatic_deployment_allowed": False,
        "production_deployment_allowed": False,
        "web3_runtime_integration_allowed": False,
    }
    return ShadowReplayPlan.model_validate({**payload, "plan_digest": _digest(payload)})


def _write_results(tmp_path: Path, plan: ShadowReplayPlan, ids: list[str]) -> Path:
    path = tmp_path / "output" / "results.jsonl"
    rows = [
        {
            "case_id": identifier,
            "candidate_id": plan.candidate_id,
            "plan_digest": plan.plan_digest,
            "authority": "explanation_only",
            "opinion": f"shadow-only:{identifier}",
        }
        for identifier in ids
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_receipt_binds_complete_ordered_shadow_results(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    results = _write_results(tmp_path, plan, ["case-1", "case-2"])

    receipt = build_shadow_replay_receipt(
        plan,
        results_path=results,
        root=tmp_path,
    )

    assert receipt.results_cases == 2
    assert receipt.complete_case_coverage is True
    assert receipt.ordered_case_identity_match is True
    assert receipt.manual_review_required is True
    assert receipt.benchmark_recheck_required is True
    assert receipt.automatic_promotion_allowed is False
    assert receipt.production_deployment_allowed is False
    assert receipt.web3_runtime_integration_allowed is False
    assert receipt.verdict_mutation_allowed is False
    assert receipt.results_sha256 == hashlib.sha256(results.read_bytes()).hexdigest()


def test_missing_or_reordered_result_identifiers_fail_closed(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    results = _write_results(tmp_path, plan, ["case-2", "case-1"])

    with pytest.raises(ShadowReceiptBlocked, match="sealed order"):
        build_shadow_replay_receipt(plan, results_path=results, root=tmp_path)

    results = _write_results(tmp_path, plan, ["case-1"])
    with pytest.raises(ShadowReceiptBlocked, match="sealed order"):
        build_shadow_replay_receipt(plan, results_path=results, root=tmp_path)


def test_candidate_plan_and_authority_escalation_are_rejected(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    results = tmp_path / "output" / "results.jsonl"
    rows = [
        {
            "case_id": "case-1",
            "candidate_id": "wrong-candidate",
            "plan_digest": plan.plan_digest,
        },
        {
            "case_id": "case-2",
            "candidate_id": plan.candidate_id,
            "plan_digest": plan.plan_digest,
        },
    ]
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ShadowReceiptBlocked, match="candidate_id"):
        build_shadow_replay_receipt(plan, results_path=results, root=tmp_path)

    rows[0]["candidate_id"] = plan.candidate_id
    rows[0]["production_deployment_allowed"] = True
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ShadowReceiptBlocked, match="production_deployment_allowed"):
        build_shadow_replay_receipt(plan, results_path=results, root=tmp_path)


def test_replay_drift_and_output_path_escape_are_rejected(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    results = _write_results(tmp_path, plan, ["case-1", "case-2"])
    (tmp_path / "replay.jsonl").write_text('{"case_id":"changed"}\n', encoding="utf-8")

    with pytest.raises(ShadowReceiptBlocked, match="replay bytes"):
        build_shadow_replay_receipt(plan, results_path=results, root=tmp_path)

    plan = _plan(tmp_path)
    outside = tmp_path / "outside-results.jsonl"
    outside.write_text(results.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ShadowReceiptBlocked, match="sealed output directory"):
        build_shadow_replay_receipt(plan, results_path=outside, root=tmp_path)


def test_receipt_digest_tamper_and_overwrite_fail_closed(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    results = _write_results(tmp_path, plan, ["case-1", "case-2"])
    receipt = build_shadow_replay_receipt(plan, results_path=results, root=tmp_path)
    path = tmp_path / "receipt.json"
    write_shadow_replay_receipt(receipt, path)

    with pytest.raises(FileExistsError):
        write_shadow_replay_receipt(receipt, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["manual_review_required"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid shadow replay receipt"):
        load_shadow_replay_receipt(path)