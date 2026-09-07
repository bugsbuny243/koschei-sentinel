import json
from pathlib import Path

import pytest

from koschei_sentinel.web4_agent_trust_authoring import (
    Web4AgentTrustAuthoringReport,
    evaluate_web4_agent_trust_authoring_plan,
)

_ROOT = Path(__file__).resolve().parents[1]


def _paths(root: Path) -> dict[str, Path]:
    return {
        "source_registry_path": root / "configs/corpus/web4-v1.sources.proposed.jsonl",
        "trust_chain_map_path": root / "evals/web4-agent-trust-chain.v1.json",
        "intake_policy_path": root / "evals/web4-benchmark-intake-policy.v1.json",
        "authoring_plan_path": root / "evals/web4-agent-trust-authoring-plan.v1.json",
    }


def _evaluate(root: Path = _ROOT) -> Web4AgentTrustAuthoringReport:
    return evaluate_web4_agent_trust_authoring_plan(**_paths(root))


def _copy_inputs(tmp_path: Path) -> dict[str, Path]:
    paths = _paths(tmp_path)
    originals = _paths(_ROOT)
    for key, destination in paths.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(originals[key].read_bytes())
    return paths


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_agent_trust_authoring_plan_is_valid_and_non_authorizing() -> None:
    report = _evaluate()

    assert report.plan_valid is True
    assert report.training_authorization is False
    assert report.evaluation_authorization is False
    assert report.promotion_eligible is False
    assert report.production_activation_allowed is False
    assert report.subfamily_count == 10
    assert report.p0_subfamily_count == 8
    assert report.p1_subfamily_count == 2
    assert report.planning_target_holdout_cases == 120
    assert report.expected_proposal_count == 600
    assert report.source_ref_count == 10
    assert report.violations == []


def test_agent_trust_authoring_report_rejects_tampering() -> None:
    report = _evaluate()
    tampered = report.model_dump(mode="json")
    tampered["expected_proposal_count"] += 1

    with pytest.raises(ValueError, match="self-hash does not verify"):
        Web4AgentTrustAuthoringReport.model_validate(tampered)


def test_holdout_rate_drift_blocks_authoring_plan(tmp_path: Path) -> None:
    paths = _copy_inputs(tmp_path)
    policy = json.loads(paths["intake_policy_path"].read_text(encoding="utf-8"))
    policy["development_bps"] = 6500
    policy["holdout_bps"] = 1500
    _write_json(paths["intake_policy_path"], policy)

    report = evaluate_web4_agent_trust_authoring_plan(**paths)

    assert report.plan_valid is False
    assert "Agent Trust authoring HOLDOUT rate differs from intake policy" in report.violations


def test_automatic_gold_generation_cannot_be_enabled(tmp_path: Path) -> None:
    paths = _copy_inputs(tmp_path)
    plan = json.loads(paths["authoring_plan_path"].read_text(encoding="utf-8"))
    plan["automatic_gold_generation_allowed"] = True
    _write_json(paths["authoring_plan_path"], plan)

    report = evaluate_web4_agent_trust_authoring_plan(**paths)

    assert report.plan_valid is False
    assert any("automatic_gold_generation_allowed" in row for row in report.violations)


def test_p0_must_keep_aid_aip_a2a_and_acs(tmp_path: Path) -> None:
    paths = _copy_inputs(tmp_path)
    plan = json.loads(paths["authoring_plan_path"].read_text(encoding="utf-8"))
    moved = plan["priority_source_groups"]["P0"].pop()
    plan["priority_source_groups"]["P1"].append(moved)
    _write_json(paths["authoring_plan_path"], plan)

    report = evaluate_web4_agent_trust_authoring_plan(**paths)

    assert report.plan_valid is False
    assert "AID/AIP/A2A/ACS sources are not all P0" in report.violations


def test_source_license_review_cannot_be_bypassed(tmp_path: Path) -> None:
    paths = _copy_inputs(tmp_path)
    rows = [
        json.loads(line)
        for line in paths["source_registry_path"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source = next(
        row
        for row in rows
        if row["source_id"] == "ietf.draft.nemethi.aid-agent-identity-discovery-00"
    )
    source["license_status"] = "APPROVED"
    paths["source_registry_path"].write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = evaluate_web4_agent_trust_authoring_plan(**paths)

    assert report.plan_valid is False
    assert any("bypasses license review" in row for row in report.violations)


def test_subfamily_source_refs_must_match_trust_chain(tmp_path: Path) -> None:
    paths = _copy_inputs(tmp_path)
    plan = json.loads(paths["authoring_plan_path"].read_text(encoding="utf-8"))
    plan["subfamilies"][0]["required_source_refs"] = [
        "ietf.draft.nemethi.aid-agent-identity-discovery-00"
    ]
    _write_json(paths["authoring_plan_path"], plan)

    report = evaluate_web4_agent_trust_authoring_plan(**paths)

    assert report.plan_valid is False
    assert any("source refs drifted" in row for row in report.violations)
