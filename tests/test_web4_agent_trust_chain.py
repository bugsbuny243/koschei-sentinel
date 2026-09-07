import json
from pathlib import Path

import pytest

from koschei_sentinel.web4_agent_trust_chain import (
    Web4AgentTrustChainReport,
    evaluate_web4_agent_trust_chain,
)

_ROOT = Path(__file__).resolve().parents[1]


def _paths(root: Path) -> dict[str, Path]:
    return {
        "source_registry_path": root / "configs/corpus/web4-v1.sources.proposed.jsonl",
        "protocol_tracking_path": root / "configs/corpus/web4-v1.protocol-tracking.json",
        "benchmark_path": root / "evals/web4-security-benchmark.v1.json",
        "trust_chain_map_path": root / "evals/web4-agent-trust-chain.v1.json",
    }


def _evaluate(root: Path = _ROOT) -> Web4AgentTrustChainReport:
    return evaluate_web4_agent_trust_chain(**_paths(root))


def _copy_inputs(tmp_path: Path) -> dict[str, Path]:
    paths = _paths(tmp_path)
    original = _paths(_ROOT)
    for key, destination in paths.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(original[key].read_bytes())
    return paths


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_agent_trust_chain_contract_is_research_ready_and_non_authorizing() -> None:
    report = _evaluate()

    assert report.ready_for_research is True
    assert report.training_authorization is False
    assert report.promotion_eligible is False
    assert report.production_activation_allowed is False
    assert report.chain_stage_count == 6
    assert report.family_count == 10
    assert report.source_ref_count == 10
    assert report.violations == []


def test_agent_trust_chain_receipt_rejects_tampering() -> None:
    report = _evaluate()
    tampered = report.model_dump(mode="json")
    tampered["family_count"] += 1

    with pytest.raises(ValueError, match="self-hash does not verify"):
        Web4AgentTrustChainReport.model_validate(tampered)


def test_unverified_a2a_revision_drift_blocks_readiness(tmp_path: Path) -> None:
    paths = _copy_inputs(tmp_path)
    tracking = json.loads(paths["protocol_tracking_path"].read_text(encoding="utf-8"))
    family = next(
        row
        for row in tracking["families"]
        if row["document_id"] == "draft-tonyai-a2a-trust"
    )
    family["current_observed_revision"] = "03"
    _write_json(paths["protocol_tracking_path"], tracking)

    report = evaluate_web4_agent_trust_chain(**paths)

    assert report.ready_for_research is False
    assert any("verified revision drifted: draft-tonyai-a2a-trust" in row for row in report.violations)


def test_training_authorization_drift_on_bound_source_blocks_readiness(tmp_path: Path) -> None:
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
    source["training_authorization"] = True
    paths["source_registry_path"].write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = evaluate_web4_agent_trust_chain(**paths)

    assert report.ready_for_research is False
    assert any("training authorization opened" in row for row in report.violations)


def test_w3c_working_draft_cannot_be_promoted_to_standard(tmp_path: Path) -> None:
    paths = _copy_inputs(tmp_path)
    rows = [
        json.loads(line)
        for line in paths["source_registry_path"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source = next(
        row
        for row in rows
        if row["source_id"] == "w3c.vc-forgery-defense-20260714"
    )
    source["authority_tier"] = "T0_STANDARD"
    paths["source_registry_path"].write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = evaluate_web4_agent_trust_chain(**paths)

    assert report.ready_for_research is False
    assert any("W3C Working Draft elevated to standard" in row for row in report.violations)


def test_unknown_benchmark_parent_blocks_readiness(tmp_path: Path) -> None:
    paths = _copy_inputs(tmp_path)
    trust_map = json.loads(paths["trust_chain_map_path"].read_text(encoding="utf-8"))
    trust_map["families"][0]["benchmark_parent_family"] = "not-a-real-family"
    _write_json(paths["trust_chain_map_path"], trust_map)

    report = evaluate_web4_agent_trust_chain(**paths)

    assert report.ready_for_research is False
    assert any("unknown benchmark parent" in row for row in report.violations)


def test_required_agent_trust_subfamily_cannot_disappear(tmp_path: Path) -> None:
    paths = _copy_inputs(tmp_path)
    trust_map = json.loads(paths["trust_chain_map_path"].read_text(encoding="utf-8"))
    trust_map["families"] = trust_map["families"][:-1]
    _write_json(paths["trust_chain_map_path"], trust_map)

    report = evaluate_web4_agent_trust_chain(**paths)

    assert report.ready_for_research is False
    assert "Agent Trust Chain required subfamilies are missing" in report.violations
