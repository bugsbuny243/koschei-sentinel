from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from koschei_sentinel.web4_benchmark_intake import (
    Web4BenchmarkIntakePacket,
    Web4BenchmarkSplit,
    build_web4_benchmark_intake,
)

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _ROOT / "fixtures/web4/benchmark-intake"
_PROPOSAL = _FIXTURE / "proposal.json"
_ANSWER_KEY = _FIXTURE / "answer-key.json"
_SOURCES = _ROOT / "configs/corpus/web4-v1.sources.proposed.jsonl"
_BENCHMARK = _ROOT / "evals/web4-security-benchmark.v1.json"
_INTAKE_POLICY = _ROOT / "evals/web4-benchmark-intake-policy.v1.json"


def _build(
    *,
    proposal_path: Path = _PROPOSAL,
    answer_key_path: Path = _ANSWER_KEY,
    source_registry_path: Path = _SOURCES,
    intake_policy_path: Path = _INTAKE_POLICY,
    snapshot_root: Path | None = None,
) -> Web4BenchmarkIntakePacket:
    return build_web4_benchmark_intake(
        proposal_path=proposal_path,
        answer_key_path=answer_key_path,
        source_registry_path=source_registry_path,
        benchmark_policy_path=_BENCHMARK,
        intake_policy_path=intake_policy_path,
        snapshot_root=snapshot_root,
    )


def test_intake_is_deterministic_and_keeps_answer_key_isolated() -> None:
    first = _build()
    second = _build()

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.split in {
        Web4BenchmarkSplit.DEVELOPMENT,
        Web4BenchmarkSplit.VALIDATION,
        Web4BenchmarkSplit.HOLDOUT,
    }
    assert first.review_status == "UNREVIEWED"
    assert first.human_reviewed is False
    assert first.contains_answer_key is False
    assert first.training_authorization is False
    assert first.evaluation_authorization is False
    assert first.promotion_eligible is False
    assert first.answer_key_sha256 == hashlib.sha256(_ANSWER_KEY.read_bytes()).hexdigest()

    serialized = json.dumps(first.model_dump(mode="json"), sort_keys=True)
    assert "REJECT_UNAUTHORIZED_WIDENING" not in serialized
    assert "must_request_new_grant_for_wider_scope" not in serialized


def test_intake_packet_rejects_tampering() -> None:
    packet = _build()
    tampered = packet.model_dump(mode="json")
    tampered["case_id"] = "tampered-web4-case"

    with pytest.raises(ValueError, match="self-hash does not verify"):
        Web4BenchmarkIntakePacket.model_validate(tampered)


def test_model_visible_input_rejects_answer_key_like_fields(tmp_path: Path) -> None:
    proposal = json.loads(_PROPOSAL.read_text(encoding="utf-8"))
    proposal["model_input"]["ground_truth"] = "must never be model-visible"
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(proposal), encoding="utf-8")

    with pytest.raises(ValueError, match="answer-key-like field"):
        _build(proposal_path=path, snapshot_root=_FIXTURE)


def test_proposal_cannot_manually_override_split(tmp_path: Path) -> None:
    proposal = json.loads(_PROPOSAL.read_text(encoding="utf-8"))
    proposal["split"] = "HOLDOUT"
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(proposal), encoding="utf-8")

    with pytest.raises(ValueError):
        _build(proposal_path=path, snapshot_root=_FIXTURE)


def test_unknown_source_ref_is_rejected(tmp_path: Path) -> None:
    proposal = json.loads(_PROPOSAL.read_text(encoding="utf-8"))
    proposal["sources"][0]["source_ref"] = "unknown.web4.source"
    answer_key = json.loads(_ANSWER_KEY.read_text(encoding="utf-8"))
    answer_key["source_refs"] = ["unknown.web4.source"]
    proposal_path = tmp_path / "proposal.json"
    answer_path = tmp_path / "answer-key.json"
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
    answer_path.write_text(json.dumps(answer_key), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown Web4 benchmark source_ref"):
        _build(
            proposal_path=proposal_path,
            answer_key_path=answer_path,
            snapshot_root=_FIXTURE,
        )


def test_snapshot_path_cannot_escape_intake_root(tmp_path: Path) -> None:
    proposal = json.loads(_PROPOSAL.read_text(encoding="utf-8"))
    proposal["sources"][0]["snapshot_path"] = "../answer-key.json"
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(proposal), encoding="utf-8")

    with pytest.raises(ValueError, match="must stay under snapshot root"):
        _build(proposal_path=path, snapshot_root=_FIXTURE)


def test_source_registry_training_gate_cannot_be_opened(tmp_path: Path) -> None:
    rows = []
    for line in _SOURCES.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["source_id"] == "nist.ai-agent-identity-authority-2026":
            row["training_authorization"] = True
        rows.append(json.dumps(row, sort_keys=True))
    source_registry = tmp_path / "sources.jsonl"
    source_registry.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="training authorization is not closed"):
        _build(source_registry_path=source_registry)


def test_answer_key_must_bind_the_same_case(tmp_path: Path) -> None:
    answer_key = json.loads(_ANSWER_KEY.read_text(encoding="utf-8"))
    answer_key["case_id"] = "different-web4-case"
    path = tmp_path / "answer-key.json"
    path.write_text(json.dumps(answer_key), encoding="utf-8")

    with pytest.raises(ValueError, match="answer key case_id differs"):
        _build(answer_key_path=path)


def test_intake_policy_cannot_enable_manual_split_override(tmp_path: Path) -> None:
    policy = json.loads(_INTAKE_POLICY.read_text(encoding="utf-8"))
    policy["manual_split_override_allowed"] = True
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(ValueError):
        _build(intake_policy_path=path)
