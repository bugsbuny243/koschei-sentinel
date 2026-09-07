from __future__ import annotations

import json
from pathlib import Path

import pytest

from koschei_sentinel.web4_benchmark_intake import build_web4_benchmark_intake
from koschei_sentinel.web4_research_snapshot import (
    build_web4_research_snapshot_receipt,
    write_web4_research_snapshot_receipt,
)

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _ROOT / "fixtures/web4/benchmark-intake"
_PROPOSAL = _FIXTURE / "proposal.json"
_ANSWER_KEY = _FIXTURE / "answer-key.json"
_SOURCES = _ROOT / "configs/corpus/web4-v1.sources.proposed.jsonl"
_BENCHMARK = _ROOT / "evals/web4-security-benchmark.v1.json"
_INTAKE_POLICY = _ROOT / "evals/web4-benchmark-intake-policy.v1.json"
_CAPTURED_AT = "2026-09-07T10:57:00+03:00"


def _build(
    *,
    proposal_path: Path = _PROPOSAL,
    source_registry_path: Path = _SOURCES,
    intake_policy_path: Path = _INTAKE_POLICY,
    snapshot_root: Path = _FIXTURE,
):
    return build_web4_benchmark_intake(
        proposal_path=proposal_path,
        answer_key_path=_ANSWER_KEY,
        source_registry_path=source_registry_path,
        benchmark_policy_path=_BENCHMARK,
        intake_policy_path=intake_policy_path,
        snapshot_root=snapshot_root,
    )


def _write_proposal(tmp_path: Path, mutate) -> Path:
    proposal = json.loads(_PROPOSAL.read_text(encoding="utf-8"))
    mutate(proposal)
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(proposal), encoding="utf-8")
    return path


def test_intake_packet_binds_non_authorizing_snapshot_receipt() -> None:
    packet = _build()

    assert packet.source_snapshot_receipt_sha256s == {
        "nist.ai-agent-identity-authority-2026": (
            "2610f4b628148dadb18b95c5879edc5961e1d9cb940c4d4bd467661b8ca3656f"
        )
    }
    assert packet.source_match_verified == {
        "nist.ai-agent-identity-authority-2026": False
    }
    assert packet.source_provenance_review_status == {
        "nist.ai-agent-identity-authority-2026": "REVIEW_REQUIRED"
    }
    assert packet.review_status == "UNREVIEWED"
    assert packet.human_reviewed is False
    assert packet.training_authorization is False
    assert packet.evaluation_authorization is False
    assert packet.promotion_eligible is False


def test_missing_snapshot_receipt_path_is_rejected(tmp_path: Path) -> None:
    path = _write_proposal(
        tmp_path,
        lambda proposal: proposal["sources"][0].pop("snapshot_receipt_path"),
    )

    with pytest.raises(ValueError):
        _build(proposal_path=path)


def test_valid_receipt_for_different_source_is_rejected(tmp_path: Path) -> None:
    snapshot = tmp_path / "source-snapshot.txt"
    snapshot.write_bytes((_FIXTURE / "source-snapshot.txt").read_bytes())
    wrong_receipt = build_web4_research_snapshot_receipt(
        source_id="ietf.draft.nemethi.aid-agent-identity-discovery-00",
        snapshot_path=snapshot,
        captured_at=_CAPTURED_AT,
        source_registry_path=_SOURCES,
    )
    write_web4_research_snapshot_receipt(wrong_receipt, tmp_path / "wrong-receipt.json")
    proposal = json.loads(_PROPOSAL.read_text(encoding="utf-8"))
    proposal["sources"][0]["snapshot_receipt_path"] = "wrong-receipt.json"
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")

    with pytest.raises(ValueError, match="receipt source differs from proposal"):
        _build(proposal_path=proposal_path, snapshot_root=tmp_path)


def test_snapshot_byte_tamper_is_rejected(tmp_path: Path) -> None:
    snapshot = tmp_path / "source-snapshot.txt"
    snapshot.write_bytes((_FIXTURE / "source-snapshot.txt").read_bytes())
    receipt = tmp_path / "source-snapshot.receipt.json"
    receipt.write_bytes((_FIXTURE / "source-snapshot.receipt.json").read_bytes())
    snapshot.write_text("tampered after receipt capture\n", encoding="utf-8")
    proposal = json.loads(_PROPOSAL.read_text(encoding="utf-8"))
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")

    with pytest.raises(ValueError, match="differs from current source/snapshot binding"):
        _build(proposal_path=proposal_path, snapshot_root=tmp_path)


def test_source_registry_drift_invalidates_snapshot_receipt(tmp_path: Path) -> None:
    registry = tmp_path / "sources.jsonl"
    rows = _SOURCES.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["notes"] = str(first.get("notes", "")) + " drift"
    rows[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
    registry.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="differs from current source/snapshot binding"):
        _build(source_registry_path=registry)


def test_snapshot_receipt_self_hash_tamper_is_rejected(tmp_path: Path) -> None:
    snapshot = tmp_path / "source-snapshot.txt"
    snapshot.write_bytes((_FIXTURE / "source-snapshot.txt").read_bytes())
    payload = json.loads(
        (_FIXTURE / "source-snapshot.receipt.json").read_text(encoding="utf-8")
    )
    payload["snapshot_size_bytes"] += 1
    receipt_path = tmp_path / "source-snapshot.receipt.json"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    proposal = json.loads(_PROPOSAL.read_text(encoding="utf-8"))
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")

    with pytest.raises(ValueError, match="self-hash does not verify"):
        _build(proposal_path=proposal_path, snapshot_root=tmp_path)


def test_intake_policy_cannot_drop_snapshot_receipt_requirement(tmp_path: Path) -> None:
    policy = json.loads(_INTAKE_POLICY.read_text(encoding="utf-8"))
    policy["require_source_snapshot_receipts"] = False
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(ValueError):
        _build(intake_policy_path=path)
