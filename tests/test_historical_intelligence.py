from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from koschei_sentinel.historical_intelligence import (
    ActorRelation,
    HistoricalIntelligenceBundle,
    RelationBasis,
    RepeatOperatorFamily,
    VerdictRevision,
)

NOW = datetime(2026, 8, 9, tzinfo=UTC)
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def actor_relation(**overrides):
    payload = {
        "relation_ref": "relation_001",
        "source_actor_ref": "actor_alpha",
        "target_actor_ref": "actor_beta",
        "relation_kind": "shared_funder",
        "basis": RelationBasis.DIRECT_ONCHAIN,
        "confidence": "VERIFIED",
        "evidence_ids": ["evidence_001"],
        "first_seen_at": NOW,
        "last_seen_at": NOW + timedelta(minutes=1),
    }
    payload.update(overrides)
    return ActorRelation.model_validate(payload)


def revision(number: int, *, signature: str, supersedes: str | None, current: bool):
    return VerdictRevision(
        case_ref="case_001",
        revision=number,
        verdict_signature=signature,
        grade="D" if number == 1 else "F",
        current=current,
        supersedes_signature=supersedes,
        evidence_bundle_digest=DIGEST_A if number == 1 else DIGEST_B,
        occurred_at=NOW + timedelta(minutes=number),
    )


def test_direct_actor_relation_is_accepted() -> None:
    relation = actor_relation()
    assert relation.basis is RelationBasis.DIRECT_ONCHAIN
    assert relation.confidence.value == "VERIFIED"


def test_watch_only_inference_cannot_claim_verified_confidence() -> None:
    with pytest.raises(ValidationError, match="watch-only inference"):
        actor_relation(basis=RelationBasis.WATCH_ONLY_INFERENCE)


def test_actor_relation_cannot_point_to_itself() -> None:
    with pytest.raises(ValidationError, match="endpoints must be different"):
        actor_relation(target_actor_ref="actor_alpha")


def test_repeat_operator_family_requires_distinct_cases() -> None:
    with pytest.raises(ValidationError, match="case_refs must be unique"):
        RepeatOperatorFamily(
            family_ref="family_001",
            operator_refs=["actor_alpha"],
            case_refs=["case_001", "case_001"],
            evidence_ids=["evidence_001"],
            first_seen_at=NOW,
            last_seen_at=NOW,
        )


def test_verdict_chain_must_be_contiguous_and_cryptographically_linked() -> None:
    first = revision(1, signature="signature_0001", supersedes=None, current=False)
    second = revision(
        2,
        signature="signature_0002",
        supersedes="signature_0001",
        current=True,
    )
    bundle = HistoricalIntelligenceBundle(
        snapshot_digest="c" * 64,
        actor_relations=[actor_relation()],
        verdict_revisions=[first, second],
    )
    assert bundle.verdict_revisions[-1].current is True

    broken = second.model_copy(update={"supersedes_signature": "signature_wrong"})
    with pytest.raises(ValidationError, match="supersession chain is broken"):
        HistoricalIntelligenceBundle(
            snapshot_digest="c" * 64,
            verdict_revisions=[first, broken],
        )


def test_only_latest_revision_can_be_current() -> None:
    first = revision(1, signature="signature_0001", supersedes=None, current=True)
    second = revision(
        2,
        signature="signature_0002",
        supersedes="signature_0001",
        current=False,
    )
    with pytest.raises(ValidationError, match="current verdict is not latest"):
        HistoricalIntelligenceBundle(
            snapshot_digest="d" * 64,
            verdict_revisions=[first, second],
        )


def test_bundle_rejects_duplicate_relation_refs() -> None:
    first = actor_relation()
    second = actor_relation(
        source_actor_ref="actor_gamma",
        target_actor_ref="actor_delta",
    )
    with pytest.raises(ValidationError, match="actor relation refs must be unique"):
        HistoricalIntelligenceBundle(
            snapshot_digest="e" * 64,
            actor_relations=[first, second],
        )
