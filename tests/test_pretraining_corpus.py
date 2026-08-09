from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from koschei_sentinel.pretraining_corpus import (
    PretrainingCorpusPolicy,
    PretrainingDocument,
    PretrainingHoldoutSet,
    PretrainingSourceClass,
    RightsBasis,
    audit_pretraining_corpus,
    content_digest,
)


def pseudonym(prefix: str, character: str) -> str:
    return f"{prefix}_{character * 24}"


def document(
    index: int,
    *,
    text: str | None = None,
    source_class: PretrainingSourceClass = PretrainingSourceClass.KOSCHEI_SECURITY_CASE,
    rights_basis: RightsBasis = RightsBasis.KOSCHEI_OWNED,
    families: list[str] | None = None,
) -> PretrainingDocument:
    body = text or f"Sanitized Solana security training document {index}."
    return PretrainingDocument(
        document_ref=pseudonym("doc", format(index, "x")[-1]),
        source_class=source_class,
        rights_basis=rights_basis,
        source_snapshot_digest=format(index + 1, "064x"),
        content_digest=content_digest(body),
        family_refs=families or [pseudonym("family", format(index + 3, "x")[-1])],
        text=body,
    )


def policy(**updates) -> PretrainingCorpusPolicy:
    payload = {
        "policy_id": "stage2-test",
        "min_documents": 3,
        "min_source_classes": 2,
        "min_unique_families": 2,
        "max_single_family_bps": 5000,
    }
    payload.update(updates)
    return PretrainingCorpusPolicy.model_validate(payload)


def holdout(**updates) -> PretrainingHoldoutSet:
    payload = {
        "benchmark_suite_digest": "a" * 64,
        "content_digests": [],
        "family_refs": [],
    }
    payload.update(updates)
    return PretrainingHoldoutSet.model_validate(payload)


def safe_documents() -> list[PretrainingDocument]:
    return [
        document(
            1,
            source_class=PretrainingSourceClass.KOSCHEI_SECURITY_CASE,
            families=[pseudonym("family", "a")],
        ),
        document(
            2,
            source_class=PretrainingSourceClass.SOLANA_PROTOCOL,
            rights_basis=RightsBasis.APACHE_2_0,
            families=[pseudonym("family", "b")],
        ),
        document(
            3,
            source_class=PretrainingSourceClass.PUBLIC_SECURITY_REPORT,
            rights_basis=RightsBasis.CC_BY_4_0,
            families=[pseudonym("family", "c")],
        ),
    ]


def test_safe_diverse_corpus_passes_audit() -> None:
    audit = audit_pretraining_corpus(safe_documents(), holdout(), policy())
    assert audit.ready is True
    assert audit.documents == 3
    assert audit.unique_families == 3
    assert audit.holdout_content_hits == []
    assert audit.holdout_family_hits == []
    assert audit.violations == []


def test_raw_sensitive_material_is_rejected_before_audit() -> None:
    text = "Send the training export to analyst@example.com."
    with pytest.raises(ValidationError, match="sensitive or raw identifier"):
        document(1, text=text)


def test_document_text_is_cryptographically_bound() -> None:
    with pytest.raises(ValidationError, match="content_digest"):
        PretrainingDocument(
            document_ref=pseudonym("doc", "a"),
            source_class=PretrainingSourceClass.KOSCHEI_SECURITY_CASE,
            rights_basis=RightsBasis.KOSCHEI_OWNED,
            source_snapshot_digest="b" * 64,
            content_digest="c" * 64,
            family_refs=[pseudonym("family", "d")],
            text="Sanitized content whose hash does not match.",
        )


def test_heldout_content_overlap_blocks_stage2_readiness() -> None:
    documents = safe_documents()
    audit = audit_pretraining_corpus(
        documents,
        holdout(content_digests=[documents[0].content_digest]),
        policy(),
    )
    assert audit.ready is False
    assert audit.holdout_content_hits == [documents[0].content_digest]
    assert "held-out benchmark content" in " ".join(audit.violations)


def test_heldout_incident_family_overlap_blocks_stage2_readiness() -> None:
    documents = safe_documents()
    family = documents[1].family_refs[0]
    audit = audit_pretraining_corpus(
        documents,
        holdout(family_refs=[family]),
        policy(),
    )
    assert audit.ready is False
    assert audit.holdout_family_hits == [family]
    assert "actor/incident families" in " ".join(audit.violations)


def test_duplicate_content_cannot_inflate_corpus_size() -> None:
    documents = safe_documents()
    duplicate = documents[0].model_copy(
        update={"document_ref": pseudonym("doc", "f")}
    )
    audit = audit_pretraining_corpus(
        [*documents, duplicate],
        holdout(),
        policy(min_documents=4),
    )
    assert audit.ready is False
    assert audit.duplicate_content_digests == [documents[0].content_digest]
    assert "duplicate document content" in " ".join(audit.violations)


def test_single_family_dominance_is_rejected() -> None:
    family = pseudonym("family", "e")
    documents = [
        item.model_copy(update={"family_refs": [family]})
        for item in safe_documents()
    ]
    audit = audit_pretraining_corpus(
        documents,
        holdout(),
        policy(min_unique_families=1, max_single_family_bps=4000),
    )
    assert audit.ready is False
    assert audit.max_single_family_bps == 10000
    assert "single-family concentration" in " ".join(audit.violations)


def test_audit_is_deterministic_across_input_order() -> None:
    documents = safe_documents()
    first = audit_pretraining_corpus(documents, holdout(), policy())
    second = audit_pretraining_corpus(list(reversed(documents)), holdout(), policy())
    assert first == second


def test_stage2_repository_policy_remains_fail_closed() -> None:
    root = Path(__file__).parents[1]
    payload = json.loads(
        (root / "configs" / "pretraining" / "stage2-corpus.v1.json").read_text()
    )
    repository_policy = PretrainingCorpusPolicy.model_validate(payload)
    assert repository_policy.min_documents >= 10_000
    assert repository_policy.min_source_classes >= 3
    assert repository_policy.min_unique_families >= 500
    assert repository_policy.max_single_family_bps <= 200
    assert repository_policy.require_zero_holdout_content_overlap is True
    assert repository_policy.require_zero_holdout_family_overlap is True
    assert repository_policy.raw_sensitive_text_allowed is False
