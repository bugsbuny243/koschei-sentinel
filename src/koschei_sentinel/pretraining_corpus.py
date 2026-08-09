from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.anonymize import detect_sensitive_text
from koschei_sentinel.models import StrictModel

_DIGEST = r"^[a-f0-9]{64}$"
_PSEUDONYM = r"^[a-z][a-z0-9_]*_[a-f0-9]{24}$"


class PretrainingSourceClass(StrEnum):
    SOLANA_PROTOCOL = "SOLANA_PROTOCOL"
    KOSCHEI_SECURITY_CASE = "KOSCHEI_SECURITY_CASE"
    PUBLIC_SECURITY_REPORT = "PUBLIC_SECURITY_REPORT"
    KOSCHEI_SYNTHETIC = "KOSCHEI_SYNTHETIC"


class RightsBasis(StrEnum):
    KOSCHEI_OWNED = "KOSCHEI_OWNED"
    PUBLIC_DOMAIN = "PUBLIC_DOMAIN"
    CC0_1_0 = "CC0_1_0"
    CC_BY_4_0 = "CC_BY_4_0"
    APACHE_2_0 = "APACHE_2_0"
    MIT = "MIT"
    LICENSED_FOR_TRAINING = "LICENSED_FOR_TRAINING"


class PretrainingDocument(StrictModel):
    schema_version: Literal["sentinel.pretraining-document.v1"] = (
        "sentinel.pretraining-document.v1"
    )
    document_ref: str = Field(pattern=_PSEUDONYM)
    source_class: PretrainingSourceClass
    rights_basis: RightsBasis
    source_snapshot_digest: str = Field(pattern=_DIGEST)
    content_digest: str = Field(pattern=_DIGEST)
    family_refs: list[str] = Field(default_factory=list, max_length=128)
    text: str = Field(min_length=1, max_length=100_000)

    @model_validator(mode="after")
    def document_is_safe_and_bound(self) -> PretrainingDocument:
        actual = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.content_digest != actual:
            raise ValueError("content_digest does not match document text")
        if len(self.family_refs) != len(set(self.family_refs)):
            raise ValueError("family_refs must be unique")
        for family_ref in self.family_refs:
            if not _is_pseudonym(family_ref):
                raise ValueError("family_refs must be pseudonymized identifiers")
        findings = detect_sensitive_text(self.text)
        if findings:
            raise ValueError(
                "pretraining text contains sensitive or raw identifier material: "
                + ", ".join(findings)
            )
        return self


class PretrainingHoldoutSet(StrictModel):
    schema_version: Literal["sentinel.pretraining-holdout.v1"] = (
        "sentinel.pretraining-holdout.v1"
    )
    benchmark_suite_digest: str = Field(pattern=_DIGEST)
    content_digests: list[str] = Field(default_factory=list, max_length=1_000_000)
    family_refs: list[str] = Field(default_factory=list, max_length=1_000_000)

    @model_validator(mode="after")
    def holdout_is_canonical(self) -> PretrainingHoldoutSet:
        if len(self.content_digests) != len(set(self.content_digests)):
            raise ValueError("holdout content_digests must be unique")
        if len(self.family_refs) != len(set(self.family_refs)):
            raise ValueError("holdout family_refs must be unique")
        if any(not _is_digest(item) for item in self.content_digests):
            raise ValueError("holdout content_digests must be SHA-256 values")
        if any(not _is_pseudonym(item) for item in self.family_refs):
            raise ValueError("holdout family_refs must be pseudonymized identifiers")
        return self


class PretrainingCorpusPolicy(StrictModel):
    schema_version: Literal["sentinel.pretraining-corpus-policy.v1"] = (
        "sentinel.pretraining-corpus-policy.v1"
    )
    policy_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    min_documents: int = Field(default=1000, ge=1, le=100_000_000)
    min_source_classes: int = Field(default=3, ge=1, le=len(PretrainingSourceClass))
    min_unique_families: int = Field(default=100, ge=0, le=10_000_000)
    max_single_family_bps: int = Field(default=500, ge=1, le=10_000)
    require_zero_holdout_content_overlap: Literal[True] = True
    require_zero_holdout_family_overlap: Literal[True] = True
    require_known_rights_basis: Literal[True] = True
    raw_sensitive_text_allowed: Literal[False] = False


class PretrainingCorpusAudit(StrictModel):
    schema_version: Literal["sentinel.pretraining-corpus-audit.v1"] = (
        "sentinel.pretraining-corpus-audit.v1"
    )
    ready: bool
    policy_id: str
    corpus_digest: str = Field(pattern=_DIGEST)
    policy_digest: str = Field(pattern=_DIGEST)
    holdout_digest: str = Field(pattern=_DIGEST)
    benchmark_suite_digest: str = Field(pattern=_DIGEST)
    documents: int
    unique_families: int
    source_class_counts: dict[str, int]
    rights_basis_counts: dict[str, int]
    max_single_family_bps: int
    duplicate_content_digests: list[str]
    holdout_content_hits: list[str]
    holdout_family_hits: list[str]
    violations: list[str]


def audit_pretraining_corpus(
    documents: list[PretrainingDocument],
    holdout: PretrainingHoldoutSet,
    policy: PretrainingCorpusPolicy,
) -> PretrainingCorpusAudit:
    ordered = sorted(documents, key=lambda item: item.document_ref)
    violations: list[str] = []

    document_refs = [item.document_ref for item in ordered]
    if len(document_refs) != len(set(document_refs)):
        violations.append("duplicate document_ref detected")

    content_counts = Counter(item.content_digest for item in ordered)
    duplicate_content = sorted(
        digest for digest, count in content_counts.items() if count > 1
    )
    if duplicate_content:
        violations.append("duplicate document content detected")

    source_counts = Counter(item.source_class.value for item in ordered)
    rights_counts = Counter(item.rights_basis.value for item in ordered)
    family_counts = Counter(
        family_ref for item in ordered for family_ref in item.family_refs
    )

    holdout_content = set(holdout.content_digests)
    content_hits = sorted(set(content_counts).intersection(holdout_content))
    if content_hits:
        violations.append("pretraining corpus overlaps held-out benchmark content")

    holdout_families = set(holdout.family_refs)
    family_hits = sorted(set(family_counts).intersection(holdout_families))
    if family_hits:
        violations.append("pretraining corpus overlaps held-out actor/incident families")

    if len(ordered) < policy.min_documents:
        violations.append(
            f"documents {len(ordered)} below minimum {policy.min_documents}"
        )
    if len(source_counts) < policy.min_source_classes:
        violations.append(
            f"source classes {len(source_counts)} below minimum {policy.min_source_classes}"
        )
    if len(family_counts) < policy.min_unique_families:
        violations.append(
            f"unique families {len(family_counts)} below minimum {policy.min_unique_families}"
        )

    max_family_bps = _max_family_bps(family_counts)
    if max_family_bps > policy.max_single_family_bps:
        violations.append(
            f"single-family concentration {max_family_bps} bps exceeds "
            f"maximum {policy.max_single_family_bps} bps"
        )

    return PretrainingCorpusAudit(
        ready=not violations,
        policy_id=policy.policy_id,
        corpus_digest=_digest_documents(ordered),
        policy_digest=_digest_model(policy),
        holdout_digest=_digest_model(holdout),
        benchmark_suite_digest=holdout.benchmark_suite_digest,
        documents=len(ordered),
        unique_families=len(family_counts),
        source_class_counts=dict(sorted(source_counts.items())),
        rights_basis_counts=dict(sorted(rights_counts.items())),
        max_single_family_bps=max_family_bps,
        duplicate_content_digests=duplicate_content,
        holdout_content_hits=content_hits,
        holdout_family_hits=family_hits,
        violations=violations,
    )


def load_pretraining_documents(path: str | Path) -> list[PretrainingDocument]:
    source = Path(path)
    rows: list[PretrainingDocument] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank pretraining corpus row at line {line_number}")
        try:
            rows.append(PretrainingDocument.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(
                f"invalid pretraining corpus row at line {line_number}"
            ) from exc
    if not rows:
        raise ValueError("pretraining corpus contains no documents")
    return rows


def load_pretraining_holdout(path: str | Path) -> PretrainingHoldoutSet:
    try:
        return PretrainingHoldoutSet.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid pretraining holdout set") from exc


def load_pretraining_policy(path: str | Path) -> PretrainingCorpusPolicy:
    try:
        return PretrainingCorpusPolicy.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid pretraining corpus policy") from exc


def write_pretraining_audit(audit: PretrainingCorpusAudit, path: str | Path) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"pretraining audit already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(audit.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def content_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _max_family_bps(counts: Counter[str]) -> int:
    total = sum(counts.values())
    if total == 0:
        return 0
    return max((count * 10_000 + total - 1) // total for count in counts.values())


def _digest_documents(documents: list[PretrainingDocument]) -> str:
    payload = "".join(
        json.dumps(
            item.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
        for item in documents
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _digest_model(model: StrictModel) -> str:
    payload = json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_digest(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_pseudonym(value: str) -> bool:
    prefix, separator, digest = value.rpartition("_")
    return bool(
        separator
        and prefix
        and prefix[0].isalpha()
        and all(character.islower() or character.isdigit() or character == "_" for character in prefix)
        and len(digest) == 24
        and all(character in "0123456789abcdef" for character in digest)
    )
