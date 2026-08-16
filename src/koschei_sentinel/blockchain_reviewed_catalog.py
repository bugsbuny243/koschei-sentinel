from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.blockchain_security_corpus import (
    BlockchainSecurityDocument,
    BlockchainSourceClass,
    ChainFamily,
    ThreatDomain,
)
from koschei_sentinel.blockchain_source_catalog import (
    BlockchainSourceCatalogPolicy,
    SourceTrustTier,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.pretraining_corpus import RightsBasis

_DIGEST = r"^[a-f0-9]{64}$"
_COMMIT = r"^[a-f0-9]{40}$"
_HIGH_TRUST_TIERS = frozenset(
    {
        SourceTrustTier.PRIMARY_PROTOCOL,
        SourceTrustTier.OFFICIAL_INCIDENT,
        SourceTrustTier.INDEPENDENT_AUDIT,
        SourceTrustTier.FORMAL_SPECIFICATION,
    }
)
_RIGHTS_BY_SPDX = {
    "Apache-2.0": RightsBasis.APACHE_2_0,
    "MIT": RightsBasis.MIT,
    "CC0-1.0": RightsBasis.CC0_1_0,
    "CC-BY-4.0": RightsBasis.CC_BY_4_0,
}


def _digest_payload(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_relative_path(value: str, field: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or value.startswith("~")
        or "\\" in value
    ):
        raise ValueError(f"{field} must stay within the catalog root")


class ReviewedCatalogSource(StrictModel):
    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    repository: str = Field(min_length=3, max_length=256)
    commit: str = Field(pattern=_COMMIT)
    lineage_lane: Literal["ORIGINAL", "REPAIR", "CLASS_GAP"]
    source_level_class: BlockchainSourceClass
    trust_tier: SourceTrustTier
    snapshot_digest: str = Field(pattern=_DIGEST)
    license_spdx: Literal["Apache-2.0", "MIT", "CC0-1.0", "CC-BY-4.0"]
    license_evidence_path: str = Field(min_length=1, max_length=1024)
    license_evidence_sha256: str = Field(pattern=_DIGEST)
    retained_documents: int = Field(ge=1)
    retained_document_set_digest: str = Field(pattern=_DIGEST)
    reviewed_document_class_counts: dict[str, int]
    reviewed_chain_families: list[ChainFamily] = Field(min_length=1)
    reviewed_threat_domains: list[ThreatDomain] = Field(min_length=1)

    @model_validator(mode="after")
    def reviewed_source_is_canonical(self) -> ReviewedCatalogSource:
        if "/" not in self.repository:
            raise ValueError("repository must use owner/name form")
        _validate_relative_path(self.license_evidence_path, "license_evidence_path")
        if len(self.reviewed_chain_families) != len(set(self.reviewed_chain_families)):
            raise ValueError("reviewed_chain_families must be unique")
        if len(self.reviewed_threat_domains) != len(set(self.reviewed_threat_domains)):
            raise ValueError("reviewed_threat_domains must be unique")
        if sum(self.reviewed_document_class_counts.values()) != self.retained_documents:
            raise ValueError("reviewed document class counts do not match retained_documents")
        for key, value in self.reviewed_document_class_counts.items():
            BlockchainSourceClass(key)
            if value <= 0:
                raise ValueError("reviewed document class counts must be positive")
        return self


class ReviewedSourceCatalog(StrictModel):
    schema_version: Literal["sentinel.reviewed-source-catalog.v1"] = (
        "sentinel.reviewed-source-catalog.v1"
    )
    catalog_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    parent_document_review_manifest_digest: str = Field(pattern=_DIGEST)
    parent_document_review_sha256: str = Field(pattern=_DIGEST)
    approved_document_corpus_file: str = Field(min_length=1, max_length=1024)
    approved_document_corpus_sha256: str = Field(pattern=_DIGEST)
    sources: list[ReviewedCatalogSource] = Field(min_length=1, max_length=100_000)
    catalog_digest: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def reviewed_catalog_is_canonical(self) -> ReviewedSourceCatalog:
        _validate_relative_path(
            self.approved_document_corpus_file,
            "approved_document_corpus_file",
        )
        source_ids = [item.source_id for item in self.sources]
        if source_ids != sorted(source_ids):
            raise ValueError("reviewed catalog sources must be sorted by source_id")
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("reviewed catalog source_id values must be unique")
        payload = self.model_dump(mode="json", exclude={"catalog_digest"})
        if _digest_payload(payload) != self.catalog_digest:
            raise ValueError("reviewed catalog digest mismatch")
        return self


class ApprovedBlockchainDocument(StrictModel):
    schema_version: Literal["sentinel.approved-blockchain-document.v1"] = (
        "sentinel.approved-blockchain-document.v1"
    )
    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    repository: str = Field(min_length=3, max_length=256)
    commit: str = Field(pattern=_COMMIT)
    lineage_lane: Literal["ORIGINAL", "REPAIR", "CLASS_GAP"]
    path: str = Field(min_length=1, max_length=2048)
    source_snapshot_digest: str = Field(pattern=_DIGEST)
    content_sha256: str = Field(pattern=_DIGEST)
    source_class: BlockchainSourceClass
    chain_family: ChainFamily
    threat_domains: list[ThreatDomain] = Field(min_length=1, max_length=32)
    text: str = Field(min_length=1, max_length=100_000)
    license_spdx: Literal["Apache-2.0", "MIT", "CC0-1.0", "CC-BY-4.0"]
    review_decision: Literal["RETAIN"]
    review_reason: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def approved_document_is_bound(self) -> ApprovedBlockchainDocument:
        if "/" not in self.repository:
            raise ValueError("repository must use owner/name form")
        _validate_relative_path(self.path, "approved document path")
        if len(self.threat_domains) != len(set(self.threat_domains)):
            raise ValueError("approved document threat_domains must be unique")
        actual = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if actual != self.content_sha256:
            raise ValueError("approved document content_sha256 does not match text")
        return self


class ReviewedCatalogSourceBinding(StrictModel):
    source_id: str
    trust_tier: SourceTrustTier
    source_class: BlockchainSourceClass
    snapshot_digest: str = Field(pattern=_DIGEST)
    documents: int = Field(ge=1)
    document_class_counts: dict[str, int]
    chain_families: list[ChainFamily]
    threat_domains: list[ThreatDomain]
    retained_document_set_digest: str = Field(pattern=_DIGEST)


class ReviewedSourceCatalogManifest(StrictModel):
    schema_version: Literal["sentinel.reviewed-source-catalog-manifest.v1"] = (
        "sentinel.reviewed-source-catalog-manifest.v1"
    )
    ready: bool
    catalog_id: str
    catalog_digest: str = Field(pattern=_DIGEST)
    policy_id: str
    policy_digest: str = Field(pattern=_DIGEST)
    approved_corpus_digest: str = Field(pattern=_DIGEST)
    sources: int
    documents: int
    high_trust_share_bps: int
    synthetic_source_share_bps: int
    max_single_source_document_share_bps: int
    source_class_counts: dict[str, int]
    document_class_counts: dict[str, int]
    trust_tier_counts: dict[str, int]
    chain_source_counts: dict[str, int]
    threat_source_counts: dict[str, int]
    document_chain_counts: dict[str, int]
    document_threat_counts: dict[str, int]
    duplicate_snapshot_digests: list[str]
    duplicate_content_digests: list[str]
    missing_required_chain_families: list[str]
    missing_required_threat_domains: list[str]
    combined_corpus_digest: str = Field(pattern=_DIGEST)
    bindings: list[ReviewedCatalogSourceBinding]
    violations: list[str]


class ReviewedSourceCatalogResult(StrictModel):
    manifest: ReviewedSourceCatalogManifest
    documents: list[BlockchainSecurityDocument]


def load_reviewed_source_catalog(path: str | Path) -> ReviewedSourceCatalog:
    try:
        return ReviewedSourceCatalog.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError("invalid reviewed blockchain source catalog") from exc


def load_approved_blockchain_documents(path: str | Path) -> list[ApprovedBlockchainDocument]:
    rows: list[ApprovedBlockchainDocument] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank approved blockchain document row at line {line_number}")
            try:
                rows.append(ApprovedBlockchainDocument.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(
                    f"invalid approved blockchain document row at line {line_number}"
                ) from exc
    if not rows:
        raise ValueError("approved blockchain document corpus contains no documents")
    keys = [(item.source_id, item.path) for item in rows]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ValueError("approved blockchain document rows are not canonical and unique")
    return rows


def audit_reviewed_source_catalog(
    catalog: ReviewedSourceCatalog,
    policy: BlockchainSourceCatalogPolicy,
    *,
    root: str | Path = ".",
) -> ReviewedSourceCatalogResult:
    root_path = Path(root).resolve()
    corpus_path = _resolve_file(
        root_path,
        catalog.approved_document_corpus_file,
        "approved document corpus",
    )
    corpus_digest = _hash_file(corpus_path)
    if corpus_digest != catalog.approved_document_corpus_sha256:
        raise ValueError("approved document corpus digest does not match reviewed catalog")

    approved = load_approved_blockchain_documents(corpus_path)
    sources = {item.source_id: item for item in catalog.sources}
    grouped: dict[str, list[ApprovedBlockchainDocument]] = defaultdict(list)
    content_sources: dict[str, str] = {}
    duplicate_content: set[str] = set()

    for item in approved:
        source = sources.get(item.source_id)
        if source is None:
            raise ValueError(f"approved document references unknown source: {item.source_id}")
        if (
            item.repository != source.repository
            or item.commit != source.commit
            or item.lineage_lane != source.lineage_lane
            or item.source_snapshot_digest != source.snapshot_digest
            or item.license_spdx != source.license_spdx
        ):
            raise ValueError(f"approved document lineage mismatch: {item.source_id}")
        previous = content_sources.setdefault(item.content_sha256, item.source_id)
        if previous != item.source_id:
            duplicate_content.add(item.content_sha256)
        grouped[item.source_id].append(item)

    missing_sources = sorted(set(sources).difference(grouped))
    if missing_sources:
        raise ValueError(
            "reviewed catalog sources have no approved documents: " + ", ".join(missing_sources)
        )

    bindings: list[ReviewedCatalogSourceBinding] = []
    source_classes: Counter[str] = Counter()
    trust_tiers: Counter[str] = Counter()
    chain_sources: Counter[str] = Counter()
    threat_sources: Counter[str] = Counter()
    document_classes: Counter[str] = Counter()
    document_chains: Counter[str] = Counter()
    document_threats: Counter[str] = Counter()
    snapshot_counts: Counter[str] = Counter()
    documents: list[BlockchainSecurityDocument] = []

    for source in catalog.sources:
        rows = grouped[source.source_id]
        if len(rows) != source.retained_documents:
            raise ValueError(f"retained document count mismatch: {source.source_id}")
        class_counts = Counter(item.source_class.value for item in rows)
        if class_counts != Counter(source.reviewed_document_class_counts):
            raise ValueError(f"reviewed document class counts mismatch: {source.source_id}")
        chains = sorted({item.chain_family for item in rows}, key=lambda item: item.value)
        if chains != sorted(source.reviewed_chain_families, key=lambda item: item.value):
            raise ValueError(f"reviewed chain families mismatch: {source.source_id}")
        threats = sorted(
            {domain for item in rows for domain in item.threat_domains},
            key=lambda item: item.value,
        )
        if threats != sorted(source.reviewed_threat_domains, key=lambda item: item.value):
            raise ValueError(f"reviewed threat domains mismatch: {source.source_id}")
        if _reviewed_document_set_digest(rows) != source.retained_document_set_digest:
            raise ValueError(f"retained document set digest mismatch: {source.source_id}")

        source_classes[source.source_level_class.value] += 1
        trust_tiers[source.trust_tier.value] += 1
        snapshot_counts[source.snapshot_digest] += 1
        for chain in source.reviewed_chain_families:
            chain_sources[chain.value] += 1
        for threat in source.reviewed_threat_domains:
            threat_sources[threat.value] += 1

        document_classes.update(class_counts)
        for item in rows:
            document_chains[item.chain_family.value] += 1
            document_threats.update(domain.value for domain in item.threat_domains)
            documents.append(_to_security_document(item))

        bindings.append(
            ReviewedCatalogSourceBinding(
                source_id=source.source_id,
                trust_tier=source.trust_tier,
                source_class=source.source_level_class,
                snapshot_digest=source.snapshot_digest,
                documents=len(rows),
                document_class_counts=dict(sorted(class_counts.items())),
                chain_families=source.reviewed_chain_families,
                threat_domains=source.reviewed_threat_domains,
                retained_document_set_digest=source.retained_document_set_digest,
            )
        )

    duplicate_snapshots = sorted(
        digest for digest, count in snapshot_counts.items() if count > 1
    )
    violations = _policy_violations(
        bindings=bindings,
        source_classes=source_classes,
        chain_sources=chain_sources,
        threat_sources=threat_sources,
        duplicate_snapshots=duplicate_snapshots,
        duplicate_content=duplicate_content,
        total_documents=len(documents),
        policy=policy,
    )

    documents.sort(key=lambda item: item.document_ref)
    refs = [item.document_ref for item in documents]
    if len(refs) != len(set(refs)):
        violations.append("duplicate document_ref detected across reviewed source catalog")

    high_trust_sources = sum(
        1 for binding in bindings if binding.trust_tier in _HIGH_TRUST_TIERS
    )
    synthetic_sources = sum(
        1
        for binding in bindings
        if binding.source_class is BlockchainSourceClass.KOSCHEI_SYNTHETIC
    )
    high_trust_share = _share_bps(high_trust_sources, len(bindings))
    synthetic_share = _share_bps(synthetic_sources, len(bindings))
    max_source_share = _share_bps(
        max((binding.documents for binding in bindings), default=0),
        len(documents),
    )

    manifest = ReviewedSourceCatalogManifest(
        ready=not violations,
        catalog_id=catalog.catalog_id,
        catalog_digest=catalog.catalog_digest,
        policy_id=policy.policy_id,
        policy_digest=_model_digest(policy),
        approved_corpus_digest=corpus_digest,
        sources=len(bindings),
        documents=len(documents),
        high_trust_share_bps=high_trust_share,
        synthetic_source_share_bps=synthetic_share,
        max_single_source_document_share_bps=max_source_share,
        source_class_counts=dict(sorted(source_classes.items())),
        document_class_counts=dict(sorted(document_classes.items())),
        trust_tier_counts=dict(sorted(trust_tiers.items())),
        chain_source_counts=dict(sorted(chain_sources.items())),
        threat_source_counts=dict(sorted(threat_sources.items())),
        document_chain_counts=dict(sorted(document_chains.items())),
        document_threat_counts=dict(sorted(document_threats.items())),
        duplicate_snapshot_digests=duplicate_snapshots,
        duplicate_content_digests=sorted(duplicate_content),
        missing_required_chain_families=sorted(
            {item.value for item in policy.required_chain_families}.difference(chain_sources)
        ),
        missing_required_threat_domains=sorted(
            {item.value for item in policy.required_threat_domains}.difference(threat_sources)
        ),
        combined_corpus_digest=_documents_digest(documents),
        bindings=bindings,
        violations=violations,
    )
    return ReviewedSourceCatalogResult(manifest=manifest, documents=documents)


def _policy_violations(
    *,
    bindings: list[ReviewedCatalogSourceBinding],
    source_classes: Counter[str],
    chain_sources: Counter[str],
    threat_sources: Counter[str],
    duplicate_snapshots: list[str],
    duplicate_content: set[str],
    total_documents: int,
    policy: BlockchainSourceCatalogPolicy,
) -> list[str]:
    violations: list[str] = []
    if duplicate_snapshots:
        violations.append("duplicate source snapshot digests detected")
    if duplicate_content:
        violations.append("duplicate document content detected across source snapshots")
    if len(bindings) < policy.min_sources:
        violations.append(f"sources {len(bindings)} below minimum {policy.min_sources}")
    if len(source_classes) < policy.min_source_classes:
        violations.append(
            f"source classes {len(source_classes)} below minimum {policy.min_source_classes}"
        )

    required_chains = {item.value for item in policy.required_chain_families}
    required_threats = {item.value for item in policy.required_threat_domains}
    if required_chains.difference(chain_sources):
        violations.append("required chain families are missing from source catalog")
    if required_threats.difference(threat_sources):
        violations.append("required threat domains are missing from source catalog")
    for chain in sorted(required_chains):
        if chain_sources[chain] < policy.min_sources_per_required_chain:
            violations.append(
                f"chain {chain} sources {chain_sources[chain]} below minimum "
                f"{policy.min_sources_per_required_chain}"
            )
    for threat in sorted(required_threats):
        if threat_sources[threat] < policy.min_sources_per_required_threat_domain:
            violations.append(
                f"threat domain {threat} sources {threat_sources[threat]} below minimum "
                f"{policy.min_sources_per_required_threat_domain}"
            )

    high_trust_sources = sum(
        1 for binding in bindings if binding.trust_tier in _HIGH_TRUST_TIERS
    )
    high_trust_share = _share_bps(high_trust_sources, len(bindings))
    if high_trust_share < policy.min_high_trust_share_bps:
        violations.append(
            f"high-trust source share {high_trust_share} bps below minimum "
            f"{policy.min_high_trust_share_bps} bps"
        )

    synthetic_sources = sum(
        1
        for binding in bindings
        if binding.source_class is BlockchainSourceClass.KOSCHEI_SYNTHETIC
    )
    synthetic_share = _share_bps(synthetic_sources, len(bindings))
    if synthetic_share > policy.max_synthetic_source_share_bps:
        violations.append(
            f"synthetic source share {synthetic_share} bps exceeds maximum "
            f"{policy.max_synthetic_source_share_bps} bps"
        )

    max_source_share = _share_bps(
        max((binding.documents for binding in bindings), default=0),
        total_documents,
    )
    if max_source_share > policy.max_single_source_document_share_bps:
        violations.append(
            f"single-source document share {max_source_share} bps exceeds maximum "
            f"{policy.max_single_source_document_share_bps} bps"
        )
    return violations


def _to_security_document(item: ApprovedBlockchainDocument) -> BlockchainSecurityDocument:
    ref_digest = hashlib.sha256(
        f"{item.source_id}|{item.path}|{item.content_sha256}".encode("utf-8")
    ).hexdigest()[:24]
    return BlockchainSecurityDocument(
        document_ref=f"doc_{ref_digest}",
        source_class=item.source_class,
        rights_basis=_RIGHTS_BY_SPDX[item.license_spdx],
        source_snapshot_digest=item.source_snapshot_digest,
        content_digest=item.content_sha256,
        family_refs=[],
        chain_families=[item.chain_family],
        threat_domains=item.threat_domains,
        text=item.text,
    )


def _reviewed_document_set_digest(rows: list[ApprovedBlockchainDocument]) -> str:
    return _digest_payload(
        [
            {
                "path": item.path,
                "content_sha256": item.content_sha256,
                "source_class": item.source_class.value,
                "chain_family": item.chain_family.value,
                "threat_domains": [domain.value for domain in item.threat_domains],
            }
            for item in sorted(rows, key=lambda item: item.path)
        ]
    )


def _documents_digest(documents: list[BlockchainSecurityDocument]) -> str:
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


def _model_digest(model: StrictModel) -> str:
    return _digest_payload(model.model_dump(mode="json"))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def _share_bps(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        return 0
    return (numerator * 10_000 + denominator - 1) // denominator


def _resolve_file(root: Path, relative: str, label: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"{label} path escapes the repository root")
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError(f"{label} is missing or not a regular file: {relative}")
    return candidate
