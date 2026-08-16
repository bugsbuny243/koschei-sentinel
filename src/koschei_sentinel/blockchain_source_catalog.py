from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.blockchain_security_corpus import (
    BlockchainSecurityDocument,
    BlockchainSourceClass,
    ChainFamily,
    ThreatDomain,
    load_blockchain_security_documents,
)
from koschei_sentinel.blockchain_security_ingest import BlockchainSourceSnapshotManifest
from koschei_sentinel.models import StrictModel

_DIGEST = r"^[a-f0-9]{64}$"


class SourceTrustTier(StrEnum):
    PRIMARY_PROTOCOL = "PRIMARY_PROTOCOL"
    OFFICIAL_INCIDENT = "OFFICIAL_INCIDENT"
    INDEPENDENT_AUDIT = "INDEPENDENT_AUDIT"
    FORMAL_SPECIFICATION = "FORMAL_SPECIFICATION"
    PUBLIC_RESEARCH = "PUBLIC_RESEARCH"
    KOSCHEI_CURATED = "KOSCHEI_CURATED"


_HIGH_TRUST_TIERS = frozenset(
    {
        SourceTrustTier.PRIMARY_PROTOCOL,
        SourceTrustTier.OFFICIAL_INCIDENT,
        SourceTrustTier.INDEPENDENT_AUDIT,
        SourceTrustTier.FORMAL_SPECIFICATION,
    }
)


class BlockchainCatalogSource(StrictModel):
    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    trust_tier: SourceTrustTier
    manifest_path: str = Field(min_length=1, max_length=1024)
    corpus_path: str = Field(min_length=1, max_length=1024)
    expected_snapshot_digest: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def paths_stay_local(self) -> BlockchainCatalogSource:
        _validate_relative_path(self.manifest_path, "manifest_path")
        _validate_relative_path(self.corpus_path, "corpus_path")
        if self.manifest_path == self.corpus_path:
            raise ValueError("manifest_path and corpus_path must be different")
        return self


class BlockchainSourceCatalog(StrictModel):
    schema_version: Literal["sentinel.blockchain-source-catalog.v1"] = (
        "sentinel.blockchain-source-catalog.v1"
    )
    catalog_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    sources: list[BlockchainCatalogSource] = Field(min_length=1, max_length=100_000)

    @model_validator(mode="after")
    def sources_are_unique(self) -> BlockchainSourceCatalog:
        source_ids = [item.source_id for item in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("catalog source_id values must be unique")
        manifests = [item.manifest_path for item in self.sources]
        corpora = [item.corpus_path for item in self.sources]
        if len(manifests) != len(set(manifests)):
            raise ValueError("catalog manifest_path values must be unique")
        if len(corpora) != len(set(corpora)):
            raise ValueError("catalog corpus_path values must be unique")
        return self


class BlockchainSourceCatalogPolicy(StrictModel):
    schema_version: Literal["sentinel.blockchain-source-catalog-policy.v1"] = (
        "sentinel.blockchain-source-catalog-policy.v1"
    )
    policy_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    min_sources: int = Field(default=20, ge=1, le=100_000)
    min_source_classes: int = Field(default=4, ge=1, le=len(BlockchainSourceClass))
    required_chain_families: list[ChainFamily] = Field(min_length=1)
    required_threat_domains: list[ThreatDomain] = Field(min_length=1)
    min_sources_per_required_chain: int = Field(default=2, ge=1, le=100_000)
    min_sources_per_required_threat_domain: int = Field(default=2, ge=1, le=100_000)
    min_high_trust_share_bps: int = Field(default=5000, ge=0, le=10_000)
    max_synthetic_source_share_bps: int = Field(default=2000, ge=0, le=10_000)
    max_single_source_document_share_bps: int = Field(default=1000, ge=1, le=10_000)
    require_unique_snapshot_digests: Literal[True] = True
    require_zero_cross_source_content_duplicates: Literal[True] = True

    @model_validator(mode="after")
    def coverage_lists_are_unique(self) -> BlockchainSourceCatalogPolicy:
        if len(self.required_chain_families) != len(set(self.required_chain_families)):
            raise ValueError("required_chain_families must be unique")
        if len(self.required_threat_domains) != len(set(self.required_threat_domains)):
            raise ValueError("required_threat_domains must be unique")
        return self


class CatalogSourceBinding(StrictModel):
    source_id: str
    trust_tier: SourceTrustTier
    source_class: BlockchainSourceClass
    snapshot_digest: str = Field(pattern=_DIGEST)
    manifest_path: str
    manifest_file_digest: str = Field(pattern=_DIGEST)
    corpus_path: str
    corpus_file_digest: str = Field(pattern=_DIGEST)
    documents: int = Field(ge=1)
    chain_families: list[ChainFamily]
    threat_domains: list[ThreatDomain]


class BlockchainSourceCatalogManifest(StrictModel):
    schema_version: Literal["sentinel.blockchain-source-catalog-manifest.v1"] = (
        "sentinel.blockchain-source-catalog-manifest.v1"
    )
    ready: bool
    catalog_id: str
    catalog_digest: str = Field(pattern=_DIGEST)
    policy_id: str
    policy_digest: str = Field(pattern=_DIGEST)
    sources: int
    documents: int
    high_trust_share_bps: int
    synthetic_source_share_bps: int
    max_single_source_document_share_bps: int
    source_class_counts: dict[str, int]
    trust_tier_counts: dict[str, int]
    chain_source_counts: dict[str, int]
    threat_source_counts: dict[str, int]
    duplicate_snapshot_digests: list[str]
    duplicate_content_digests: list[str]
    missing_required_chain_families: list[str]
    missing_required_threat_domains: list[str]
    combined_corpus_digest: str = Field(pattern=_DIGEST)
    bindings: list[CatalogSourceBinding]
    violations: list[str]


class BlockchainSourceCatalogResult(StrictModel):
    manifest: BlockchainSourceCatalogManifest
    documents: list[BlockchainSecurityDocument]


def load_source_catalog(path: str | Path) -> BlockchainSourceCatalog:
    try:
        return BlockchainSourceCatalog.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError("invalid blockchain source catalog") from exc


def load_source_catalog_policy(path: str | Path) -> BlockchainSourceCatalogPolicy:
    try:
        return BlockchainSourceCatalogPolicy.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid blockchain source catalog policy") from exc


def audit_source_catalog(
    catalog: BlockchainSourceCatalog,
    policy: BlockchainSourceCatalogPolicy,
    *,
    root: str | Path = ".",
) -> BlockchainSourceCatalogResult:
    root_path = Path(root).resolve()
    bindings: list[CatalogSourceBinding] = []
    documents: list[BlockchainSecurityDocument] = []
    source_classes: Counter[str] = Counter()
    trust_tiers: Counter[str] = Counter()
    chain_sources: Counter[str] = Counter()
    threat_sources: Counter[str] = Counter()
    snapshot_counts: Counter[str] = Counter()
    content_sources: dict[str, str] = {}
    duplicate_content: set[str] = set()
    violations: list[str] = []

    for entry in sorted(catalog.sources, key=lambda item: item.source_id):
        manifest_path = _resolve_file(root_path, entry.manifest_path, "source manifest")
        corpus_path = _resolve_file(root_path, entry.corpus_path, "source corpus")
        try:
            manifest = BlockchainSourceSnapshotManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except ValueError as exc:
            raise ValueError(f"invalid source manifest: {entry.source_id}") from exc

        if manifest.source_id != entry.source_id:
            raise ValueError(f"catalog source_id does not match manifest: {entry.source_id}")
        if manifest.snapshot_digest != entry.expected_snapshot_digest:
            raise ValueError(f"snapshot digest pin mismatch: {entry.source_id}")

        corpus_bytes = corpus_path.read_bytes()
        corpus_digest = hashlib.sha256(corpus_bytes).hexdigest()
        if corpus_digest != manifest.corpus_file_digest:
            raise ValueError(f"source corpus digest does not match manifest: {entry.source_id}")

        rows = load_blockchain_security_documents(corpus_path)
        if len(rows) != manifest.documents:
            raise ValueError(f"source corpus document count mismatch: {entry.source_id}")
        row_refs = [item.document_ref for item in rows]
        if row_refs != sorted(row_refs) or len(row_refs) != len(set(row_refs)):
            raise ValueError(f"source corpus rows are not canonical and unique: {entry.source_id}")

        for row in rows:
            if row.source_snapshot_digest != manifest.snapshot_digest:
                raise ValueError(f"document snapshot lineage mismatch: {entry.source_id}")
            if row.source_class != manifest.source_class:
                raise ValueError(f"document source class mismatch: {entry.source_id}")
            if row.rights_basis != manifest.rights_basis:
                raise ValueError(f"document rights basis mismatch: {entry.source_id}")
            if row.chain_families != manifest.chain_families:
                raise ValueError(f"document chain labels mismatch: {entry.source_id}")
            if row.threat_domains != manifest.threat_domains:
                raise ValueError(f"document threat labels mismatch: {entry.source_id}")
            if row.family_refs != manifest.family_refs:
                raise ValueError(f"document family lineage mismatch: {entry.source_id}")
            previous = content_sources.setdefault(row.content_digest, entry.source_id)
            if previous != entry.source_id:
                duplicate_content.add(row.content_digest)

        snapshot_counts[manifest.snapshot_digest] += 1
        source_classes[manifest.source_class.value] += 1
        trust_tiers[entry.trust_tier.value] += 1
        for chain in set(manifest.chain_families):
            chain_sources[chain.value] += 1
        for threat in set(manifest.threat_domains):
            threat_sources[threat.value] += 1

        bindings.append(
            CatalogSourceBinding(
                source_id=entry.source_id,
                trust_tier=entry.trust_tier,
                source_class=manifest.source_class,
                snapshot_digest=manifest.snapshot_digest,
                manifest_path=entry.manifest_path,
                manifest_file_digest=_hash_file(manifest_path),
                corpus_path=entry.corpus_path,
                corpus_file_digest=corpus_digest,
                documents=manifest.documents,
                chain_families=manifest.chain_families,
                threat_domains=manifest.threat_domains,
            )
        )
        documents.extend(rows)

    duplicate_snapshots = sorted(
        digest for digest, count in snapshot_counts.items() if count > 1
    )
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
    missing_chains = sorted(required_chains.difference(chain_sources))
    missing_threats = sorted(required_threats.difference(threat_sources))
    if missing_chains:
        violations.append("required chain families are missing from source catalog")
    if missing_threats:
        violations.append("required threat domains are missing from source catalog")

    for chain in sorted(required_chains):
        count = chain_sources[chain]
        if count < policy.min_sources_per_required_chain:
            violations.append(
                f"chain {chain} sources {count} below minimum "
                f"{policy.min_sources_per_required_chain}"
            )
    for threat in sorted(required_threats):
        count = threat_sources[threat]
        if count < policy.min_sources_per_required_threat_domain:
            violations.append(
                f"threat domain {threat} sources {count} below minimum "
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
        1 for binding in bindings if binding.source_class is BlockchainSourceClass.KOSCHEI_SYNTHETIC
    )
    synthetic_share = _share_bps(synthetic_sources, len(bindings))
    if synthetic_share > policy.max_synthetic_source_share_bps:
        violations.append(
            f"synthetic source share {synthetic_share} bps exceeds maximum "
            f"{policy.max_synthetic_source_share_bps} bps"
        )

    total_documents = len(documents)
    max_source_documents = max((binding.documents for binding in bindings), default=0)
    max_source_document_share = _share_bps(max_source_documents, total_documents)
    if max_source_document_share > policy.max_single_source_document_share_bps:
        violations.append(
            f"single-source document share {max_source_document_share} bps exceeds maximum "
            f"{policy.max_single_source_document_share_bps} bps"
        )

    document_refs = [item.document_ref for item in documents]
    if len(document_refs) != len(set(document_refs)):
        violations.append("duplicate document_ref detected across source snapshots")

    ordered_documents = sorted(documents, key=lambda item: item.document_ref)
    combined_digest = _documents_digest(ordered_documents)
    manifest = BlockchainSourceCatalogManifest(
        ready=not violations,
        catalog_id=catalog.catalog_id,
        catalog_digest=_model_digest(catalog),
        policy_id=policy.policy_id,
        policy_digest=_model_digest(policy),
        sources=len(bindings),
        documents=total_documents,
        high_trust_share_bps=high_trust_share,
        synthetic_source_share_bps=synthetic_share,
        max_single_source_document_share_bps=max_source_document_share,
        source_class_counts=dict(sorted(source_classes.items())),
        trust_tier_counts=dict(sorted(trust_tiers.items())),
        chain_source_counts=dict(sorted(chain_sources.items())),
        threat_source_counts=dict(sorted(threat_sources.items())),
        duplicate_snapshot_digests=duplicate_snapshots,
        duplicate_content_digests=sorted(duplicate_content),
        missing_required_chain_families=missing_chains,
        missing_required_threat_domains=missing_threats,
        combined_corpus_digest=combined_digest,
        bindings=bindings,
        violations=violations,
    )
    return BlockchainSourceCatalogResult(manifest=manifest, documents=ordered_documents)


def write_catalog_outputs(
    result: BlockchainSourceCatalogResult,
    *,
    corpus_path: str | Path,
    manifest_path: str | Path,
) -> None:
    if not result.manifest.ready:
        raise ValueError("source catalog is not ready; refusing to materialize training corpus")
    corpus = Path(corpus_path)
    manifest = Path(manifest_path)
    if corpus.exists():
        raise FileExistsError(f"composed blockchain corpus already exists: {corpus}")
    if manifest.exists():
        raise FileExistsError(f"source catalog manifest already exists: {manifest}")
    corpus.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    corpus_payload = _documents_payload(result.documents)
    if hashlib.sha256(corpus_payload.encode("utf-8")).hexdigest() != (
        result.manifest.combined_corpus_digest
    ):
        raise ValueError("composed corpus digest is inconsistent")
    _atomic_no_replace(corpus, corpus_payload)
    try:
        _atomic_no_replace(
            manifest,
            json.dumps(result.manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        )
    except Exception:
        try:
            corpus.unlink()
        except FileNotFoundError:
            pass
        raise


def _documents_payload(documents: list[BlockchainSecurityDocument]) -> str:
    return "".join(
        json.dumps(
            item.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
        for item in documents
    )


def _documents_digest(documents: list[BlockchainSecurityDocument]) -> str:
    return hashlib.sha256(_documents_payload(documents).encode("utf-8")).hexdigest()


def _model_digest(model: StrictModel) -> str:
    payload = json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _validate_relative_path(value: str, field: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or value.startswith("~")
        or "\\" in value
    ):
        raise ValueError(f"{field} must stay within the repository root")


def _atomic_no_replace(path: Path, payload: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    published = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"artifact already exists: {path}") from None
        published = True
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if published:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
