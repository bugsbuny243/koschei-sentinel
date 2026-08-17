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
from koschei_sentinel.pretraining_corpus import PretrainingHoldoutSet, RightsBasis

_DIGEST = r"^[a-f0-9]{64}$"
_PSEUDONYM = r"^[a-z][a-z0-9_]*_[a-f0-9]{24}$"


class BlockchainSourceClass(StrEnum):
    PROTOCOL_SOURCE = "PROTOCOL_SOURCE"
    PUBLIC_SECURITY_REPORT = "PUBLIC_SECURITY_REPORT"
    INCIDENT_POSTMORTEM = "INCIDENT_POSTMORTEM"
    AUDIT_FINDING = "AUDIT_FINDING"
    FORMAL_SPECIFICATION = "FORMAL_SPECIFICATION"
    KOSCHEI_SECURITY_CASE = "KOSCHEI_SECURITY_CASE"
    KOSCHEI_SYNTHETIC = "KOSCHEI_SYNTHETIC"


class ChainFamily(StrEnum):
    EVM = "EVM"
    SOLANA = "SOLANA"
    BITCOIN = "BITCOIN"
    COSMOS = "COSMOS"
    MOVE = "MOVE"
    TRON = "TRON"
    TON = "TON"
    CROSS_CHAIN = "CROSS_CHAIN"
    OFF_CHAIN = "OFF_CHAIN"


class ThreatDomain(StrEnum):
    SMART_CONTRACT = "SMART_CONTRACT"
    KEY_WALLET = "KEY_WALLET"
    PRIVILEGED_ACCESS = "PRIVILEGED_ACCESS"
    SIGNING_UI = "SIGNING_UI"
    BRIDGE_CROSS_CHAIN = "BRIDGE_CROSS_CHAIN"
    ORACLE_PRICE = "ORACLE_PRICE"
    RPC_INFRASTRUCTURE = "RPC_INFRASTRUCTURE"
    SUPPLY_CHAIN = "SUPPLY_CHAIN"
    CONSENSUS_VALIDATOR = "CONSENSUS_VALIDATOR"
    SOCIAL_ENGINEERING = "SOCIAL_ENGINEERING"
    TOKEN_ECONOMICS = "TOKEN_ECONOMICS"
    INCIDENT_RESPONSE = "INCIDENT_RESPONSE"


INFRASTRUCTURE_DOMAINS = frozenset(
    {
        ThreatDomain.KEY_WALLET,
        ThreatDomain.PRIVILEGED_ACCESS,
        ThreatDomain.SIGNING_UI,
        ThreatDomain.RPC_INFRASTRUCTURE,
        ThreatDomain.SUPPLY_CHAIN,
        ThreatDomain.SOCIAL_ENGINEERING,
    }
)


class BlockchainSecurityDocument(StrictModel):
    schema_version: Literal["sentinel.blockchain-security-document.v1"] = (
        "sentinel.blockchain-security-document.v1"
    )
    document_ref: str = Field(pattern=_PSEUDONYM)
    source_class: BlockchainSourceClass
    rights_basis: RightsBasis
    source_snapshot_digest: str = Field(pattern=_DIGEST)
    content_digest: str = Field(pattern=_DIGEST)
    family_refs: list[str] = Field(default_factory=list, max_length=128)
    chain_families: list[ChainFamily] = Field(min_length=1, max_length=16)
    threat_domains: list[ThreatDomain] = Field(min_length=1, max_length=32)
    text: str = Field(min_length=1, max_length=100_000)

    @model_validator(mode="after")
    def document_is_safe_and_bound(self) -> BlockchainSecurityDocument:
        actual = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.content_digest != actual:
            raise ValueError("content_digest does not match document text")
        if len(self.family_refs) != len(set(self.family_refs)):
            raise ValueError("family_refs must be unique")
        if len(self.chain_families) != len(set(self.chain_families)):
            raise ValueError("chain_families must be unique")
        if len(self.threat_domains) != len(set(self.threat_domains)):
            raise ValueError("threat_domains must be unique")
        for family_ref in self.family_refs:
            if not _is_pseudonym(family_ref):
                raise ValueError("family_refs must be pseudonymized identifiers")
        findings = detect_sensitive_text(self.text)
        if findings:
            raise ValueError(
                "blockchain security text contains sensitive or raw identifier material: "
                + ", ".join(findings)
            )
        return self


class BlockchainSecurityCorpusPolicy(StrictModel):
    schema_version: Literal["sentinel.blockchain-security-corpus-policy.v1"] = (
        "sentinel.blockchain-security-corpus-policy.v1"
    )
    policy_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    min_documents: int = Field(default=10_000, ge=1, le=100_000_000)
    min_source_classes: int = Field(default=4, ge=1, le=len(BlockchainSourceClass))
    min_unique_families: int = Field(default=500, ge=0, le=10_000_000)
    max_single_family_bps: int = Field(default=200, ge=1, le=10_000)
    required_chain_families: list[ChainFamily] = Field(min_length=1)
    required_threat_domains: list[ThreatDomain] = Field(min_length=1)
    min_documents_per_required_chain: int = Field(default=100, ge=1, le=100_000_000)
    min_documents_per_required_threat_domain: int = Field(
        default=100,
        ge=1,
        le=100_000_000,
    )
    min_infrastructure_share_bps: int = Field(default=2500, ge=0, le=10_000)
    max_smart_contract_only_share_bps: int = Field(default=6000, ge=0, le=10_000)
    require_zero_holdout_content_overlap: Literal[True] = True
    require_zero_holdout_family_overlap: Literal[True] = True
    require_known_rights_basis: Literal[True] = True
    raw_sensitive_text_allowed: Literal[False] = False

    @model_validator(mode="after")
    def policy_is_canonical(self) -> BlockchainSecurityCorpusPolicy:
        if len(self.required_chain_families) != len(set(self.required_chain_families)):
            raise ValueError("required_chain_families must be unique")
        if len(self.required_threat_domains) != len(set(self.required_threat_domains)):
            raise ValueError("required_threat_domains must be unique")
        return self


class BlockchainSecurityCorpusAudit(StrictModel):
    schema_version: Literal["sentinel.blockchain-security-corpus-audit.v1"] = (
        "sentinel.blockchain-security-corpus-audit.v1"
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
    chain_family_counts: dict[str, int]
    threat_domain_counts: dict[str, int]
    rights_basis_counts: dict[str, int]
    infrastructure_share_bps: int
    smart_contract_only_share_bps: int
    max_single_family_bps: int
    duplicate_content_digests: list[str]
    holdout_content_hits: list[str]
    holdout_family_hits: list[str]
    missing_required_chain_families: list[str]
    missing_required_threat_domains: list[str]
    violations: list[str]


def audit_blockchain_security_corpus(
    documents: list[BlockchainSecurityDocument],
    holdout: PretrainingHoldoutSet,
    policy: BlockchainSecurityCorpusPolicy,
) -> BlockchainSecurityCorpusAudit:
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
    chain_counts = Counter(
        chain.value for item in ordered for chain in item.chain_families
    )
    threat_counts = Counter(
        domain.value for item in ordered for domain in item.threat_domains
    )
    family_counts = Counter(
        family_ref for item in ordered for family_ref in item.family_refs
    )

    content_hits = sorted(set(content_counts).intersection(holdout.content_digests))
    if content_hits:
        violations.append("blockchain security corpus overlaps held-out benchmark content")

    family_hits = sorted(set(family_counts).intersection(holdout.family_refs))
    if family_hits:
        violations.append(
            "blockchain security corpus overlaps held-out actor/incident families"
        )

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

    required_chains = {item.value for item in policy.required_chain_families}
    missing_chains = sorted(required_chains.difference(chain_counts))
    if missing_chains:
        violations.append("required chain-family coverage is incomplete")

    required_threats = {item.value for item in policy.required_threat_domains}
    missing_threats = sorted(required_threats.difference(threat_counts))
    if missing_threats:
        violations.append("required threat-domain coverage is incomplete")

    for chain in sorted(required_chains):
        count = chain_counts[chain]
        if count < policy.min_documents_per_required_chain:
            violations.append(
                f"chain {chain} documents {count} below minimum "
                f"{policy.min_documents_per_required_chain}"
            )

    for domain in sorted(required_threats):
        count = threat_counts[domain]
        if count < policy.min_documents_per_required_threat_domain:
            violations.append(
                f"threat domain {domain} documents {count} below minimum "
                f"{policy.min_documents_per_required_threat_domain}"
            )

    max_family_bps = _share_bps(max(family_counts.values(), default=0), sum(family_counts.values()))
    if max_family_bps > policy.max_single_family_bps:
        violations.append(
            f"single-family concentration {max_family_bps} bps exceeds maximum "
            f"{policy.max_single_family_bps} bps"
        )

    infrastructure_documents = sum(
        1
        for item in ordered
        if INFRASTRUCTURE_DOMAINS.intersection(item.threat_domains)
    )
    infrastructure_share_bps = _share_bps(infrastructure_documents, len(ordered))
    if infrastructure_share_bps < policy.min_infrastructure_share_bps:
        violations.append(
            f"infrastructure-security share {infrastructure_share_bps} bps below minimum "
            f"{policy.min_infrastructure_share_bps} bps"
        )

    smart_contract_only_documents = sum(
        1
        for item in ordered
        if set(item.threat_domains) == {ThreatDomain.SMART_CONTRACT}
    )
    smart_contract_only_share_bps = _share_bps(
        smart_contract_only_documents,
        len(ordered),
    )
    if smart_contract_only_share_bps > policy.max_smart_contract_only_share_bps:
        violations.append(
            f"smart-contract-only share {smart_contract_only_share_bps} bps exceeds maximum "
            f"{policy.max_smart_contract_only_share_bps} bps"
        )

    return BlockchainSecurityCorpusAudit(
        ready=not violations,
        policy_id=policy.policy_id,
        corpus_digest=_digest_documents(ordered),
        policy_digest=_digest_model(policy),
        holdout_digest=_digest_model(holdout),
        benchmark_suite_digest=holdout.benchmark_suite_digest,
        documents=len(ordered),
        unique_families=len(family_counts),
        source_class_counts=dict(sorted(source_counts.items())),
        chain_family_counts=dict(sorted(chain_counts.items())),
        threat_domain_counts=dict(sorted(threat_counts.items())),
        rights_basis_counts=dict(sorted(rights_counts.items())),
        infrastructure_share_bps=infrastructure_share_bps,
        smart_contract_only_share_bps=smart_contract_only_share_bps,
        max_single_family_bps=max_family_bps,
        duplicate_content_digests=duplicate_content,
        holdout_content_hits=content_hits,
        holdout_family_hits=family_hits,
        missing_required_chain_families=missing_chains,
        missing_required_threat_domains=missing_threats,
        violations=violations,
    )


def load_blockchain_security_documents(
    path: str | Path,
) -> list[BlockchainSecurityDocument]:
    source = Path(path)
    rows: list[BlockchainSecurityDocument] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank blockchain security corpus row at line {line_number}")
        try:
            rows.append(BlockchainSecurityDocument.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(
                f"invalid blockchain security corpus row at line {line_number}"
            ) from exc
    if not rows:
        raise ValueError("blockchain security corpus contains no documents")
    return rows


def load_blockchain_security_policy(
    path: str | Path,
) -> BlockchainSecurityCorpusPolicy:
    try:
        return BlockchainSecurityCorpusPolicy.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid blockchain security corpus policy") from exc


def write_blockchain_security_audit(
    audit: BlockchainSecurityCorpusAudit,
    path: str | Path,
) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(
            f"blockchain security corpus audit already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(audit.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
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


def _share_bps(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        return 0
    return (numerator * 10_000 + denominator - 1) // denominator


def _digest_documents(documents: list[BlockchainSecurityDocument]) -> str:
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


def _is_pseudonym(value: str) -> bool:
    prefix, separator, digest = value.rpartition("_")
    prefix_is_valid = all(
        character.islower() or character.isdigit() or character == "_"
        for character in prefix
    )
    return bool(
        separator
        and prefix
        and prefix[0].isalpha()
        and prefix_is_valid
        and len(digest) == 24
        and all(character in "0123456789abcdef" for character in digest)
    )
