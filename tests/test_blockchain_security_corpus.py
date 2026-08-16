from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from koschei_sentinel.blockchain_security_corpus import (
    BlockchainSecurityCorpusPolicy,
    BlockchainSecurityDocument,
    BlockchainSourceClass,
    ChainFamily,
    ThreatDomain,
    audit_blockchain_security_corpus,
    content_digest,
)
from koschei_sentinel.pretraining_corpus import PretrainingHoldoutSet, RightsBasis


def pseudonym(prefix: str, character: str) -> str:
    return f"{prefix}_{character * 24}"


def document(
    index: int,
    *,
    source_class: BlockchainSourceClass,
    chains: list[ChainFamily],
    threats: list[ThreatDomain],
    text: str | None = None,
) -> BlockchainSecurityDocument:
    body = text or f"Sanitized blockchain security training document {index}."
    return BlockchainSecurityDocument(
        document_ref=pseudonym("doc", format(index, "x")[-1]),
        source_class=source_class,
        rights_basis=RightsBasis.KOSCHEI_OWNED,
        source_snapshot_digest=format(index + 1, "064x"),
        content_digest=content_digest(body),
        family_refs=[pseudonym("family", format(index + 8, "x")[-1])],
        chain_families=chains,
        threat_domains=threats,
        text=body,
    )


def holdout(**updates) -> PretrainingHoldoutSet:
    payload = {
        "benchmark_suite_digest": "a" * 64,
        "content_digests": [],
        "family_refs": [],
    }
    payload.update(updates)
    return PretrainingHoldoutSet.model_validate(payload)


def policy(**updates) -> BlockchainSecurityCorpusPolicy:
    payload = {
        "policy_id": "multichain-test",
        "min_documents": 4,
        "min_source_classes": 4,
        "min_unique_families": 4,
        "max_single_family_bps": 2500,
        "required_chain_families": ["EVM", "SOLANA", "BITCOIN", "CROSS_CHAIN"],
        "required_threat_domains": [
            "SMART_CONTRACT",
            "KEY_WALLET",
            "PRIVILEGED_ACCESS",
            "BRIDGE_CROSS_CHAIN",
        ],
        "min_documents_per_required_chain": 1,
        "min_documents_per_required_threat_domain": 1,
        "min_infrastructure_share_bps": 2500,
        "max_smart_contract_only_share_bps": 5000,
    }
    payload.update(updates)
    return BlockchainSecurityCorpusPolicy.model_validate(payload)


def safe_documents() -> list[BlockchainSecurityDocument]:
    return [
        document(
            1,
            source_class=BlockchainSourceClass.PROTOCOL_SOURCE,
            chains=[ChainFamily.EVM],
            threats=[ThreatDomain.SMART_CONTRACT],
        ),
        document(
            2,
            source_class=BlockchainSourceClass.AUDIT_FINDING,
            chains=[ChainFamily.SOLANA],
            threats=[ThreatDomain.PRIVILEGED_ACCESS],
        ),
        document(
            3,
            source_class=BlockchainSourceClass.INCIDENT_POSTMORTEM,
            chains=[ChainFamily.BITCOIN],
            threats=[ThreatDomain.KEY_WALLET],
        ),
        document(
            4,
            source_class=BlockchainSourceClass.PUBLIC_SECURITY_REPORT,
            chains=[ChainFamily.CROSS_CHAIN],
            threats=[ThreatDomain.BRIDGE_CROSS_CHAIN],
        ),
    ]


def test_diverse_multichain_corpus_passes() -> None:
    audit = audit_blockchain_security_corpus(safe_documents(), holdout(), policy())
    assert audit.ready is True
    assert audit.documents == 4
    assert audit.infrastructure_share_bps == 5000
    assert audit.smart_contract_only_share_bps == 2500
    assert audit.missing_required_chain_families == []
    assert audit.missing_required_threat_domains == []
    assert audit.violations == []


def test_missing_chain_family_blocks_readiness() -> None:
    documents = [
        item
        for item in safe_documents()
        if ChainFamily.BITCOIN not in item.chain_families
    ]
    audit = audit_blockchain_security_corpus(
        documents,
        holdout(),
        policy(min_documents=3, min_source_classes=3, min_unique_families=3),
    )
    assert audit.ready is False
    assert audit.missing_required_chain_families == ["BITCOIN"]
    assert "chain-family coverage" in " ".join(audit.violations)


def test_missing_threat_domain_blocks_readiness() -> None:
    documents = [
        item.model_copy(
            update={"threat_domains": [ThreatDomain.PRIVILEGED_ACCESS]}
        )
        if ThreatDomain.KEY_WALLET in item.threat_domains
        else item
        for item in safe_documents()
    ]
    audit = audit_blockchain_security_corpus(documents, holdout(), policy())
    assert audit.ready is False
    assert audit.missing_required_threat_domains == ["KEY_WALLET"]


def test_infrastructure_underweight_is_rejected() -> None:
    documents = [
        item.model_copy(update={"threat_domains": [ThreatDomain.SMART_CONTRACT]})
        for item in safe_documents()
    ]
    audit = audit_blockchain_security_corpus(
        documents,
        holdout(),
        policy(
            required_threat_domains=["SMART_CONTRACT"],
            min_infrastructure_share_bps=2500,
            max_smart_contract_only_share_bps=10000,
        ),
    )
    assert audit.ready is False
    assert audit.infrastructure_share_bps == 0
    assert "infrastructure-security share" in " ".join(audit.violations)


def test_smart_contract_only_overweight_is_rejected() -> None:
    documents = [
        item.model_copy(update={"threat_domains": [ThreatDomain.SMART_CONTRACT]})
        for item in safe_documents()
    ]
    audit = audit_blockchain_security_corpus(
        documents,
        holdout(),
        policy(
            required_threat_domains=["SMART_CONTRACT"],
            min_infrastructure_share_bps=0,
            max_smart_contract_only_share_bps=5000,
        ),
    )
    assert audit.ready is False
    assert audit.smart_contract_only_share_bps == 10000
    assert "smart-contract-only share" in " ".join(audit.violations)


def test_sensitive_text_is_rejected_before_audit() -> None:
    with pytest.raises(ValidationError, match="sensitive or raw identifier"):
        document(
            1,
            source_class=BlockchainSourceClass.INCIDENT_POSTMORTEM,
            chains=[ChainFamily.OFF_CHAIN],
            threats=[ThreatDomain.SOCIAL_ENGINEERING],
            text="Send this corpus export to analyst@example.com.",
        )


def test_holdout_family_overlap_blocks_readiness() -> None:
    documents = safe_documents()
    family = documents[0].family_refs[0]
    audit = audit_blockchain_security_corpus(
        documents,
        holdout(family_refs=[family]),
        policy(),
    )
    assert audit.ready is False
    assert audit.holdout_family_hits == [family]


def test_repository_multichain_policy_is_fail_closed() -> None:
    root = Path(__file__).parents[1]
    payload = json.loads(
        (root / "configs" / "pretraining" / "blockchain-security.v1.json").read_text()
    )
    repository_policy = BlockchainSecurityCorpusPolicy.model_validate(payload)
    assert repository_policy.min_documents >= 50_000
    assert repository_policy.min_source_classes >= 5
    assert repository_policy.min_unique_families >= 2_000
    assert repository_policy.max_single_family_bps <= 100
    assert repository_policy.min_infrastructure_share_bps >= 3000
    assert repository_policy.max_smart_contract_only_share_bps <= 5500
    assert set(repository_policy.required_chain_families) >= {
        ChainFamily.EVM,
        ChainFamily.SOLANA,
        ChainFamily.BITCOIN,
        ChainFamily.COSMOS,
        ChainFamily.MOVE,
        ChainFamily.TRON,
        ChainFamily.CROSS_CHAIN,
        ChainFamily.OFF_CHAIN,
    }
    assert set(repository_policy.required_threat_domains) >= {
        ThreatDomain.SMART_CONTRACT,
        ThreatDomain.KEY_WALLET,
        ThreatDomain.PRIVILEGED_ACCESS,
        ThreatDomain.SIGNING_UI,
        ThreatDomain.BRIDGE_CROSS_CHAIN,
        ThreatDomain.ORACLE_PRICE,
        ThreatDomain.RPC_INFRASTRUCTURE,
        ThreatDomain.SUPPLY_CHAIN,
        ThreatDomain.CONSENSUS_VALIDATOR,
        ThreatDomain.SOCIAL_ENGINEERING,
        ThreatDomain.INCIDENT_RESPONSE,
    }
    assert repository_policy.require_zero_holdout_content_overlap is True
    assert repository_policy.require_zero_holdout_family_overlap is True
    assert repository_policy.raw_sensitive_text_allowed is False
