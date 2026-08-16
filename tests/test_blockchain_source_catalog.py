from __future__ import annotations

import json
from pathlib import Path

import pytest

from koschei_sentinel.blockchain_security_corpus import (
    BlockchainSourceClass,
    ChainFamily,
    ThreatDomain,
)
from koschei_sentinel.blockchain_security_ingest import (
    BlockchainSourceSnapshotSpec,
    ingest_snapshot,
    inspect_snapshot,
    snapshot_digest,
    write_ingest_result,
)
from koschei_sentinel.blockchain_source_catalog import (
    BlockchainCatalogSource,
    BlockchainSourceCatalog,
    BlockchainSourceCatalogPolicy,
    SourceTrustTier,
    audit_source_catalog,
    write_catalog_outputs,
)
from koschei_sentinel.pretraining_corpus import RightsBasis


def _source(
    root: Path,
    *,
    source_id: str,
    source_class: BlockchainSourceClass,
    chain: ChainFamily,
    threat: ThreatDomain,
    text: str,
) -> BlockchainCatalogSource:
    snapshot = root / "snapshots" / source_id
    snapshot.mkdir(parents=True)
    (snapshot / "document.md").write_text(text, encoding="utf-8")
    provisional = BlockchainSourceSnapshotSpec(
        source_id=source_id,
        source_class=source_class,
        rights_basis=RightsBasis.KOSCHEI_OWNED,
        snapshot_path=snapshot.relative_to(root).as_posix(),
        expected_snapshot_digest="0" * 64,
        chain_families=[chain],
        threat_domains=[threat],
    )
    _, files = inspect_snapshot(provisional, root=root)
    spec = provisional.model_copy(
        update={"expected_snapshot_digest": snapshot_digest(files)}
    )
    result = ingest_snapshot(spec, root=root)
    output = root / "ingested" / source_id
    corpus = output / "corpus.jsonl"
    manifest = output / "manifest.json"
    write_ingest_result(result, corpus_path=corpus, manifest_path=manifest)
    return BlockchainCatalogSource(
        source_id=source_id,
        trust_tier=SourceTrustTier.PRIMARY_PROTOCOL,
        manifest_path=manifest.relative_to(root).as_posix(),
        corpus_path=corpus.relative_to(root).as_posix(),
        expected_snapshot_digest=result.manifest.snapshot_digest,
    )


def _catalog(root: Path) -> BlockchainSourceCatalog:
    sources = [
        _source(
            root,
            source_id="evm.protocol",
            source_class=BlockchainSourceClass.PROTOCOL_SOURCE,
            chain=ChainFamily.EVM,
            threat=ThreatDomain.SMART_CONTRACT,
            text="EVM protocol security boundary.\n",
        ),
        _source(
            root,
            source_id="solana.audit",
            source_class=BlockchainSourceClass.AUDIT_FINDING,
            chain=ChainFamily.SOLANA,
            threat=ThreatDomain.PRIVILEGED_ACCESS,
            text="Solana privileged access review.\n",
        ),
        _source(
            root,
            source_id="bitcoin.incident",
            source_class=BlockchainSourceClass.INCIDENT_POSTMORTEM,
            chain=ChainFamily.BITCOIN,
            threat=ThreatDomain.KEY_WALLET,
            text="Bitcoin wallet incident analysis.\n",
        ),
        _source(
            root,
            source_id="bridge.spec",
            source_class=BlockchainSourceClass.FORMAL_SPECIFICATION,
            chain=ChainFamily.CROSS_CHAIN,
            threat=ThreatDomain.BRIDGE_CROSS_CHAIN,
            text="Cross-chain bridge invariant specification.\n",
        ),
    ]
    return BlockchainSourceCatalog(catalog_id="catalog-test", sources=sources)


def _policy(**updates) -> BlockchainSourceCatalogPolicy:
    payload = {
        "policy_id": "catalog-test",
        "min_sources": 4,
        "min_source_classes": 4,
        "required_chain_families": ["EVM", "SOLANA", "BITCOIN", "CROSS_CHAIN"],
        "required_threat_domains": [
            "SMART_CONTRACT",
            "PRIVILEGED_ACCESS",
            "KEY_WALLET",
            "BRIDGE_CROSS_CHAIN",
        ],
        "min_sources_per_required_chain": 1,
        "min_sources_per_required_threat_domain": 1,
        "min_high_trust_share_bps": 7500,
        "max_synthetic_source_share_bps": 2500,
        "max_single_source_document_share_bps": 2500,
    }
    payload.update(updates)
    return BlockchainSourceCatalogPolicy.model_validate(payload)


def test_diverse_source_catalog_passes_and_is_deterministic(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    first = audit_source_catalog(catalog, _policy(), root=tmp_path)
    second = audit_source_catalog(catalog, _policy(), root=tmp_path)

    assert first == second
    assert first.manifest.ready is True
    assert first.manifest.sources == 4
    assert first.manifest.documents == 4
    assert first.manifest.high_trust_share_bps == 10000
    assert first.manifest.violations == []


def test_snapshot_pin_drift_is_rejected(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    changed = catalog.sources[0].model_copy(
        update={"expected_snapshot_digest": "f" * 64}
    )
    catalog = catalog.model_copy(update={"sources": [changed, *catalog.sources[1:]]})

    with pytest.raises(ValueError, match="snapshot digest pin mismatch"):
        audit_source_catalog(catalog, _policy(), root=tmp_path)


def test_cross_source_duplicate_content_blocks_readiness(tmp_path: Path) -> None:
    first = _source(
        tmp_path,
        source_id="duplicate.one",
        source_class=BlockchainSourceClass.PROTOCOL_SOURCE,
        chain=ChainFamily.EVM,
        threat=ThreatDomain.SMART_CONTRACT,
        text="Shared duplicate security text.\n",
    )
    second = _source(
        tmp_path,
        source_id="duplicate.two",
        source_class=BlockchainSourceClass.AUDIT_FINDING,
        chain=ChainFamily.SOLANA,
        threat=ThreatDomain.PRIVILEGED_ACCESS,
        text="Shared duplicate security text.\n",
    )
    catalog = BlockchainSourceCatalog(catalog_id="duplicates", sources=[first, second])
    policy = _policy(
        min_sources=2,
        min_source_classes=2,
        required_chain_families=["EVM", "SOLANA"],
        required_threat_domains=["SMART_CONTRACT", "PRIVILEGED_ACCESS"],
        min_high_trust_share_bps=0,
        max_single_source_document_share_bps=5000,
    )

    result = audit_source_catalog(catalog, policy, root=tmp_path)
    assert result.manifest.ready is False
    assert result.manifest.duplicate_content_digests
    assert "duplicate document content" in " ".join(result.manifest.violations)


def test_low_trust_catalog_is_rejected(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    catalog = catalog.model_copy(
        update={
            "sources": [
                item.model_copy(update={"trust_tier": SourceTrustTier.PUBLIC_RESEARCH})
                for item in catalog.sources
            ]
        }
    )
    result = audit_source_catalog(catalog, _policy(), root=tmp_path)
    assert result.manifest.ready is False
    assert result.manifest.high_trust_share_bps == 0
    assert "high-trust source share" in " ".join(result.manifest.violations)


def test_ready_catalog_writes_no_replace_composed_artifacts(tmp_path: Path) -> None:
    result = audit_source_catalog(_catalog(tmp_path), _policy(), root=tmp_path)
    corpus = tmp_path / "composed" / "corpus.jsonl"
    manifest = tmp_path / "composed" / "manifest.json"

    write_catalog_outputs(result, corpus_path=corpus, manifest_path=manifest)

    rows = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines()]
    stored = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(rows) == 4
    assert stored["combined_corpus_digest"] == result.manifest.combined_corpus_digest
    with pytest.raises(FileExistsError):
        write_catalog_outputs(result, corpus_path=corpus, manifest_path=manifest)


def test_repository_source_catalog_policy_is_fail_closed() -> None:
    root = Path(__file__).parents[1]
    payload = json.loads(
        (root / "configs" / "pretraining" / "blockchain-source-catalog.v1.json").read_text()
    )
    policy = BlockchainSourceCatalogPolicy.model_validate(payload)
    assert policy.min_sources >= 200
    assert policy.min_source_classes >= 5
    assert policy.min_sources_per_required_chain >= 10
    assert policy.min_sources_per_required_threat_domain >= 10
    assert policy.min_high_trust_share_bps >= 6000
    assert policy.max_synthetic_source_share_bps <= 1500
    assert policy.max_single_source_document_share_bps <= 500
    assert set(policy.required_chain_families) >= {
        ChainFamily.EVM,
        ChainFamily.SOLANA,
        ChainFamily.BITCOIN,
        ChainFamily.COSMOS,
        ChainFamily.MOVE,
        ChainFamily.TRON,
        ChainFamily.CROSS_CHAIN,
        ChainFamily.OFF_CHAIN,
    }
    assert set(policy.required_threat_domains) >= {
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
