from __future__ import annotations

import json
from pathlib import Path

import pytest

from koschei_sentinel.blockchain_security_corpus import (
    BlockchainSecurityCorpusPolicy,
    BlockchainSourceClass,
    ChainFamily,
    ThreatDomain,
    audit_blockchain_security_corpus,
    load_blockchain_security_documents,
    write_blockchain_security_audit,
)
from koschei_sentinel.blockchain_security_ingest import (
    BlockchainSourceSnapshotSpec,
    ingest_snapshot,
    inspect_snapshot,
    snapshot_digest,
    write_ingest_result,
)
from koschei_sentinel.blockchain_security_release import (
    BlockchainSecurityReleasePolicy,
    build_blockchain_security_release,
    verify_blockchain_security_release,
)
from koschei_sentinel.blockchain_source_catalog import (
    BlockchainCatalogSource,
    BlockchainSourceCatalog,
    BlockchainSourceCatalogPolicy,
    SourceTrustTier,
    audit_source_catalog,
    write_catalog_outputs,
)
from koschei_sentinel.pretraining_corpus import PretrainingHoldoutSet, RightsBasis


def _family(index: int) -> str:
    character = "0123456789abcdef"[index]
    return f"family_{character * 24}"


def _write_model(path: Path, model: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _materialize_inputs(
    root: Path,
    *,
    shared_family: bool = False,
) -> tuple[dict[str, Path], list[str]]:
    chains = [ChainFamily.EVM, ChainFamily.SOLANA, ChainFamily.BITCOIN]
    threats = [
        ThreatDomain.SMART_CONTRACT,
        ThreatDomain.PRIVILEGED_ACCESS,
        ThreatDomain.KEY_WALLET,
    ]
    sources: list[BlockchainCatalogSource] = []
    snapshots: list[str] = []

    for index in range(9):
        source_id = f"source.{index}"
        snapshot = root / "snapshots" / source_id
        snapshot.mkdir(parents=True)
        (snapshot / "document.md").write_text(
            f"Independent blockchain security source {index}.\n",
            encoding="utf-8",
        )
        family = _family(0) if shared_family and index in {0, 1} else _family(index)
        provisional = BlockchainSourceSnapshotSpec(
            source_id=source_id,
            source_class=BlockchainSourceClass.PROTOCOL_SOURCE,
            rights_basis=RightsBasis.KOSCHEI_OWNED,
            snapshot_path=snapshot.relative_to(root).as_posix(),
            expected_snapshot_digest="0" * 64,
            chain_families=chains,
            threat_domains=threats,
            family_refs=[family],
        )
        _, files = inspect_snapshot(provisional, root=root)
        spec = provisional.model_copy(
            update={"expected_snapshot_digest": snapshot_digest(files)}
        )
        result = ingest_snapshot(spec, root=root)
        output = root / "ingested" / source_id
        corpus_path = output / "corpus.jsonl"
        manifest_path = output / "manifest.json"
        write_ingest_result(
            result,
            corpus_path=corpus_path,
            manifest_path=manifest_path,
        )
        snapshots.append(result.manifest.snapshot_digest)
        sources.append(
            BlockchainCatalogSource(
                source_id=source_id,
                trust_tier=SourceTrustTier.PRIMARY_PROTOCOL,
                manifest_path=manifest_path.relative_to(root).as_posix(),
                corpus_path=corpus_path.relative_to(root).as_posix(),
                expected_snapshot_digest=result.manifest.snapshot_digest,
            )
        )

    catalog = BlockchainSourceCatalog(catalog_id="release-test", sources=sources)
    catalog_policy = BlockchainSourceCatalogPolicy(
        policy_id="release-test",
        min_sources=9,
        min_source_classes=1,
        required_chain_families=chains,
        required_threat_domains=threats,
        min_sources_per_required_chain=1,
        min_sources_per_required_threat_domain=1,
        min_high_trust_share_bps=10_000,
        max_synthetic_source_share_bps=1,
        max_single_source_document_share_bps=1200,
    )
    catalog_path = root / "inputs/catalog.json"
    catalog_policy_path = root / "inputs/catalog-policy.json"
    _write_model(catalog_path, catalog)
    _write_model(catalog_policy_path, catalog_policy)

    catalog_result = audit_source_catalog(catalog, catalog_policy, root=root)
    assert catalog_result.manifest.ready is True
    composed_corpus = root / "inputs/composed.jsonl"
    source_manifest = root / "inputs/source-manifest.json"
    write_catalog_outputs(
        catalog_result,
        corpus_path=composed_corpus,
        manifest_path=source_manifest,
    )

    holdout = PretrainingHoldoutSet(benchmark_suite_digest="a" * 64)
    holdout_path = root / "inputs/holdout.json"
    _write_model(holdout_path, holdout)

    blockchain_policy = BlockchainSecurityCorpusPolicy(
        policy_id="release-test",
        min_documents=9,
        min_source_classes=1,
        min_unique_families=8 if shared_family else 9,
        max_single_family_bps=2500,
        required_chain_families=chains,
        required_threat_domains=threats,
        min_documents_per_required_chain=1,
        min_documents_per_required_threat_domain=1,
        min_infrastructure_share_bps=5000,
        max_smart_contract_only_share_bps=5000,
    )
    blockchain_policy_path = root / "inputs/blockchain-policy.json"
    _write_model(blockchain_policy_path, blockchain_policy)

    documents = load_blockchain_security_documents(composed_corpus)
    audit = audit_blockchain_security_corpus(documents, holdout, blockchain_policy)
    assert audit.ready is True
    audit_path = root / "inputs/corpus-audit.json"
    write_blockchain_security_audit(audit, audit_path)

    release_policy = BlockchainSecurityReleasePolicy(
        policy_id="release-test",
        split_seed="release-test-seed",
        validation_target_bps=1000,
        test_target_bps=1000,
        train_min_bps=6000,
        required_chain_families=chains,
        required_threat_domains=threats,
        min_sources_per_required_chain_per_split=1,
        min_sources_per_required_threat_per_split=1,
    )
    release_policy_path = root / "inputs/release-policy.json"
    _write_model(release_policy_path, release_policy)

    return (
        {
            "catalog_path": catalog_path,
            "catalog_policy_path": catalog_policy_path,
            "source_catalog_manifest_path": source_manifest,
            "corpus_path": composed_corpus,
            "holdout_path": holdout_path,
            "blockchain_policy_path": blockchain_policy_path,
            "corpus_audit_path": audit_path,
            "release_policy_path": release_policy_path,
        },
        snapshots,
    )


def _build(root: Path, inputs: dict[str, Path], output: str):
    return build_blockchain_security_release(
        **inputs,
        output_dir=output,
        root=root,
    )


def test_release_is_source_isolated_deterministic_and_verifiable(tmp_path: Path) -> None:
    inputs, _ = _materialize_inputs(tmp_path)
    first = _build(tmp_path, inputs, "release-a")
    second = _build(tmp_path, inputs, "release-b")

    assert first == second
    assert first.documents == 9
    assert first.sources == 9
    assert first.source_leakage_detected is False
    assert first.family_leakage_detected is False
    source_sets = [set(first.splits[name].source_snapshot_digests) for name in ("train", "validation", "test")]
    assert source_sets[0].isdisjoint(source_sets[1])
    assert source_sets[0].isdisjoint(source_sets[2])
    assert source_sets[1].isdisjoint(source_sets[2])
    assert verify_blockchain_security_release(tmp_path / "release-a") == first


def test_shared_family_sources_are_kept_in_one_split(tmp_path: Path) -> None:
    inputs, snapshots = _materialize_inputs(tmp_path, shared_family=True)
    manifest = _build(tmp_path, inputs, "release")

    split_for_snapshot: dict[str, str] = {}
    for split_name, report in manifest.splits.items():
        for snapshot in report.source_snapshot_digests:
            split_for_snapshot[snapshot] = split_name
    assert split_for_snapshot[snapshots[0]] == split_for_snapshot[snapshots[1]]
    verify_blockchain_security_release(tmp_path / "release")


def test_stale_corpus_audit_blocks_release(tmp_path: Path) -> None:
    inputs, _ = _materialize_inputs(tmp_path)
    audit_path = inputs["corpus_audit_path"]
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["documents"] = 8
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="stale or does not match"):
        _build(tmp_path, inputs, "release")


def test_release_split_tampering_is_detected(tmp_path: Path) -> None:
    inputs, _ = _materialize_inputs(tmp_path)
    _build(tmp_path, inputs, "release")
    train = tmp_path / "release/train.jsonl"
    train.write_text(train.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch"):
        verify_blockchain_security_release(tmp_path / "release")


def test_release_directory_is_no_replace(tmp_path: Path) -> None:
    inputs, _ = _materialize_inputs(tmp_path)
    _build(tmp_path, inputs, "release")

    with pytest.raises(FileExistsError):
        _build(tmp_path, inputs, "release")


def test_repository_release_policy_is_fail_closed() -> None:
    root = Path(__file__).parents[1]
    payload = json.loads(
        (root / "configs/pretraining/blockchain-release.v1.json").read_text(encoding="utf-8")
    )
    policy = BlockchainSecurityReleasePolicy.model_validate(payload)
    assert policy.validation_target_bps >= 1000
    assert policy.test_target_bps >= 1000
    assert policy.train_min_bps >= 7500
    assert policy.require_source_isolation is True
    assert policy.require_family_isolation is True
    assert policy.min_sources_per_required_chain_per_split >= 1
    assert policy.min_sources_per_required_threat_per_split >= 1
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
