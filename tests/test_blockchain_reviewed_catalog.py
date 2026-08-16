from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from koschei_sentinel.blockchain_reviewed_catalog import (
    ReviewedSourceCatalog,
    audit_reviewed_source_catalog,
    load_approved_blockchain_documents,
)
from koschei_sentinel.blockchain_source_catalog import (
    BlockchainSourceCatalogPolicy,
    write_catalog_outputs,
)


def _digest(payload: object) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _document(
    *,
    source_id: str,
    repository: str,
    commit: str,
    snapshot: str,
    path: str,
    text: str,
    source_class: str,
    chain: str,
    threats: list[str],
    license_spdx: str,
) -> dict[str, object]:
    return {
        "schema_version": "sentinel.approved-blockchain-document.v1",
        "source_id": source_id,
        "repository": repository,
        "commit": commit,
        "lineage_lane": "ORIGINAL",
        "path": path,
        "source_snapshot_digest": snapshot,
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_class": source_class,
        "chain_family": chain,
        "threat_domains": threats,
        "text": text,
        "license_spdx": license_spdx,
        "review_decision": "RETAIN",
        "review_reason": "reviewed security material",
    }


def _document_set_digest(rows: list[dict[str, object]]) -> str:
    payload = [
        {
            "path": row["path"],
            "content_sha256": row["content_sha256"],
            "source_class": row["source_class"],
            "chain_family": row["chain_family"],
            "threat_domains": row["threat_domains"],
        }
        for row in sorted(rows, key=lambda row: str(row["path"]))
    ]
    return _digest(payload)


def _write_fixture(tmp_path: Path) -> tuple[ReviewedSourceCatalog, BlockchainSourceCatalogPolicy]:
    alpha_rows = [
        _document(
            source_id="alpha.protocol",
            repository="example/alpha",
            commit="1" * 40,
            snapshot="a" * 64,
            path="docs/protocol.md",
            text="EVM protocol security boundary.\n",
            source_class="PROTOCOL_SOURCE",
            chain="EVM",
            threats=["SMART_CONTRACT"],
            license_spdx="MIT",
        ),
        _document(
            source_id="alpha.protocol",
            repository="example/alpha",
            commit="1" * 40,
            snapshot="a" * 64,
            path="spec/invariant.md",
            text="Normative privileged access invariant.\n",
            source_class="FORMAL_SPECIFICATION",
            chain="EVM",
            threats=["PRIVILEGED_ACCESS"],
            license_spdx="MIT",
        ),
    ]
    beta_rows = [
        _document(
            source_id="beta.audit",
            repository="example/beta",
            commit="2" * 40,
            snapshot="b" * 64,
            path="findings/high.md",
            text="Independent Solana privileged access audit finding.\n",
            source_class="AUDIT_FINDING",
            chain="SOLANA",
            threats=["PRIVILEGED_ACCESS"],
            license_spdx="Apache-2.0",
        )
    ]
    rows = sorted([*alpha_rows, *beta_rows], key=lambda row: (row["source_id"], row["path"]))
    corpus = tmp_path / "approved.jsonl"
    corpus.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    corpus_sha = hashlib.sha256(corpus.read_bytes()).hexdigest()

    payload: dict[str, object] = {
        "schema_version": "sentinel.reviewed-source-catalog.v1",
        "catalog_id": "reviewed-test",
        "parent_document_review_manifest_digest": "c" * 64,
        "parent_document_review_sha256": "d" * 64,
        "approved_document_corpus_file": corpus.name,
        "approved_document_corpus_sha256": corpus_sha,
        "sources": [
            {
                "source_id": "alpha.protocol",
                "repository": "example/alpha",
                "commit": "1" * 40,
                "lineage_lane": "ORIGINAL",
                "source_level_class": "PROTOCOL_SOURCE",
                "trust_tier": "PRIMARY_PROTOCOL",
                "snapshot_digest": "a" * 64,
                "license_spdx": "MIT",
                "license_evidence_path": "LICENSE",
                "license_evidence_sha256": "e" * 64,
                "retained_documents": 2,
                "retained_document_set_digest": _document_set_digest(alpha_rows),
                "reviewed_document_class_counts": {
                    "FORMAL_SPECIFICATION": 1,
                    "PROTOCOL_SOURCE": 1,
                },
                "reviewed_chain_families": ["EVM"],
                "reviewed_threat_domains": ["PRIVILEGED_ACCESS", "SMART_CONTRACT"],
            },
            {
                "source_id": "beta.audit",
                "repository": "example/beta",
                "commit": "2" * 40,
                "lineage_lane": "ORIGINAL",
                "source_level_class": "AUDIT_FINDING",
                "trust_tier": "INDEPENDENT_AUDIT",
                "snapshot_digest": "b" * 64,
                "license_spdx": "Apache-2.0",
                "license_evidence_path": "LICENSE",
                "license_evidence_sha256": "f" * 64,
                "retained_documents": 1,
                "retained_document_set_digest": _document_set_digest(beta_rows),
                "reviewed_document_class_counts": {"AUDIT_FINDING": 1},
                "reviewed_chain_families": ["SOLANA"],
                "reviewed_threat_domains": ["PRIVILEGED_ACCESS"],
            },
        ],
    }
    payload["catalog_digest"] = _digest(payload)
    catalog = ReviewedSourceCatalog.model_validate(payload)
    policy = BlockchainSourceCatalogPolicy.model_validate(
        {
            "policy_id": "reviewed-test",
            "min_sources": 2,
            "min_source_classes": 2,
            "required_chain_families": ["EVM", "SOLANA"],
            "required_threat_domains": ["SMART_CONTRACT", "PRIVILEGED_ACCESS"],
            "min_sources_per_required_chain": 1,
            "min_sources_per_required_threat_domain": 1,
            "min_high_trust_share_bps": 10000,
            "max_synthetic_source_share_bps": 0,
            "max_single_source_document_share_bps": 7000,
        }
    )
    return catalog, policy


def test_reviewed_catalog_preserves_document_level_class_overrides(tmp_path: Path) -> None:
    catalog, policy = _write_fixture(tmp_path)
    result = audit_reviewed_source_catalog(catalog, policy, root=tmp_path)

    assert result.manifest.ready is True
    assert result.manifest.sources == 2
    assert result.manifest.documents == 3
    assert result.manifest.source_class_counts == {
        "AUDIT_FINDING": 1,
        "PROTOCOL_SOURCE": 1,
    }
    assert result.manifest.document_class_counts == {
        "AUDIT_FINDING": 1,
        "FORMAL_SPECIFICATION": 1,
        "PROTOCOL_SOURCE": 1,
    }
    assert result.manifest.violations == []

    corpus = tmp_path / "composed.jsonl"
    manifest = tmp_path / "composed.manifest.json"
    write_catalog_outputs(result, corpus_path=corpus, manifest_path=manifest)
    stored = [json.loads(line) for line in corpus.read_text().splitlines()]
    assert {row["source_class"] for row in stored} == {
        "AUDIT_FINDING",
        "FORMAL_SPECIFICATION",
        "PROTOCOL_SOURCE",
    }


def test_approved_document_requires_nonempty_threat_mapping(tmp_path: Path) -> None:
    row = _document(
        source_id="alpha.protocol",
        repository="example/alpha",
        commit="1" * 40,
        snapshot="a" * 64,
        path="README.md",
        text="Generic overview.\n",
        source_class="PROTOCOL_SOURCE",
        chain="EVM",
        threats=[],
        license_spdx="MIT",
    )
    path = tmp_path / "approved.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid approved blockchain document row"):
        load_approved_blockchain_documents(path)


def test_reviewed_catalog_rejects_approved_corpus_drift(tmp_path: Path) -> None:
    catalog, policy = _write_fixture(tmp_path)
    corpus = tmp_path / catalog.approved_document_corpus_file
    corpus.write_text(corpus.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="approved document corpus digest"):
        audit_reviewed_source_catalog(catalog, policy, root=tmp_path)
