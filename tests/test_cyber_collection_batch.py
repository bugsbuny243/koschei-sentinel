from __future__ import annotations

import hashlib
import json

from koschei_sentinel.cyber_collection_batch import CollectionBatchSpec, seal_collection_batch


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _source(source_id="official.source"):
    return {
        "schema_version": "sentinel.cyber-source.v3",
        "source_id": source_id,
        "source_class": "OFFICIAL_SECURITY_GUIDANCE",
        "domain_families": ["INCIDENT_RESPONSE"],
        "provenance_tier": "T1_AUTHORITATIVE",
        "license_status": "ALLOW_TRAINING",
        "license_scope": "UNIFORM",
        "license_reference": "license://approved",
        "acquisition_mode": "PINNED_GIT",
        "canonical_locator": "https://example.invalid/source",
        "pinned_revision": "abc123",
        "training_authorization": True,
        "eval_exclusion": True,
        "benchmark_overlap_risk": "LOW",
        "review_status": "APPROVED",
    }


def _artifact(text="defensive security guidance", source_id="official.source"):
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {
        "schema_version": "sentinel.cyber-artifact.v3",
        "artifact_id": f"{source_id}:doc-1",
        "source_id": source_id,
        "locator": "docs/security.md",
        "source_revision": "abc123",
        "source_snapshot_sha256": "a" * 64,
        "content_sha256": digest,
        "license_status": "ALLOW_TRAINING",
        "license_reference": "license://approved",
        "inherited_source_license": True,
        "training_authorization": True,
        "eval_exclusion": True,
        "benchmark_overlap_risk": "LOW",
    }, {
        "artifact_id": f"{source_id}:doc-1",
        "content_sha256": digest,
        "text": text,
        "metadata": {"provider": "TEST"},
    }


def test_collection_batch_seals_authorized_artifacts(tmp_path):
    catalog = tmp_path / "catalog.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(catalog, [_source()])
    artifact, row = _artifact()
    _write_jsonl(manifest, [artifact])
    _write_jsonl(corpus, [row])

    spec = CollectionBatchSpec(
        batch_id="batch-0001",
        approved_catalog_path=str(catalog),
        inputs=[
            {
                "source_id": "official.source",
                "manifest_path": str(manifest),
                "corpus_path": str(corpus),
            }
        ],
    )
    seal = seal_collection_batch(spec, output_dir=tmp_path / "sealed")
    assert seal.ready_for_training_pipeline is True
    assert seal.training_artifacts == 1
    assert seal.rejected_artifacts == 0
    assert seal.violations == []


def test_collection_batch_rejects_cross_source_duplicate_content(tmp_path):
    catalog = tmp_path / "catalog.jsonl"
    _write_jsonl(catalog, [_source("source.one"), _source("source.two")])
    inputs = []
    for source_id in ("source.one", "source.two"):
        manifest = tmp_path / f"{source_id}.manifest.jsonl"
        corpus = tmp_path / f"{source_id}.corpus.jsonl"
        artifact, row = _artifact(source_id=source_id)
        _write_jsonl(manifest, [artifact])
        _write_jsonl(corpus, [row])
        inputs.append(
            {
                "source_id": source_id,
                "manifest_path": str(manifest),
                "corpus_path": str(corpus),
            }
        )

    seal = seal_collection_batch(
        CollectionBatchSpec(
            batch_id="batch-duplicate",
            approved_catalog_path=str(catalog),
            inputs=inputs,
        ),
        output_dir=tmp_path / "sealed",
    )
    assert seal.ready_for_training_pipeline is False
    assert any("duplicate content sha256" in item for item in seal.violations)


def test_collection_batch_rejects_manifest_corpus_hash_mismatch(tmp_path):
    catalog = tmp_path / "catalog.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(catalog, [_source()])
    artifact, row = _artifact()
    row["content_sha256"] = "f" * 64
    _write_jsonl(manifest, [artifact])
    _write_jsonl(corpus, [row])

    spec = CollectionBatchSpec(
        batch_id="batch-mismatch",
        approved_catalog_path=str(catalog),
        inputs=[
            {
                "source_id": "official.source",
                "manifest_path": str(manifest),
                "corpus_path": str(corpus),
            }
        ],
    )
    try:
        seal_collection_batch(spec, output_dir=tmp_path / "sealed")
    except ValueError as exc:
        assert "corpus text hash mismatch" in str(exc)
    else:
        raise AssertionError("hash mismatch must fail closed")
