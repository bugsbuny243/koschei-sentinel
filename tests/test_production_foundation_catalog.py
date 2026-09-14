from __future__ import annotations

import json

import pytest

from koschei_sentinel.production_foundation_catalog import (
    FoundationCatalogCursor,
    FoundationDatasetCatalog,
    FoundationCatalogShard,
    _catalog_digest_payload,
)
from koschei_sentinel.production_foundation_catalog_reader import take_distributed_catalog_sequences
from koschei_sentinel.production_foundation_worker_prefetch import FoundationWorkerPrefetch


def _catalog(tmp_path):
    shards = [
        FoundationCatalogShard(
            manifest_path=(tmp_path / "a.json").as_posix(),
            manifest_sha256="a" * 64,
            sequence_count=3,
            sequence_length=8,
        ),
        FoundationCatalogShard(
            manifest_path=(tmp_path / "b.json").as_posix(),
            manifest_sha256="b" * 64,
            sequence_count=5,
            sequence_length=8,
        ),
    ]
    payload = {
        "schema_version": "sentinel.foundation-dataset-catalog.v1",
        "shards": [item.model_dump(mode="json") for item in shards],
        "tokenizer_ref": "tokenizer",
        "tokenizer_revision": "rev",
        "sequence_length": 8,
        "total_sequences": 8,
    }
    return FoundationDatasetCatalog(**payload, catalog_sha256=_catalog_digest_payload(payload))


def test_catalog_rejects_mutated_digest(tmp_path):
    catalog = _catalog(tmp_path)
    payload = catalog.model_dump(mode="json")
    payload["total_sequences"] = 7
    with pytest.raises(ValueError):
        FoundationDatasetCatalog.model_validate(payload)


def test_catalog_cursor_binds_dp_size():
    cursor = FoundationCatalogCursor(
        global_sequence_offset=32,
        catalog_sha256="c" * 64,
        data_parallel_size=16,
    )
    assert cursor.global_sequence_offset == 32
    assert cursor.data_parallel_size == 16


def test_worker_prefetch_rejects_invalid_depth(tmp_path):
    with pytest.raises(ValueError):
        list(FoundationWorkerPrefetch(tmp_path / "catalog.json", [0, 1], workers=4, prefetch_depth=2))


def test_catalog_total_must_match_shards(tmp_path):
    catalog = _catalog(tmp_path)
    payload = catalog.model_dump(mode="json")
    payload["total_sequences"] = 9
    payload["catalog_sha256"] = _catalog_digest_payload({k: v for k, v in payload.items() if k != "catalog_sha256"})
    with pytest.raises(ValueError):
        FoundationDatasetCatalog.model_validate(payload)
