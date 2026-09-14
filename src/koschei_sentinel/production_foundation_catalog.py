from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_foundation_pretokenized import PretokenizedFoundationReader


class FoundationCatalogShard(StrictModel):
    manifest_path: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sequence_count: int = Field(gt=0)
    sequence_length: int = Field(gt=1)


class FoundationDatasetCatalog(StrictModel):
    schema_version: Literal["sentinel.foundation-dataset-catalog.v1"] = (
        "sentinel.foundation-dataset-catalog.v1"
    )
    shards: list[FoundationCatalogShard] = Field(min_length=1)
    tokenizer_ref: str = Field(min_length=1)
    tokenizer_revision: str | None = None
    sequence_length: int = Field(gt=1)
    total_sequences: int = Field(gt=0)
    catalog_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def coherent(self) -> "FoundationDatasetCatalog":
        if self.total_sequences != sum(shard.sequence_count for shard in self.shards):
            raise ValueError("catalog total_sequences mismatch")
        if any(shard.sequence_length != self.sequence_length for shard in self.shards):
            raise ValueError("catalog shards must share sequence_length")
        expected = _catalog_digest_payload(self.model_dump(mode="json", exclude={"catalog_sha256"}))
        if expected != self.catalog_sha256:
            raise ValueError("catalog digest mismatch")
        return self


class FoundationCatalogCursor(StrictModel):
    schema_version: Literal["sentinel.foundation-catalog-cursor.v1"] = (
        "sentinel.foundation-catalog-cursor.v1"
    )
    global_sequence_offset: int = Field(ge=0)
    catalog_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    data_parallel_size: int = Field(gt=0)


def _catalog_digest_payload(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_foundation_catalog(manifest_paths: list[str | Path], *, output_path: str | Path) -> FoundationDatasetCatalog:
    if not manifest_paths:
        raise ValueError("at least one pretokenized shard manifest is required")
    shards: list[FoundationCatalogShard] = []
    tokenizer_ref: str | None = None
    tokenizer_revision: str | None = None
    sequence_length: int | None = None
    total_sequences = 0
    for raw_path in manifest_paths:
        path = Path(raw_path)
        with PretokenizedFoundationReader(path) as reader:
            manifest = reader.manifest
            if tokenizer_ref is None:
                tokenizer_ref = manifest.tokenizer_ref
                tokenizer_revision = manifest.tokenizer_revision
                sequence_length = manifest.sequence_length
            elif (
                manifest.tokenizer_ref != tokenizer_ref
                or manifest.tokenizer_revision != tokenizer_revision
                or manifest.sequence_length != sequence_length
            ):
                raise ValueError("catalog shards must share tokenizer identity and sequence length")
            shards.append(
                FoundationCatalogShard(
                    manifest_path=path.as_posix(),
                    manifest_sha256=reader.manifest_sha256,
                    sequence_count=manifest.sequence_count,
                    sequence_length=manifest.sequence_length,
                )
            )
            total_sequences += manifest.sequence_count
    assert tokenizer_ref is not None and sequence_length is not None
    payload = {
        "schema_version": "sentinel.foundation-dataset-catalog.v1",
        "shards": [shard.model_dump(mode="json") for shard in shards],
        "tokenizer_ref": tokenizer_ref,
        "tokenizer_revision": tokenizer_revision,
        "sequence_length": sequence_length,
        "total_sequences": total_sequences,
    }
    catalog = FoundationDatasetCatalog(**payload, catalog_sha256=_catalog_digest_payload(payload))
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"catalog already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(catalog.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return catalog


def load_foundation_catalog(path: str | Path) -> FoundationDatasetCatalog:
    catalog = FoundationDatasetCatalog.model_validate_json(Path(path).read_text(encoding="utf-8"))
    for shard in catalog.shards:
        with PretokenizedFoundationReader(shard.manifest_path) as reader:
            if reader.manifest_sha256 != shard.manifest_sha256:
                raise ValueError("catalog shard manifest digest mismatch")
    return catalog


def locate_catalog_sequence(catalog: FoundationDatasetCatalog, global_index: int) -> tuple[FoundationCatalogShard, int]:
    if global_index < 0:
        raise ValueError("global_index cannot be negative")
    local = global_index % catalog.total_sequences
    for shard in catalog.shards:
        if local < shard.sequence_count:
            return shard, local
        local -= shard.sequence_count
    raise RuntimeError("catalog sequence lookup failed")
