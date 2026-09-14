from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from koschei_sentinel.production_foundation_catalog import (
    FoundationCatalogCursor,
    FoundationDatasetCatalog,
    load_foundation_catalog,
    locate_catalog_sequence,
)
from koschei_sentinel.production_foundation_data import PackedFoundationSequence
from koschei_sentinel.production_foundation_pretokenized import PretokenizedFoundationReader


class FoundationCatalogReader:
    """Bounded-open-reader view across many mmap pretokenized shards."""

    def __init__(self, catalog_path: str | Path, *, max_open_shards: int = 4):
        if max_open_shards <= 0:
            raise ValueError("max_open_shards must be positive")
        self.catalog: FoundationDatasetCatalog = load_foundation_catalog(catalog_path)
        self.max_open_shards = max_open_shards
        self._readers: OrderedDict[str, PretokenizedFoundationReader] = OrderedDict()

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __enter__(self) -> "FoundationCatalogReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _reader(self, manifest_path: str) -> PretokenizedFoundationReader:
        reader = self._readers.pop(manifest_path, None)
        if reader is None:
            reader = PretokenizedFoundationReader(manifest_path)
        self._readers[manifest_path] = reader
        while len(self._readers) > self.max_open_shards:
            _, evicted = self._readers.popitem(last=False)
            evicted.close()
        return reader

    def read_global(self, global_index: int) -> PackedFoundationSequence:
        shard, local_index = locate_catalog_sequence(self.catalog, global_index)
        reader = self._reader(shard.manifest_path)
        if reader.manifest_sha256 != shard.manifest_sha256:
            raise ValueError("catalog reader observed shard digest mismatch")
        return reader.read(local_index)


def take_distributed_catalog_sequences(
    reader: FoundationCatalogReader,
    *,
    state: FoundationCatalogCursor | None,
    data_parallel_rank: int,
    data_parallel_size: int,
    count: int,
) -> tuple[list[PackedFoundationSequence], FoundationCatalogCursor]:
    if not 0 <= data_parallel_rank < data_parallel_size:
        raise ValueError("data_parallel_rank outside data_parallel_size")
    if count <= 0:
        raise ValueError("count must be positive")
    catalog_digest = reader.catalog.catalog_sha256
    start = 0
    if state is not None:
        if state.catalog_sha256 != catalog_digest:
            raise ValueError("catalog cursor digest mismatch")
        if state.data_parallel_size != data_parallel_size:
            raise ValueError("catalog cursor data_parallel_size mismatch")
        start = state.global_sequence_offset

    selected: list[PackedFoundationSequence] = []
    stop = start + count * data_parallel_size
    for ordinal in range(start, stop):
        if (ordinal - start) % data_parallel_size == data_parallel_rank:
            selected.append(reader.read_global(ordinal))
    if len(selected) != count:
        raise RuntimeError("catalog sampler failed to produce requested local sequence count")
    return selected, FoundationCatalogCursor(
        global_sequence_offset=stop,
        catalog_sha256=catalog_digest,
        data_parallel_size=data_parallel_size,
    )
