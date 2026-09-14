from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from koschei_sentinel.production_foundation_catalog import load_foundation_catalog, locate_catalog_sequence
from koschei_sentinel.production_foundation_data import PackedFoundationSequence
from koschei_sentinel.production_foundation_pretokenized import PretokenizedFoundationReader


@dataclass(frozen=True)
class PrefetchedSequence:
    global_index: int
    sequence: PackedFoundationSequence


def _read_catalog_sequence(catalog_path: str, global_index: int) -> PrefetchedSequence:
    catalog = load_foundation_catalog(catalog_path)
    shard, local_index = locate_catalog_sequence(catalog, global_index)
    with PretokenizedFoundationReader(shard.manifest_path) as reader:
        if reader.manifest_sha256 != shard.manifest_sha256:
            raise ValueError("worker observed shard manifest digest mismatch")
        return PrefetchedSequence(global_index=global_index, sequence=reader.read(local_index))


class FoundationWorkerPrefetch:
    """Ordered bounded worker prefetch for mmap catalog reads.

    Results are yielded in requested order even when I/O workers complete out of order.
    """

    def __init__(
        self,
        catalog_path: str | Path,
        indices: list[int],
        *,
        workers: int = 2,
        prefetch_depth: int = 8,
    ):
        if workers <= 0:
            raise ValueError("workers must be positive")
        if prefetch_depth < workers:
            raise ValueError("prefetch_depth must be >= workers")
        self.catalog_path = Path(catalog_path).as_posix()
        self.indices = indices
        self.workers = workers
        self.prefetch_depth = prefetch_depth

    def __iter__(self) -> Iterator[PrefetchedSequence]:
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="foundation-prefetch") as pool:
            pending: dict[int, Future[PrefetchedSequence]] = {}
            submit_cursor = 0
            yield_cursor = 0
            while yield_cursor < len(self.indices):
                while submit_cursor < len(self.indices) and len(pending) < self.prefetch_depth:
                    index = self.indices[submit_cursor]
                    pending[submit_cursor] = pool.submit(_read_catalog_sequence, self.catalog_path, index)
                    submit_cursor += 1
                future = pending.pop(yield_cursor)
                item = future.result()
                expected_index = self.indices[yield_cursor]
                if item.global_index != expected_index:
                    raise RuntimeError("foundation prefetch returned out-of-order sequence identity")
                yield item
                yield_cursor += 1
