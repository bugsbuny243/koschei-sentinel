from __future__ import annotations

import hashlib
import json
import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_foundation_data import PackedFoundationSequence

_MAGIC = b"KSFNDAT1"
_HEADER = struct.Struct("<8sIIQ")
_INDEX = struct.Struct("<QQ")
_U32 = struct.Struct("<I")


class PretokenizedFoundationManifest(StrictModel):
    schema_version: Literal["sentinel.foundation-pretokenized.v1"] = "sentinel.foundation-pretokenized.v1"
    sequence_length: int = Field(gt=1)
    sequence_count: int = Field(gt=0)
    token_count: int = Field(gt=0)
    tokenizer_ref: str = Field(min_length=1)
    tokenizer_revision: str | None = None
    corpus_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    data_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    index_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    packing_efficiency: float = Field(gt=0.0, le=1.0)
    dtype: Literal["uint32"] = "uint32"


class PretokenizedCursor(StrictModel):
    schema_version: Literal["sentinel.foundation-pretokenized-cursor.v1"] = (
        "sentinel.foundation-pretokenized-cursor.v1"
    )
    global_sequence_offset: int = Field(ge=0)
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    data_parallel_size: int = Field(gt=0)


@dataclass(frozen=True)
class PretokenizedPaths:
    data: Path
    index: Path
    manifest: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_digest(manifest: PretokenizedFoundationManifest) -> str:
    payload = json.dumps(manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_pretokenized_shard(
    sequences: Iterable[PackedFoundationSequence],
    *,
    output_prefix: str | Path,
    tokenizer_ref: str,
    tokenizer_revision: str | None,
    corpus_sha256: str,
) -> tuple[PretokenizedPaths, PretokenizedFoundationManifest]:
    prefix = Path(output_prefix)
    paths = PretokenizedPaths(
        data=prefix.with_suffix(".bin"),
        index=prefix.with_suffix(".idx"),
        manifest=prefix.with_suffix(".json"),
    )
    for path in (paths.data, paths.index, paths.manifest):
        if path.exists():
            raise FileExistsError(f"pretokenized output already exists: {path}")
    paths.data.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    seq_length: int | None = None
    token_count = 0
    non_padding = 0
    offset = 0
    with paths.data.open("wb") as data_handle, paths.index.open("wb") as index_handle:
        for sequence in sequences:
            n = len(sequence.input_ids)
            if seq_length is None:
                seq_length = n
            elif n != seq_length:
                raise ValueError("all pretokenized sequences must have identical length")
            if not (len(sequence.labels) == len(sequence.loss_mask) == len(sequence.position_ids) == n):
                raise ValueError("sequence fields have inconsistent lengths")

            index_handle.write(_INDEX.pack(offset, n))
            fields = (sequence.input_ids, sequence.labels, sequence.position_ids)
            for values in fields:
                for value in values:
                    if value < 0 or value > 0xFFFFFFFF:
                        raise ValueError("token/position id outside uint32 range")
                    data_handle.write(_U32.pack(int(value)))
            mask_bytes = bytes(1 if value > 0 else 0 for value in sequence.loss_mask)
            data_handle.write(mask_bytes)
            record_bytes = (3 * n * _U32.size) + n
            offset += record_bytes
            count += 1
            token_count += n
            non_padding += sum(1 for value in sequence.loss_mask if value > 0)

    if count == 0 or seq_length is None:
        raise ValueError("cannot build an empty pretokenized shard")

    manifest = PretokenizedFoundationManifest(
        sequence_length=seq_length,
        sequence_count=count,
        token_count=token_count,
        tokenizer_ref=tokenizer_ref,
        tokenizer_revision=tokenizer_revision,
        corpus_sha256=corpus_sha256,
        data_sha256=_sha256_file(paths.data),
        index_sha256=_sha256_file(paths.index),
        packing_efficiency=non_padding / token_count,
    )
    paths.manifest.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths, manifest


class PretokenizedFoundationReader:
    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.manifest = PretokenizedFoundationManifest.model_validate_json(
            self.manifest_path.read_text(encoding="utf-8")
        )
        stem = self.manifest_path.with_suffix("")
        self.data_path = stem.with_suffix(".bin")
        self.index_path = stem.with_suffix(".idx")
        if _sha256_file(self.data_path) != self.manifest.data_sha256:
            raise ValueError("pretokenized data digest mismatch")
        if _sha256_file(self.index_path) != self.manifest.index_sha256:
            raise ValueError("pretokenized index digest mismatch")
        expected_index_size = self.manifest.sequence_count * _INDEX.size
        if self.index_path.stat().st_size != expected_index_size:
            raise ValueError("pretokenized index size mismatch")
        self._data_handle = self.data_path.open("rb")
        self._index_handle = self.index_path.open("rb")
        self._data_mmap = mmap.mmap(self._data_handle.fileno(), 0, access=mmap.ACCESS_READ)
        self._index_mmap = mmap.mmap(self._index_handle.fileno(), 0, access=mmap.ACCESS_READ)

    @property
    def manifest_sha256(self) -> str:
        return _manifest_digest(self.manifest)

    def close(self) -> None:
        self._data_mmap.close()
        self._index_mmap.close()
        self._data_handle.close()
        self._index_handle.close()

    def __enter__(self) -> "PretokenizedFoundationReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __len__(self) -> int:
        return self.manifest.sequence_count

    def read(self, sequence_index: int) -> PackedFoundationSequence:
        if not 0 <= sequence_index < len(self):
            raise IndexError("pretokenized sequence index out of range")
        offset, n = _INDEX.unpack_from(self._index_mmap, sequence_index * _INDEX.size)
        if n != self.manifest.sequence_length:
            raise ValueError("pretokenized sequence length mismatch")
        cursor = offset

        def read_u32(count: int) -> list[int]:
            nonlocal cursor
            size = count * _U32.size
            raw = self._data_mmap[cursor : cursor + size]
            cursor += size
            return list(struct.unpack(f"<{count}I", raw))

        input_ids = read_u32(n)
        labels = read_u32(n)
        position_ids = read_u32(n)
        loss_mask = [float(value) for value in self._data_mmap[cursor : cursor + n]]
        return PackedFoundationSequence(
            input_ids=input_ids,
            labels=labels,
            loss_mask=loss_mask,
            position_ids=position_ids,
            document_ids=[f"pretokenized:{sequence_index}"],
        )


def take_distributed_pretokenized_sequences(
    reader: PretokenizedFoundationReader,
    *,
    state: PretokenizedCursor | None,
    data_parallel_rank: int,
    data_parallel_size: int,
    count: int,
) -> tuple[list[PackedFoundationSequence], PretokenizedCursor]:
    if not 0 <= data_parallel_rank < data_parallel_size:
        raise ValueError("data_parallel_rank outside data_parallel_size")
    if count <= 0:
        raise ValueError("count must be positive")
    manifest_digest = reader.manifest_sha256
    start = 0
    if state is not None:
        if state.manifest_sha256 != manifest_digest:
            raise ValueError("pretokenized cursor manifest mismatch")
        if state.data_parallel_size != data_parallel_size:
            raise ValueError("pretokenized cursor data_parallel_size mismatch")
        start = state.global_sequence_offset

    selected: list[PackedFoundationSequence] = []
    stop = start + count * data_parallel_size
    total = len(reader)
    for ordinal in range(start, stop):
        if (ordinal - start) % data_parallel_size != data_parallel_rank:
            continue
        selected.append(reader.read(ordinal % total))
    return selected, PretokenizedCursor(
        global_sequence_offset=stop,
        manifest_sha256=manifest_digest,
        data_parallel_size=data_parallel_size,
    )
