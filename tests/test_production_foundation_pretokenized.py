from __future__ import annotations

import json

import pytest

from koschei_sentinel.production_foundation_data import PackedFoundationSequence
from koschei_sentinel.production_foundation_pretokenized import (
    PretokenizedCursor,
    PretokenizedFoundationReader,
    build_pretokenized_shard,
    take_distributed_pretokenized_sequences,
)


def _sequence(base: int, length: int = 8) -> PackedFoundationSequence:
    return PackedFoundationSequence(
        input_ids=list(range(base, base + length)),
        labels=list(range(base + 1, base + length + 1)),
        loss_mask=[1.0] * length,
        position_ids=list(range(length)),
        document_ids=[f"doc-{base}"],
    )


def test_build_and_mmap_roundtrip(tmp_path):
    paths, manifest = build_pretokenized_shard(
        [_sequence(0), _sequence(100)],
        output_prefix=tmp_path / "foundation-00000",
        tokenizer_ref="fixture/tokenizer",
        tokenizer_revision="rev1",
        corpus_sha256="a" * 64,
    )
    assert manifest.sequence_count == 2
    assert manifest.sequence_length == 8
    assert manifest.token_count == 16
    assert manifest.packing_efficiency == 1.0
    with PretokenizedFoundationReader(paths.manifest) as reader:
        first = reader.read(0)
        second = reader.read(1)
        assert first.input_ids == list(range(8))
        assert second.input_ids == list(range(100, 108))
        assert len(reader.manifest_sha256) == 64


def test_reader_rejects_data_mutation(tmp_path):
    paths, _ = build_pretokenized_shard(
        [_sequence(0)],
        output_prefix=tmp_path / "foundation-00000",
        tokenizer_ref="fixture/tokenizer",
        tokenizer_revision=None,
        corpus_sha256="b" * 64,
    )
    paths.data.write_bytes(paths.data.read_bytes() + b"x")
    with pytest.raises(ValueError, match="data digest mismatch"):
        PretokenizedFoundationReader(paths.manifest)


def test_dp_ranks_receive_distinct_sequences_and_advance_global_cursor(tmp_path):
    paths, _ = build_pretokenized_shard(
        [_sequence(i * 100) for i in range(8)],
        output_prefix=tmp_path / "foundation-00000",
        tokenizer_ref="fixture/tokenizer",
        tokenizer_revision="rev1",
        corpus_sha256="c" * 64,
    )
    with PretokenizedFoundationReader(paths.manifest) as reader:
        rank0, state0 = take_distributed_pretokenized_sequences(
            reader, state=None, data_parallel_rank=0, data_parallel_size=2, count=2
        )
        rank1, state1 = take_distributed_pretokenized_sequences(
            reader, state=None, data_parallel_rank=1, data_parallel_size=2, count=2
        )
        assert [x.input_ids[0] for x in rank0] == [0, 200]
        assert [x.input_ids[0] for x in rank1] == [100, 300]
        assert state0.global_sequence_offset == state1.global_sequence_offset == 4

        resumed, next_state = take_distributed_pretokenized_sequences(
            reader, state=state0, data_parallel_rank=0, data_parallel_size=2, count=1
        )
        assert resumed[0].input_ids[0] == 400
        assert next_state.global_sequence_offset == 6


def test_cursor_rejects_manifest_or_dp_change(tmp_path):
    paths, _ = build_pretokenized_shard(
        [_sequence(i * 100) for i in range(4)],
        output_prefix=tmp_path / "foundation-00000",
        tokenizer_ref="fixture/tokenizer",
        tokenizer_revision="rev1",
        corpus_sha256="d" * 64,
    )
    with PretokenizedFoundationReader(paths.manifest) as reader:
        bad_manifest = PretokenizedCursor(
            global_sequence_offset=0,
            manifest_sha256="f" * 64,
            data_parallel_size=2,
        )
        with pytest.raises(ValueError, match="manifest mismatch"):
            take_distributed_pretokenized_sequences(
                reader, state=bad_manifest, data_parallel_rank=0, data_parallel_size=2, count=1
            )
        _, state = take_distributed_pretokenized_sequences(
            reader, state=None, data_parallel_rank=0, data_parallel_size=2, count=1
        )
        with pytest.raises(ValueError, match="data_parallel_size mismatch"):
            take_distributed_pretokenized_sequences(
                reader, state=state, data_parallel_rank=0, data_parallel_size=4, count=1
            )
