from pathlib import Path

import pytest

import koschei_sentinel.production_foundation_data as foundation
from koschei_sentinel.production_foundation_data import (
    FoundationDataConfig,
    FoundationSamplerState,
    iter_packed_foundation_sequences,
    take_distributed_foundation_sequences,
)


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        assert add_special_tokens is False
        base = sum(ord(ch) for ch in text) % 50 + 1
        return [base, base + 1, base + 2, base + 3]


def _config(path: Path) -> FoundationDataConfig:
    return FoundationDataConfig(
        corpus_jsonl=path.as_posix(),
        tokenizer_ref="test-tokenizer",
        tokenizer_revision="unit-test",
        seq_length=4,
        eos_token_id=99,
        pad_token_id=0,
        shuffle_seed=39735,
        mask_cross_document_loss=True,
    )


def _write_corpus(path: Path) -> None:
    path.write_text(
        '\n'.join([
            '{"id":"a","text":"alpha"}',
            '{"id":"b","text":"beta"}',
            '{"id":"c","text":"gamma"}',
            '{"id":"d","text":"delta"}',
        ]) + '\n',
        encoding="utf-8",
    )


def test_packer_is_deterministic_and_masks_document_boundary(tmp_path, monkeypatch) -> None:
    corpus = tmp_path / "foundation.jsonl"
    _write_corpus(corpus)
    monkeypatch.setattr(foundation, "load_foundation_tokenizer", lambda config: FakeTokenizer())
    config = _config(corpus)
    left = iter_packed_foundation_sequences(config)
    right = iter_packed_foundation_sequences(config)
    left_items = [next(left)[0] for _ in range(3)]
    right_items = [next(right)[0] for _ in range(3)]
    assert [item.input_ids for item in left_items] == [item.input_ids for item in right_items]
    assert [item.labels for item in left_items] == [item.labels for item in right_items]
    assert all(len(item.input_ids) == config.seq_length for item in left_items)
    assert any(0.0 in item.loss_mask for item in left_items)


def test_data_parallel_ranks_receive_disjoint_global_slots(tmp_path, monkeypatch) -> None:
    corpus = tmp_path / "foundation.jsonl"
    _write_corpus(corpus)
    monkeypatch.setattr(foundation, "load_foundation_tokenizer", lambda config: FakeTokenizer())
    config = _config(corpus)
    rank0, state0 = take_distributed_foundation_sequences(
        config, state=None, data_parallel_rank=0, data_parallel_size=2, count=2
    )
    rank1, state1 = take_distributed_foundation_sequences(
        config, state=None, data_parallel_rank=1, data_parallel_size=2, count=2
    )
    assert state0.global_sequence_offset == state1.global_sequence_offset == 4
    assert [x.input_ids for x in rank0] != [x.input_ids for x in rank1]


def test_sampler_resume_rejects_changed_corpus(tmp_path, monkeypatch) -> None:
    corpus = tmp_path / "foundation.jsonl"
    _write_corpus(corpus)
    monkeypatch.setattr(foundation, "load_foundation_tokenizer", lambda config: FakeTokenizer())
    config = _config(corpus)
    _, state = take_distributed_foundation_sequences(
        config, state=None, data_parallel_rank=0, data_parallel_size=2, count=1
    )
    corpus.write_text(corpus.read_text(encoding="utf-8") + '{"id":"e","text":"epsilon"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="corpus digest mismatch"):
        take_distributed_foundation_sequences(
            config, state=state, data_parallel_rank=0, data_parallel_size=2, count=1
        )


def test_sampler_resume_rejects_parallelism_change(tmp_path, monkeypatch) -> None:
    corpus = tmp_path / "foundation.jsonl"
    _write_corpus(corpus)
    monkeypatch.setattr(foundation, "load_foundation_tokenizer", lambda config: FakeTokenizer())
    config = _config(corpus)
    _, state = take_distributed_foundation_sequences(
        config, state=None, data_parallel_rank=0, data_parallel_size=2, count=1
    )
    with pytest.raises(ValueError, match="data_parallel_size mismatch"):
        take_distributed_foundation_sequences(
            config, state=state, data_parallel_rank=0, data_parallel_size=4, count=1
        )
