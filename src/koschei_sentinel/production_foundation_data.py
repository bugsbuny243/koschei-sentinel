from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel


class FoundationDataConfig(StrictModel):
    schema_version: Literal["sentinel.foundation-data-config.v1"] = "sentinel.foundation-data-config.v1"
    corpus_jsonl: str = Field(min_length=1)
    tokenizer_ref: str = Field(min_length=1)
    tokenizer_revision: str | None = None
    text_field: str = "text"
    id_field: str = "id"
    seq_length: int = Field(gt=1)
    eos_token_id: int = Field(ge=0)
    pad_token_id: int = Field(ge=0)
    shuffle_seed: int = Field(ge=0, lt=2**63)
    mask_cross_document_loss: bool = True


class FoundationCursor(StrictModel):
    schema_version: Literal["sentinel.foundation-cursor.v1"] = "sentinel.foundation-cursor.v1"
    epoch: int = Field(ge=0)
    packed_sequence_index: int = Field(ge=0)
    consumed_sequences: int = Field(ge=0)
    corpus_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    tokenizer_ref: str = Field(min_length=1)
    tokenizer_revision: str | None = None


class PackedFoundationSequence(StrictModel):
    input_ids: list[int] = Field(min_length=2)
    labels: list[int] = Field(min_length=2)
    loss_mask: list[float] = Field(min_length=2)
    position_ids: list[int] = Field(min_length=2)
    document_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def coherent(self) -> "PackedFoundationSequence":
        n = len(self.input_ids)
        if len(self.labels) != n or len(self.loss_mask) != n or len(self.position_ids) != n:
            raise ValueError("packed sequence fields must have equal length")
        return self


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_foundation_documents(config: FoundationDataConfig) -> tuple[list[tuple[str, str]], str]:
    path = Path(config.corpus_jsonl)
    if not path.is_file():
        raise ValueError(f"foundation corpus does not exist: {path}")
    documents: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            text = payload.get(config.text_field)
            if not isinstance(text, str) or not text.strip():
                continue
            raw_id = payload.get(config.id_field, line_number)
            documents.append((str(raw_id), text))
    if not documents:
        raise ValueError("foundation corpus contained no non-empty documents")
    return documents, _sha256_file(path)


def load_foundation_tokenizer(config: FoundationDataConfig) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("transformers is required for foundation tokenization") from exc
    return AutoTokenizer.from_pretrained(
        config.tokenizer_ref,
        revision=config.tokenizer_revision,
        trust_remote_code=False,
        use_fast=True,
    )


def _epoch_order(count: int, seed: int, epoch: int) -> list[int]:
    import random

    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    return order


def iter_packed_foundation_sequences(
    config: FoundationDataConfig,
    *,
    cursor: FoundationCursor | None = None,
) -> Iterator[tuple[PackedFoundationSequence, FoundationCursor]]:
    documents, corpus_sha256 = load_foundation_documents(config)
    tokenizer = load_foundation_tokenizer(config)
    if cursor is not None:
        if cursor.corpus_sha256 != corpus_sha256:
            raise ValueError("resume cursor corpus digest mismatch")
        if cursor.tokenizer_ref != config.tokenizer_ref or cursor.tokenizer_revision != config.tokenizer_revision:
            raise ValueError("resume cursor tokenizer identity mismatch")
        epoch = cursor.epoch
        skip_sequences = cursor.packed_sequence_index
        consumed = cursor.consumed_sequences
    else:
        epoch = 0
        skip_sequences = 0
        consumed = 0

    while True:
        order = _epoch_order(len(documents), config.shuffle_seed, epoch)
        tokens: list[int] = []
        loss_mask: list[float] = []
        doc_ids: list[str] = []
        produced = 0
        for index in order:
            doc_id, text = documents[index]
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if not encoded:
                continue
            start = len(tokens)
            tokens.extend(int(x) for x in encoded)
            tokens.append(config.eos_token_id)
            loss_mask.extend([1.0] * (len(encoded) + 1))
            if config.mask_cross_document_loss and start > 0:
                loss_mask[start - 1] = 0.0
            doc_ids.append(doc_id)

            while len(tokens) >= config.seq_length + 1:
                window = tokens[: config.seq_length + 1]
                mask_window = loss_mask[1 : config.seq_length + 1]
                packed = PackedFoundationSequence(
                    input_ids=window[:-1],
                    labels=window[1:],
                    loss_mask=mask_window,
                    position_ids=list(range(config.seq_length)),
                    document_ids=list(doc_ids),
                )
                tokens = tokens[config.seq_length:]
                loss_mask = loss_mask[config.seq_length:]
                if produced >= skip_sequences:
                    consumed += 1
                    next_cursor = FoundationCursor(
                        epoch=epoch,
                        packed_sequence_index=produced + 1,
                        consumed_sequences=consumed,
                        corpus_sha256=corpus_sha256,
                        tokenizer_ref=config.tokenizer_ref,
                        tokenizer_revision=config.tokenizer_revision,
                    )
                    yield packed, next_cursor
                produced += 1
        epoch += 1
        skip_sequences = 0
