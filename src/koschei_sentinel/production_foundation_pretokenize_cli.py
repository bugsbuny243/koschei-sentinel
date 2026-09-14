from __future__ import annotations

import argparse
import json

from koschei_sentinel.production_foundation_data import FoundationDataConfig, iter_packed_foundation_sequences
from koschei_sentinel.production_foundation_pretokenized import build_pretokenized_shard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize deterministic mmap-ready foundation token shards")
    parser.add_argument("--corpus-jsonl", required=True)
    parser.add_argument("--tokenizer-ref", required=True)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--sequence-count", type=int, required=True)
    parser.add_argument("--seq-length", type=int, required=True)
    parser.add_argument("--eos-token-id", type=int, required=True)
    parser.add_argument("--pad-token-id", type=int, required=True)
    parser.add_argument("--shuffle-seed", type=int, default=39735)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.sequence_count <= 0:
            raise ValueError("sequence-count must be positive")
        config = FoundationDataConfig(
            corpus_jsonl=args.corpus_jsonl,
            tokenizer_ref=args.tokenizer_ref,
            tokenizer_revision=args.tokenizer_revision,
            seq_length=args.seq_length,
            eos_token_id=args.eos_token_id,
            pad_token_id=args.pad_token_id,
            shuffle_seed=args.shuffle_seed,
            mask_cross_document_loss=True,
        )
        stream = iter_packed_foundation_sequences(config)
        sequences = []
        corpus_sha256 = None
        for _ in range(args.sequence_count):
            sequence, cursor = next(stream)
            sequences.append(sequence)
            corpus_sha256 = cursor.corpus_sha256
        if corpus_sha256 is None:
            raise RuntimeError("foundation stream produced no sequences")
        paths, manifest = build_pretokenized_shard(
            sequences,
            output_prefix=args.output_prefix,
            tokenizer_ref=args.tokenizer_ref,
            tokenizer_revision=args.tokenizer_revision,
            corpus_sha256=corpus_sha256,
        )
        print(json.dumps({
            "manifest": manifest.model_dump(mode="json"),
            "data_path": paths.data.as_posix(),
            "index_path": paths.index.as_posix(),
            "manifest_path": paths.manifest.as_posix(),
        }, sort_keys=True))
        return 0
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, StopIteration) as exc:
        print(f"sentinel-foundation-pretokenize: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
