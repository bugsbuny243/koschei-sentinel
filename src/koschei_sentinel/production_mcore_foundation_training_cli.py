from __future__ import annotations

import argparse
import json

from koschei_sentinel.production_foundation_data import FoundationDataConfig
from koschei_sentinel.production_mcore_distributed_runtime import initialize_mcore_distributed_runtime
from koschei_sentinel.production_mcore_foundation_training import (
    FoundationTrainingConfig,
    run_foundation_training,
)
from koschei_sentinel.production_megatron_model_spec import load_production_megatron_model_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Sentinel 397B/35B foundation pretraining on packed JSONL text")
    parser.add_argument("--model-spec", required=True)
    parser.add_argument("--corpus-jsonl", required=True)
    parser.add_argument("--tokenizer-ref", required=True)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--eos-token-id", required=True, type=int)
    parser.add_argument("--pad-token-id", required=True, type=int)
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--shuffle-seed", type=int, default=39735)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--microbatches-per-step", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--global-seed", type=int, default=39735)
    parser.add_argument("--checkpoint-every-steps", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec = load_production_megatron_model_spec(args.model_spec)
        data_config = FoundationDataConfig(
            corpus_jsonl=args.corpus_jsonl,
            tokenizer_ref=args.tokenizer_ref,
            tokenizer_revision=args.tokenizer_revision,
            seq_length=args.seq_length,
            eos_token_id=args.eos_token_id,
            pad_token_id=args.pad_token_id,
            shuffle_seed=args.shuffle_seed,
        )
        train_config = FoundationTrainingConfig(
            max_steps=args.max_steps,
            microbatches_per_step=args.microbatches_per_step,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            clip_grad=args.clip_grad,
            global_seed=args.global_seed,
            checkpoint_every_steps=args.checkpoint_every_steps,
        )
        runtime = initialize_mcore_distributed_runtime(spec)
        result, sampler_state = run_foundation_training(
            runtime,
            spec,
            data_config,
            train_config,
            checkpoint_root=args.checkpoint_root,
            resume_checkpoint=args.resume_checkpoint,
        )
        print(json.dumps({"result": result.model_dump(mode="json"), "sampler_state": sampler_state.model_dump(mode="json")}, sort_keys=True))
        return 0
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"sentinel-production-foundation-train: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
