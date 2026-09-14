from __future__ import annotations

import argparse
import json

from koschei_sentinel.production_mcore_distributed_runtime import initialize_mcore_distributed_runtime
from koschei_sentinel.production_mcore_training_loop import (
    MCoreTrainingLoopConfig,
    run_mcore_resumable_training_loop,
)
from koschei_sentinel.production_megatron_model_spec import load_production_megatron_model_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bounded resumable Sentinel 397B/35B MCore systems-training loop"
    )
    parser.add_argument("--model-spec", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--microbatches-per-step", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--min-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--global-seed", type=int, default=39735)
    parser.add_argument("--checkpoint-every-steps", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec = load_production_megatron_model_spec(args.model_spec)
        config = MCoreTrainingLoopConfig(
            max_steps=args.max_steps,
            microbatches_per_step=args.microbatches_per_step,
            seq_length=args.seq_length,
            learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            clip_grad=args.clip_grad,
            global_seed=args.global_seed,
            checkpoint_every_steps=args.checkpoint_every_steps,
        )
        runtime = initialize_mcore_distributed_runtime(spec)
        result = run_mcore_resumable_training_loop(
            runtime,
            spec,
            config,
            checkpoint_root=args.checkpoint_root,
            resume_checkpoint=args.resume_checkpoint,
        )
        print(json.dumps(result.model_dump(mode="json"), sort_keys=True))
        return 0
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"sentinel-production-mcore-train-loop: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
