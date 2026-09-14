from __future__ import annotations

import argparse
import json

from koschei_sentinel.production_mcore_distributed_runtime import initialize_mcore_distributed_runtime
from koschei_sentinel.production_mcore_optimizer_smoke import run_mcore_optimizer_smoke
from koschei_sentinel.production_megatron_model_spec import load_production_megatron_model_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one real distributed MCore optimizer update and checkpoint roundtrip"
    )
    parser.add_argument("--model-spec", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=39735)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec = load_production_megatron_model_spec(args.model_spec)
        runtime = initialize_mcore_distributed_runtime(spec)
        result = run_mcore_optimizer_smoke(
            runtime,
            spec,
            seq_length=args.seq_length,
            global_seed=args.global_seed,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            clip_grad=args.clip_grad,
            checkpoint_dir=args.checkpoint_dir,
        )
        print(json.dumps(result.model_dump(mode="json"), sort_keys=True))
        return 0
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"sentinel-production-mcore-optimizer-smoke: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
