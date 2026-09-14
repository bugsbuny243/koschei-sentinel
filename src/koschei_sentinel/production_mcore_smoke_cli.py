from __future__ import annotations

import argparse
import json

from koschei_sentinel.production_mcore_distributed_runtime import initialize_mcore_distributed_runtime
from koschei_sentinel.production_mcore_smoke import run_mcore_forward_backward_smoke
from koschei_sentinel.production_megatron_model_spec import load_production_megatron_model_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one real distributed Megatron-Core forward/backward smoke microbatch"
    )
    parser.add_argument("--model-spec", required=True)
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=39735)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec = load_production_megatron_model_spec(args.model_spec)
        runtime = initialize_mcore_distributed_runtime(spec)
        result = run_mcore_forward_backward_smoke(
            runtime,
            spec,
            seq_length=args.seq_length,
            global_seed=args.global_seed,
        )
        print(json.dumps(result.model_dump(mode="json"), sort_keys=True))
        return 0
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"sentinel-production-mcore-smoke: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
