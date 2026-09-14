from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.production_global_checkpoint import build_global_checkpoint_index
from koschei_sentinel.production_megatron_model_spec import load_production_megatron_model_spec
from koschei_sentinel.training import atomic_write


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the canonical PP/TP/EP global checkpoint index")
    parser.add_argument("--model-spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--global-seed", type=int, default=39735)
    parser.add_argument("--shard-dir", default="model-shards")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec = load_production_megatron_model_spec(args.model_spec)
        index = build_global_checkpoint_index(
            spec,
            global_seed=args.global_seed,
            shard_dir=args.shard_dir,
        )
        payload = json.dumps(index.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        atomic_write(Path(args.output), payload)
        print(payload, end="")
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"sentinel-production-global-checkpoint: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
