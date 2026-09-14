from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.production_checkpoint_manifest import build_rank_checkpoint_manifest
from koschei_sentinel.production_megatron_model_spec import load_production_megatron_model_spec
from koschei_sentinel.training import atomic_write


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile a deterministic rank-local initialization/checkpoint manifest for the 397B/35B production candidate"
    )
    parser.add_argument("--model-spec", required=True)
    parser.add_argument("--pipeline-rank", type=int, required=True)
    parser.add_argument("--tensor-rank", type=int, required=True)
    parser.add_argument("--expert-rank", type=int, required=True)
    parser.add_argument("--global-seed", type=int, default=39735)
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec = load_production_megatron_model_spec(args.model_spec)
        manifest = build_rank_checkpoint_manifest(
            spec,
            pipeline_rank=args.pipeline_rank,
            tensor_rank=args.tensor_rank,
            expert_rank=args.expert_rank,
            global_seed=args.global_seed,
        )
        payload = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        if args.output:
            atomic_write(Path(args.output), payload)
        print(payload, end="")
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"sentinel-production-checkpoint-manifest: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
