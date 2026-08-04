from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from koschei_sentinel.neon_source import build_neon_dataset_release
from koschei_sentinel.readiness import ReadinessPolicy, load_readiness_policy
from koschei_sentinel.split import SplitConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel-neon-build",
        description=(
            "Build a private Sentinel training release directly from read-only Neon data."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--salt-version", required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--seed", default="sentinel-split-v1")
    parser.add_argument("--train-bps", type=int, default=8000)
    parser.add_argument("--validation-bps", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dry_run and args.output_dir is None:
        raise SystemExit("--output-dir is required unless --dry-run is used")
    try:
        policy = load_readiness_policy(args.policy) if args.policy else ReadinessPolicy()
        result = build_neon_dataset_release(
            output_dir=args.output_dir,
            salt_version=args.salt_version,
            limit=args.limit,
            policy=policy,
            split_config=SplitConfig(
                seed=args.seed,
                train_bps=args.train_bps,
                validation_bps=args.validation_bps,
            ),
            dry_run=args.dry_run,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Neon dataset build rejected: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    if result.build.status == "rejected":
        return 2
    return 0 if result.build.status == "ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
