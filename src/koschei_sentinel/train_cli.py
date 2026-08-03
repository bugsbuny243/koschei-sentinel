from __future__ import annotations

import argparse
import json

from koschei_sentinel.trainer import execute_training
from koschei_sentinel.training import (
    load_training_config,
    plan_training,
    write_training_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or execute Koschei Sentinel QLoRA training"
    )
    parser.add_argument(
        "--config", required=True, help="Path to sentinel.training-config.v1 JSON"
    )
    parser.add_argument(
        "--plan-output", help="Optional path for the deterministic training plan"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Load the pinned base model and run training; default is a network-free dry run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_training_config(args.config)
        plan = plan_training(config)
        if args.plan_output:
            write_training_plan(plan, args.plan_output)
        result = execute_training(config, plan) if args.execute else plan
        print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"sentinel-train: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
