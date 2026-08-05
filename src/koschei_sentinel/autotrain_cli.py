from __future__ import annotations

import argparse
import json

from koschei_sentinel.autotrain import plan_autotrain, write_autotrain_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a Koschei Sentinel training release and produce an "
            "incubation-only autotrain plan"
        )
    )
    parser.add_argument(
        "--policy",
        required=True,
        help="Path to sentinel.autotrain-policy.v1 JSON",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to sentinel.training-config.v1 JSON",
    )
    parser.add_argument(
        "--output",
        help="Optional path for sentinel.autotrain-plan.v1 JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = plan_autotrain(args.policy, args.config)
        if args.output:
            write_autotrain_plan(plan, args.output)
        print(json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0 if plan.decision == "ready_for_offline_training" else 3
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"sentinel-autotrain: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
