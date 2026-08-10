from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from koschei_sentinel.curriculum import load_curriculum_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel-curriculum-check",
        description="Validate the fail-closed Sentinel language-first curriculum policy.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a sentinel.curriculum.v1 policy.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        policy = load_curriculum_policy(args.config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"curriculum rejected: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "ok": True,
                "schema": policy.schema_,
                "id": policy.id,
                "runtime_integration": policy.runtime_integration,
                "stages": [stage.id for stage in policy.stages],
                "language_gate": policy.stages[4].promotion_gate,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
