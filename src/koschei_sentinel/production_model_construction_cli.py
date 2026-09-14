from __future__ import annotations

import argparse
import json

from koschei_sentinel.production_megatron_model_spec import load_production_megatron_model_spec
from koschei_sentinel.production_model_construction import build_construction_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile a dry-run-only 397B/35B construction and memory plan"
    )
    parser.add_argument("--model-spec", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec = load_production_megatron_model_spec(args.model_spec)
        plan = build_construction_plan(spec)
        print(json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-production-model-construction: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
