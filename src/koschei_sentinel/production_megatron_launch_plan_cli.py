from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.production_megatron_launch_plan import compile_production_megatron_launch_plan
from koschei_sentinel.production_megatron_model_spec import load_production_megatron_model_spec
from koschei_sentinel.training import atomic_write


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile a dry-run-only Megatron launch plan for the 397B/35B Sentinel research candidate")
    parser.add_argument("--model-spec", required=True)
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec = load_production_megatron_model_spec(args.model_spec)
        plan = compile_production_megatron_launch_plan(spec)
        payload = json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        if args.output:
            atomic_write(Path(args.output), payload)
        print(payload, end="")
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-production-megatron-plan: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
