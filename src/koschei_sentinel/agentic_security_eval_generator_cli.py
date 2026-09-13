from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.agentic_security_eval_generator import generate_from_paths, write_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic unreviewed Agentic Security benchmark candidates"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        bundle = generate_from_paths(Path(args.config), Path(args.seeds))
        write_bundle(bundle, Path(args.output))
        summary = {
            "schema_version": bundle.schema_version,
            "total_cases": bundle.total_cases,
            "cases_per_family": bundle.cases_per_family,
            "review_status": bundle.review_status,
            "gold_eligible": bundle.gold_eligible,
            "output": args.output,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-agentic-security-generate: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
