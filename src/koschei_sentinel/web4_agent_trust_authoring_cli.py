from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from koschei_sentinel.web4_agent_trust_authoring import (
    evaluate_web4_agent_trust_authoring_plan,
    write_web4_agent_trust_authoring_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel-web4-agent-trust-authoring",
        description=(
            "Validate the non-authorizing Web4 Agent Trust Chain authoring plan, "
            "priority sources, source-review gates, and deterministic HOLDOUT planning math."
        ),
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("configs/corpus/web4-v1.sources.proposed.jsonl"),
    )
    parser.add_argument(
        "--trust-chain-map",
        type=Path,
        default=Path("evals/web4-agent-trust-chain.v1.json"),
    )
    parser.add_argument(
        "--intake-policy",
        type=Path,
        default=Path("evals/web4-benchmark-intake-policy.v1.json"),
    )
    parser.add_argument(
        "--authoring-plan",
        type=Path,
        default=Path("evals/web4-agent-trust-authoring-plan.v1.json"),
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_web4_agent_trust_authoring_plan(
            source_registry_path=args.sources,
            trust_chain_map_path=args.trust_chain_map,
            intake_policy_path=args.intake_policy,
            authoring_plan_path=args.authoring_plan,
        )
        if args.output is not None:
            write_web4_agent_trust_authoring_report(report, args.output)
    except (OSError, ValueError) as exc:
        print(f"Web4 Agent Trust authoring plan rejected: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if report.plan_valid else 3


if __name__ == "__main__":
    raise SystemExit(main())
