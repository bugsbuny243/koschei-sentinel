from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.assured_active_defense import (
    ActiveDefenseAssurancePolicy,
    build_assured_active_defense_plan,
)
from koschei_sentinel.cyber_state_graph import CyberStateGraph
from koschei_sentinel.interception_execution import start_interception_execution
from koschei_sentinel.interception_planner import build_interception_plan
from koschei_sentinel.perception_assurance import PerceptionAssuranceSummary
from koschei_sentinel.perception_graph_binding import PerceptionGraphReceipt
from koschei_sentinel.perception_source_registry import PerceptionSourceRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a production Sentinel active-defense bundle with perception assurance gates"
    )
    parser.add_argument("--graph", required=True)
    parser.add_argument("--graph-receipt", required=True)
    parser.add_argument("--assurance-summary", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--policy", help="Optional active-defense assurance policy JSON")
    parser.add_argument(
        "--critical",
        action="append",
        default=[],
        help="Protected critical entity id; may be repeated",
    )
    parser.add_argument("--output", help="Optional assured defense bundle JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        graph = CyberStateGraph.model_validate_json(Path(args.graph).read_text(encoding="utf-8"))
        graph_receipt = PerceptionGraphReceipt.model_validate_json(
            Path(args.graph_receipt).read_text(encoding="utf-8")
        )
        assurance = PerceptionAssuranceSummary.model_validate_json(
            Path(args.assurance_summary).read_text(encoding="utf-8")
        )
        registry = PerceptionSourceRegistry.model_validate_json(
            Path(args.registry).read_text(encoding="utf-8")
        )
        policy = (
            ActiveDefenseAssurancePolicy.model_validate_json(
                Path(args.policy).read_text(encoding="utf-8")
            )
            if args.policy
            else ActiveDefenseAssurancePolicy()
        )

        assured = build_assured_active_defense_plan(
            graph,
            graph_receipt=graph_receipt,
            perception_assurance=assurance,
            registry=registry,
            critical_entity_ids=list(dict.fromkeys(args.critical)),
            policy=policy,
        )
        interception = build_interception_plan(assured.active_defense_plan)
        execution = start_interception_execution(interception)
        bundle = {
            "schema_version": "sentinel.assured-active-defense-bundle.v1",
            "graph_id": graph.graph_id,
            "active_defense_plan": assured.active_defense_plan.model_dump(mode="json"),
            "assurance": assured.assurance.model_dump(mode="json"),
            "assurance_policy": assured.policy.model_dump(mode="json"),
            "interception_plan": interception.model_dump(mode="json"),
            "execution_state": execution.model_dump(mode="json"),
        }
        payload = json.dumps(bundle, indent=2, sort_keys=True) + "\n"
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-assured-defense-plan: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
