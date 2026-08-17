from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.assured_active_defense import ActiveDefenseAssurancePolicy
from koschei_sentinel.assured_multi_incident_defense import (
    build_assured_multi_incident_defense_plan,
)
from koschei_sentinel.cyber_state_graph import CyberStateGraph
from koschei_sentinel.perception_assurance import PerceptionAssuranceSummary
from koschei_sentinel.perception_graph_binding import PerceptionGraphReceipt
from koschei_sentinel.perception_source_registry import PerceptionSourceRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build independent assurance-gated defense plans for every active attack component"
        )
    )
    parser.add_argument("--graph", required=True)
    parser.add_argument("--graph-receipt", required=True)
    parser.add_argument("--perception-assurance", required=True)
    parser.add_argument("--source-registry", required=True)
    parser.add_argument("--policy", help="Optional active-defense assurance policy JSON")
    parser.add_argument(
        "--critical",
        action="append",
        default=[],
        help="Critical protected entity id; repeat as needed",
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        graph = CyberStateGraph.model_validate_json(Path(args.graph).read_text(encoding="utf-8"))
        graph_receipt = PerceptionGraphReceipt.model_validate_json(
            Path(args.graph_receipt).read_text(encoding="utf-8")
        )
        perception_assurance = PerceptionAssuranceSummary.model_validate_json(
            Path(args.perception_assurance).read_text(encoding="utf-8")
        )
        registry = PerceptionSourceRegistry.model_validate_json(
            Path(args.source_registry).read_text(encoding="utf-8")
        )
        policy = (
            ActiveDefenseAssurancePolicy.model_validate_json(
                Path(args.policy).read_text(encoding="utf-8")
            )
            if args.policy
            else ActiveDefenseAssurancePolicy()
        )
        plan = build_assured_multi_incident_defense_plan(
            graph,
            graph_receipt=graph_receipt,
            perception_assurance=perception_assurance,
            registry=registry,
            critical_entity_ids=list(dict.fromkeys(args.critical)),
            policy=policy,
        )
        payload = json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-assured-multi-defense: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
