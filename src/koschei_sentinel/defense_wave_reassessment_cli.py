from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.assured_active_defense import ActiveDefenseAssurancePolicy
from koschei_sentinel.assured_multi_incident_defense import AssuredMultiIncidentDefensePlan
from koschei_sentinel.cyber_state_graph import CyberStateGraph
from koschei_sentinel.defense_resource_scheduler import DefenseResourceSchedule
from koschei_sentinel.defense_wave_execution import DefenseWaveExecution
from koschei_sentinel.defense_wave_reassessment import reassess_after_defense_wave
from koschei_sentinel.perception_assurance import PerceptionAssuranceSummary
from koschei_sentinel.perception_graph_binding import (
    BoundPerceptionGraph,
    PerceptionGraphReceipt,
)
from koschei_sentinel.perception_source_registry import PerceptionSourceRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Require fresh graph evidence and rebuild assured defense after a completed wave"
    )
    parser.add_argument("--wave", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--previous-plan", required=True)
    parser.add_argument("--next-graph", required=True)
    parser.add_argument("--next-graph-receipt", required=True)
    parser.add_argument("--next-assurance", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--critical", action="append", default=[])
    parser.add_argument("--assurance-policy")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        wave = DefenseWaveExecution.model_validate_json(
            Path(args.wave).read_text(encoding="utf-8")
        )
        schedule = DefenseResourceSchedule.model_validate_json(
            Path(args.schedule).read_text(encoding="utf-8")
        )
        previous_plan = AssuredMultiIncidentDefensePlan.model_validate_json(
            Path(args.previous_plan).read_text(encoding="utf-8")
        )
        graph = CyberStateGraph.model_validate_json(
            Path(args.next_graph).read_text(encoding="utf-8")
        )
        receipt = PerceptionGraphReceipt.model_validate_json(
            Path(args.next_graph_receipt).read_text(encoding="utf-8")
        )
        bound = BoundPerceptionGraph(graph=graph, receipt=receipt)
        assurance = PerceptionAssuranceSummary.model_validate_json(
            Path(args.next_assurance).read_text(encoding="utf-8")
        )
        registry = PerceptionSourceRegistry.model_validate_json(
            Path(args.registry).read_text(encoding="utf-8")
        )
        policy = (
            ActiveDefenseAssurancePolicy.model_validate_json(
                Path(args.assurance_policy).read_text(encoding="utf-8")
            )
            if args.assurance_policy
            else None
        )
        result = reassess_after_defense_wave(
            completed_wave=wave,
            previous_schedule=schedule,
            previous_plan=previous_plan,
            next_bound_graph=bound,
            next_perception_assurance=assurance,
            registry=registry,
            critical_entity_ids=list(dict.fromkeys(args.critical)),
            assurance_policy=policy,
        )
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        plan_path = output / "next-plan.json"
        receipt_path = output / "reassessment-receipt.json"
        plan_path.write_text(
            json.dumps(result.next_plan.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        receipt_path.write_text(
            json.dumps(result.receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "fresh_perception": result.receipt.fresh_perception,
                    "graph_changed": result.receipt.graph_changed,
                    "plan_changed": result.receipt.plan_changed,
                    "next_plan_sha256": result.receipt.next_plan_sha256,
                    "reassessment_sha256": result.receipt.reassessment_sha256,
                    "next_plan": str(plan_path),
                    "receipt": str(receipt_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-defense-reassess: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
