from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.cyber_defense_promotion import (
    build_cyber_defense_promotion_evidence,
)
from koschei_sentinel.cyber_range_suite import CyberRangeSuiteReport
from koschei_sentinel.cyber_training_bundle import CyberTrainingBundle
from koschei_sentinel.multi_incident_cyber_range_suite import (
    MultiIncidentCyberRangeSuiteReport,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bind Sentinel training and cyber-range evidence to one promotion receipt"
    )
    parser.add_argument("--promotion-id", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--training-bundle", required=True)
    parser.add_argument("--cyber-range-report", required=True)
    parser.add_argument("--multi-incident-range-report", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        bundle = CyberTrainingBundle.model_validate_json(
            Path(args.training_bundle).read_text(encoding="utf-8")
        )
        single = CyberRangeSuiteReport.model_validate_json(
            Path(args.cyber_range_report).read_text(encoding="utf-8")
        )
        multi = MultiIncidentCyberRangeSuiteReport.model_validate_json(
            Path(args.multi_incident_range_report).read_text(encoding="utf-8")
        )
        evidence = build_cyber_defense_promotion_evidence(
            promotion_id=args.promotion_id,
            candidate_model_ref=args.candidate_model,
            candidate_model_revision=args.candidate_revision,
            training_bundle=bundle,
            cyber_range_report=single,
            multi_incident_range_report=multi,
        )
        payload = json.dumps(
            evidence.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        ) + "\n"
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0 if evidence.ready_for_promotion else 1
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-cyber-defense-promotion: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
