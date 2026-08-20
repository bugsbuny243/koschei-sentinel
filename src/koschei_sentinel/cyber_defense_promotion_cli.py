from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.cyber_defense_promotion import (
    build_cyber_defense_promotion_evidence_from_sources,
)
from koschei_sentinel.cyber_range_suite import CyberRangeSuiteReport
from koschei_sentinel.cyber_training_bundle import CyberTrainingBundle
from koschei_sentinel.defense_load_range import DefenseLoadRangeReport
from koschei_sentinel.gold_holdout_evaluation import GoldHoldoutEvaluationPolicy
from koschei_sentinel.gold_holdout_evaluation_evidence import (
    GoldHoldoutEvaluationEvidence,
)
from koschei_sentinel.gold_review_signing import load_reviewer_public_key
from koschei_sentinel.multi_incident_cyber_range_suite import (
    MultiIncidentCyberRangeSuiteReport,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reverify signed Gold HOLDOUT source artifacts and bind all required defense "
            "gates to one promotion receipt"
        )
    )
    parser.add_argument("--promotion-id", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--training-bundle", required=True)
    parser.add_argument("--cyber-range-report", required=True)
    parser.add_argument("--multi-incident-range-report", required=True)
    parser.add_argument("--defense-load-range-report", required=True)
    parser.add_argument("--gold-holdout-evidence", required=True)
    parser.add_argument("--gold-holdout-policy", required=True)
    parser.add_argument("--gold-release-dir", required=True)
    parser.add_argument("--gold-inference-pack", required=True)
    parser.add_argument("--gold-inference-output", required=True)
    parser.add_argument("--gold-candidate-export", required=True)
    parser.add_argument("--gold-reviewer-public-key", required=True)
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
        load = DefenseLoadRangeReport.model_validate_json(
            Path(args.defense_load_range_report).read_text(encoding="utf-8")
        )
        supplied_gold = GoldHoldoutEvaluationEvidence.model_validate_json(
            Path(args.gold_holdout_evidence).read_text(encoding="utf-8")
        )
        gold_policy = GoldHoldoutEvaluationPolicy.model_validate_json(
            Path(args.gold_holdout_policy).read_text(encoding="utf-8")
        )
        evidence = build_cyber_defense_promotion_evidence_from_sources(
            promotion_id=args.promotion_id,
            candidate_model_ref=args.candidate_model,
            candidate_model_revision=args.candidate_revision,
            training_bundle=bundle,
            cyber_range_report=single,
            multi_incident_range_report=multi,
            defense_load_range_report=load,
            supplied_gold_holdout_evidence=supplied_gold,
            gold_holdout_policy=gold_policy,
            gold_release_dir=args.gold_release_dir,
            gold_inference_pack_dir=args.gold_inference_pack,
            gold_inference_output_dir=args.gold_inference_output,
            gold_candidate_export_dir=args.gold_candidate_export,
            gold_reviewer_public_key=load_reviewer_public_key(
                args.gold_reviewer_public_key
            ),
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
