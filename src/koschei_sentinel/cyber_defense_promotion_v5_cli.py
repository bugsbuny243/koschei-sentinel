from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.cyber_defense_promotion_v5 import (
    build_cyber_defense_promotion_v5_from_sources,
)
from koschei_sentinel.cyber_megatron_gold_evidence import (
    CyberMegatronGoldEvaluationEvidence,
)
from koschei_sentinel.cyber_range_suite import CyberRangeSuiteReport
from koschei_sentinel.cyber_training_bundle import CyberTrainingBundle
from koschei_sentinel.defense_load_range import DefenseLoadRangeReport
from koschei_sentinel.gold_holdout_evaluation import GoldHoldoutEvaluationPolicy
from koschei_sentinel.multi_incident_cyber_range_suite import (
    MultiIncidentCyberRangeSuiteReport,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild 397B Gold evidence from source artifacts and assemble "
            "Promotion v5 only when every binding matches"
        )
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--promotion-id", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--candidate-checkpoint-sha256", required=True)
    parser.add_argument("--training-bundle", required=True)
    parser.add_argument("--cyber-range-report", required=True)
    parser.add_argument("--multi-incident-range-report", required=True)
    parser.add_argument("--defense-load-range-report", required=True)
    parser.add_argument("--gold-evidence", required=True)
    parser.add_argument("--gold-policy", required=True)
    parser.add_argument("--gold-release", required=True)
    parser.add_argument("--gold-worker-dir", required=True)
    parser.add_argument("--gold-holdout-plan", required=True)
    parser.add_argument("--gold-candidate", required=True)
    parser.add_argument("--gold-config", required=True)
    parser.add_argument("--gold-training-plan", required=True)
    parser.add_argument("--gold-checkpoint", required=True)
    parser.add_argument("--gold-inference-pack", required=True)
    parser.add_argument("--gold-pack-signature", required=True)
    parser.add_argument("--gold-reviewer-public-key", required=True)
    parser.add_argument("--gold-reviewer-trust-policy", required=True)
    parser.add_argument("--gold-owner-public-key", required=True)
    parser.add_argument("--minimum-case-count", type=int, default=50)
    parser.add_argument("--output", required=True)
    return parser


def _load(path: str, model_type):
    return model_type.model_validate_json(Path(path).read_bytes())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        promotion = build_cyber_defense_promotion_v5_from_sources(
            promotion_id=args.promotion_id,
            candidate_model_ref=args.candidate_model,
            candidate_model_revision=args.candidate_revision,
            candidate_checkpoint_sha256=args.candidate_checkpoint_sha256,
            training_bundle=_load(args.training_bundle, CyberTrainingBundle),
            cyber_range_report=_load(args.cyber_range_report, CyberRangeSuiteReport),
            multi_incident_range_report=_load(
                args.multi_incident_range_report,
                MultiIncidentCyberRangeSuiteReport,
            ),
            defense_load_range_report=_load(
                args.defense_load_range_report,
                DefenseLoadRangeReport,
            ),
            supplied_gold_evidence=_load(
                args.gold_evidence,
                CyberMegatronGoldEvaluationEvidence,
            ),
            gold_policy=_load(args.gold_policy, GoldHoldoutEvaluationPolicy),
            gold_release_dir=args.gold_release,
            gold_worker_dir=args.gold_worker_dir,
            gold_holdout_plan_path=args.gold_holdout_plan,
            gold_candidate_manifest_path=args.gold_candidate,
            gold_candidate_config_path=args.gold_config,
            gold_candidate_training_plan_path=args.gold_training_plan,
            gold_checkpoint_dir=args.gold_checkpoint,
            gold_inference_pack=args.gold_inference_pack,
            gold_pack_signature=args.gold_pack_signature,
            gold_reviewer_public_key=args.gold_reviewer_public_key,
            gold_reviewer_trust_policy=args.gold_reviewer_trust_policy,
            gold_owner_public_key=args.gold_owner_public_key,
            minimum_case_count=args.minimum_case_count,
            root=args.root,
        )
        destination = Path(args.output)
        if destination.exists():
            raise FileExistsError(f"Promotion v5 output already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            promotion.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        ) + "\n"
        destination.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0 if promotion.ready_for_promotion else 1
    except (FileExistsError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-cyber-defense-promotion-v5: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
