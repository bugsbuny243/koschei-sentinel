from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.cyber_megatron_gold_evidence import (
    build_cyber_megatron_gold_evidence,
)
from koschei_sentinel.gold_holdout_evaluation import GoldHoldoutEvaluationPolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reverify Qwen3.5-397B-A17B Gold HOLDOUT execution and materialize "
            "checkpoint-bound evaluation evidence"
        )
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--release", required=True)
    parser.add_argument("--worker-dir", required=True)
    parser.add_argument("--holdout-plan", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-plan", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inference-pack", required=True)
    parser.add_argument("--pack-signature", required=True)
    parser.add_argument("--reviewer-public-key", required=True)
    parser.add_argument("--reviewer-trust-policy", required=True)
    parser.add_argument("--owner-public-key", required=True)
    parser.add_argument("--evaluation-policy", required=True)
    parser.add_argument("--minimum-case-count", type=int, default=50)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        policy = GoldHoldoutEvaluationPolicy.model_validate_json(
            Path(args.evaluation_policy).read_bytes()
        )
        evidence = build_cyber_megatron_gold_evidence(
            root=args.root,
            release_dir=args.release,
            worker_dir=args.worker_dir,
            holdout_plan_path=args.holdout_plan,
            candidate_manifest_path=args.candidate,
            candidate_config_path=args.config,
            candidate_training_plan_path=args.training_plan,
            checkpoint_dir=args.checkpoint,
            inference_pack=args.inference_pack,
            signature_path=args.pack_signature,
            reviewer_public_key_path=args.reviewer_public_key,
            reviewer_trust_policy_path=args.reviewer_trust_policy,
            owner_public_key_path=args.owner_public_key,
            policy=policy,
            minimum_case_count=args.minimum_case_count,
        )
        destination = Path(args.output)
        if destination.exists():
            raise FileExistsError(f"397B Gold evidence already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            evidence.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        ) + "\n"
        destination.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0 if evidence.passed else 1
    except (FileExistsError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-cyber-megatron-gold-evidence: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
