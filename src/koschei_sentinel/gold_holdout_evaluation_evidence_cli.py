from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.gold_holdout_evaluation import GoldHoldoutEvaluationPolicy
from koschei_sentinel.gold_holdout_evaluation_evidence import (
    build_owner_trusted_gold_holdout_evaluation_evidence,
)
from koschei_sentinel.gold_holdout_pack_signing import load_gold_holdout_pack_signature
from koschei_sentinel.gold_review_signing import load_reviewer_public_key
from koschei_sentinel.gold_reviewer_trust import load_gold_reviewer_trust_policy
from koschei_sentinel.promotion import load_owner_public_key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bind a verified answer-key-isolated Gold HOLDOUT inference run to its "
            "signed human-review release, candidate export, evaluation report, and policy"
        )
    )
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--inference-pack", required=True)
    parser.add_argument("--inference-pack-signature", required=True)
    parser.add_argument("--inference-output", required=True)
    parser.add_argument("--candidate-export", required=True)
    parser.add_argument("--reviewer-public-key", required=True)
    parser.add_argument("--reviewer-trust-policy", required=True)
    parser.add_argument("--owner-public-key", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _load_policy(path: str) -> GoldHoldoutEvaluationPolicy:
    try:
        return GoldHoldoutEvaluationPolicy.model_validate_json(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid Gold HOLDOUT evaluation policy: {path}") from exc


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        evidence = build_owner_trusted_gold_holdout_evaluation_evidence(
            release_dir=args.release_dir,
            inference_pack_dir=args.inference_pack,
            inference_output_dir=args.inference_output,
            candidate_export_dir=args.candidate_export,
            policy=_load_policy(args.policy),
            reviewer_public_key=load_reviewer_public_key(args.reviewer_public_key),
            reviewer_trust_policy=load_gold_reviewer_trust_policy(
                args.reviewer_trust_policy
            ),
            owner_public_key=load_owner_public_key(args.owner_public_key),
            inference_pack_signature_proof=load_gold_holdout_pack_signature(
                args.inference_pack_signature
            ),
        )
        payload = json.dumps(evidence.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0 if evidence.passed else 1
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-gold-holdout-evidence: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
