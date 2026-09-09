from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from koschei_sentinel.promotion import load_owner_public_key
from koschei_sentinel.web4_holdout_evaluation import (
    Web4HoldoutEvaluationEvidence,
    Web4HoldoutEvaluationPolicy,
    Web4HoldoutInferencePack,
    Web4HoldoutPredictionSet,
    build_web4_holdout_evaluation_evidence,
)
from koschei_sentinel.web4_holdout_release import Web4HoldoutRelease


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild and verify offline Web4 HOLDOUT evaluation evidence against its signed "
            "release, isolated answer keys, and prediction artifacts."
        )
    )
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--inference-pack", required=True)
    parser.add_argument("--prediction-set", required=True)
    parser.add_argument("--answer-key-dir", required=True)
    parser.add_argument("--benchmark-policy", required=True)
    parser.add_argument("--owner-public-key", required=True)
    parser.add_argument("--evaluation-policy")
    return parser


def _regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return candidate


def _regular_dir(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError(f"{label} must be a regular non-symlink directory")
    return candidate


def _load_model(path: str | Path, model_type, label: str):
    candidate = _regular_file(path, label)
    try:
        return model_type.model_validate_json(candidate.read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid {label}") from exc


def _load_policy(path: str | Path | None) -> Web4HoldoutEvaluationPolicy:
    if path is None:
        return Web4HoldoutEvaluationPolicy()
    candidate = _regular_file(path, "Web4 HOLDOUT evaluation policy")
    try:
        return Web4HoldoutEvaluationPolicy.model_validate_json(candidate.read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError("invalid Web4 HOLDOUT evaluation policy") from exc


def main() -> int:
    args = _parser().parse_args()
    try:
        evidence = _load_model(
            args.evidence,
            Web4HoldoutEvaluationEvidence,
            "Web4 HOLDOUT evaluation evidence",
        )
        release = _load_model(args.release, Web4HoldoutRelease, "Web4 HOLDOUT release")
        inference_pack = _load_model(
            args.inference_pack,
            Web4HoldoutInferencePack,
            "Web4 HOLDOUT inference pack",
        )
        prediction_set = _load_model(
            args.prediction_set,
            Web4HoldoutPredictionSet,
            "Web4 HOLDOUT prediction set",
        )
        answer_key_dir = _regular_dir(args.answer_key_dir, "Web4 HOLDOUT answer-key directory")
        owner_public_key = load_owner_public_key(args.owner_public_key)
        policy = _load_policy(args.evaluation_policy)
        rebuilt = build_web4_holdout_evaluation_evidence(
            release=release,
            inference_pack=inference_pack,
            prediction_set=prediction_set,
            answer_key_dir=answer_key_dir,
            owner_public_key=owner_public_key,
            benchmark_policy_path=args.benchmark_policy,
            policy=policy,
        )
        if rebuilt.evidence_sha256 != evidence.evidence_sha256:
            raise ValueError("Web4 HOLDOUT evaluation evidence differs from offline rebuild")
    except (OSError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "schema_version": "sentinel.web4-holdout-evaluation-verification.v1",
                "valid": True,
                "release_id": rebuilt.release_id,
                "release_sha256": rebuilt.release_sha256,
                "inference_pack_sha256": rebuilt.inference_pack_sha256,
                "prediction_set_sha256": rebuilt.prediction_set_sha256,
                "answer_key_bundle_sha256": rebuilt.answer_key_bundle_sha256,
                "evaluation_policy_sha256": rebuilt.evaluation_policy_sha256,
                "model_ref": rebuilt.model_ref,
                "model_revision": rebuilt.model_revision,
                "model_artifact_sha256": rebuilt.model_artifact_sha256,
                "case_count": rebuilt.case_count,
                "prediction_count": rebuilt.prediction_count,
                "complete_case_accounting": rebuilt.complete_case_accounting,
                "answer_key_values_embedded": rebuilt.answer_key_values_embedded,
                "offline_replay": rebuilt.offline_replay,
                "network_access_required": rebuilt.network_access_required,
                "gpu_required": rebuilt.gpu_required,
                "research_evaluation_only": rebuilt.research_evaluation_only,
                "training_authorization": rebuilt.training_authorization,
                "promotion_eligible": rebuilt.promotion_eligible,
                "production_activation_allowed": rebuilt.production_activation_allowed,
                "passed": rebuilt.passed,
                "evidence_sha256": rebuilt.evidence_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
