from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from koschei_sentinel.promotion import load_owner_public_key
from koschei_sentinel.web4_holdout_release import (
    Web4HoldoutRelease,
    verify_web4_holdout_release,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify an owner-signed Web4 research HOLDOUT release."
    )
    parser.add_argument("--release", required=True)
    parser.add_argument("--benchmark-policy", required=True)
    parser.add_argument("--owner-public-key", required=True)
    return parser


def _load_release(path: str | Path) -> Web4HoldoutRelease:
    release_path = Path(path)
    if release_path.is_symlink() or not release_path.is_file():
        raise ValueError("Web4 HOLDOUT release must be a regular non-symlink file")
    try:
        return Web4HoldoutRelease.model_validate_json(release_path.read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError("invalid Web4 HOLDOUT release") from exc


def main() -> int:
    args = _parser().parse_args()
    try:
        release = _load_release(args.release)
        owner_public_key = load_owner_public_key(args.owner_public_key)
        verified = verify_web4_holdout_release(
            release,
            owner_public_key=owner_public_key,
            benchmark_policy_path=args.benchmark_policy,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "schema_version": "sentinel.web4-holdout-release-verification.v1",
                "valid": True,
                "release_id": verified.release_id,
                "state": verified.state,
                "release_sha256": verified.release_sha256,
                "artifact_sha256": verified.artifact_sha256,
                "owner_key_fingerprint": verified.owner_key_fingerprint,
                "benchmark_policy_sha256": verified.benchmark_policy_sha256,
                "source_registry_sha256": verified.source_registry_sha256,
                "case_count": verified.case_count,
                "case_ids": verified.case_ids,
                "answer_keys_isolated": verified.answer_keys_isolated,
                "all_cases_human_reviewed": verified.all_cases_human_reviewed,
                "all_cases_independently_adjudicated": (
                    verified.all_cases_independently_adjudicated
                ),
                "deterministic_holdout_only": verified.deterministic_holdout_only,
                "research_evaluation_authorization": (
                    verified.research_evaluation_authorization
                ),
                "training_authorization": verified.training_authorization,
                "promotion_eligible": verified.promotion_eligible,
                "production_activation_allowed": verified.production_activation_allowed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
