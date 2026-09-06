from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from koschei_sentinel.promotion import load_owner_public_key
from koschei_sentinel.web4_holdout_evaluation import Web4HoldoutEvaluationEvidence
from koschei_sentinel.web4_holdout_evaluation_decision import (
    Web4HoldoutEvaluationDecision,
    verify_web4_holdout_evaluation_decision,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify an owner-signed Web4 HOLDOUT research evaluation decision."
    )
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--owner-public-key", required=True)
    return parser


def _regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return candidate


def _load_model(path: str | Path, model_type, label: str):
    candidate = _regular_file(path, label)
    try:
        return model_type.model_validate_json(candidate.read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid {label}") from exc


def main() -> int:
    args = _parser().parse_args()
    try:
        evidence = _load_model(
            args.evidence,
            Web4HoldoutEvaluationEvidence,
            "Web4 HOLDOUT evaluation evidence",
        )
        decision = _load_model(
            args.decision,
            Web4HoldoutEvaluationDecision,
            "Web4 HOLDOUT evaluation decision",
        )
        owner_key_path = _regular_file(args.owner_public_key, "owner public key")
        owner_public_key = load_owner_public_key(owner_key_path)
        verified = verify_web4_holdout_evaluation_decision(
            evidence=evidence,
            decision=decision,
            owner_public_key=owner_public_key,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "schema_version": "sentinel.web4-holdout-evaluation-decision-verification.v1",
                "valid": True,
                "decision_id": verified.decision_id,
                "decision": verified.decision.value,
                "approver_id": verified.approver_id,
                "evidence_sha256": verified.evidence_sha256,
                "report_sha256": verified.report_sha256,
                "release_id": verified.release_id,
                "release_sha256": verified.release_sha256,
                "model_ref": verified.model_ref,
                "model_revision": verified.model_revision,
                "model_artifact_sha256": verified.model_artifact_sha256,
                "evaluation_passed": verified.evaluation_passed,
                "complete_case_accounting": verified.complete_case_accounting,
                "research_evaluation_evidence_accepted": (
                    verified.research_evaluation_evidence_accepted
                ),
                "research_comparison_eligible": verified.research_comparison_eligible,
                "research_evaluation_only": verified.research_evaluation_only,
                "model_execution_authorized": verified.model_execution_authorized,
                "training_authorization": verified.training_authorization,
                "promotion_eligible": verified.promotion_eligible,
                "production_activation_allowed": verified.production_activation_allowed,
                "owner_key_fingerprint": verified.owner_key_fingerprint,
                "owner_signature_verified": verified.owner_signature_verified,
                "decision_sha256": verified.decision_sha256,
                "artifact_sha256": verified.artifact_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
