from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.defense_reflex_candidates import DefenseReflexCandidate
from koschei_sentinel.defense_reflex_review import (
    CorrectionReviewDecision,
    ReviewedCorrectionStep,
    review_defense_reflex_candidate,
)
from koschei_sentinel.models import StrictModel


class DefenseReflexReviewSpec(StrictModel):
    schema_version: Literal["sentinel.defense-reflex-review-spec.v1"] = (
        "sentinel.defense-reflex-review-spec.v1"
    )
    reviewer_id: str = Field(min_length=3, max_length=256)
    decision: CorrectionReviewDecision
    corrected_interpretation: str = Field(min_length=16, max_length=8000)
    corrected_steps: list[ReviewedCorrectionStep] = Field(default_factory=list, max_length=64)
    review_evidence_ids: list[str] = Field(min_length=1, max_length=256)
    outcome_verified: bool
    authorize_for_training: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an auditable reviewed Defense Reflex correction from a candidate"
    )
    parser.add_argument("--candidate", required=True, help="Defense Reflex candidate JSON")
    parser.add_argument("--review-spec", required=True, help="Reviewer-authored correction spec JSON")
    parser.add_argument("--output", required=True, help="Reviewed correction JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        candidate = DefenseReflexCandidate.model_validate_json(
            Path(args.candidate).read_text(encoding="utf-8")
        )
        spec = DefenseReflexReviewSpec.model_validate_json(
            Path(args.review_spec).read_text(encoding="utf-8")
        )
        correction = review_defense_reflex_candidate(
            candidate,
            reviewer_id=spec.reviewer_id,
            decision=spec.decision,
            corrected_interpretation=spec.corrected_interpretation,
            corrected_steps=spec.corrected_steps,
            review_evidence_ids=spec.review_evidence_ids,
            outcome_verified=spec.outcome_verified,
            authorize_for_training=spec.authorize_for_training,
        )
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(correction.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        destination.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-defense-reflex-review: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
