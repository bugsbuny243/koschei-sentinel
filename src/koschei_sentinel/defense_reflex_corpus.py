from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.defense_reflex_candidates import DefenseReflexReviewStatus
from koschei_sentinel.defense_reflex_review import (
    CorrectionReviewDecision,
    ReviewedCorrectionTrajectory,
)
from koschei_sentinel.models import StrictModel


class DefenseReflexTrainingExample(StrictModel):
    schema_version: Literal["sentinel.defense-reflex-training-example.v1"] = (
        "sentinel.defense-reflex-training-example.v1"
    )
    example_id: str
    candidate_id: str
    scenario_id: str
    failure_type: str
    source_report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    correction_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    corrected_interpretation: str
    expected_sequence: list[dict[str, object]]
    review_evidence_ids: list[str]
    provenance: dict[str, str]


class DefenseReflexCorpusManifest(StrictModel):
    schema_version: Literal["sentinel.defense-reflex-corpus-manifest.v1"] = (
        "sentinel.defense-reflex-corpus-manifest.v1"
    )
    example_count: int = Field(ge=0)
    source_candidate_count: int = Field(ge=0)
    examples_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    correction_sha256s: list[str]
    ready_for_training_pipeline: bool


def _assert_trainable(correction: ReviewedCorrectionTrajectory) -> None:
    if correction.review_decision is not CorrectionReviewDecision.APPROVE:
        raise ValueError(f"correction is not approved: {correction.correction_id}")
    if correction.review_status is not DefenseReflexReviewStatus.APPROVED:
        raise ValueError(f"correction review status is not APPROVED: {correction.correction_id}")
    if not correction.outcome_verified:
        raise ValueError(f"correction outcome is not verified: {correction.correction_id}")
    if not correction.training_authorization:
        raise ValueError(f"correction lacks training authorization: {correction.correction_id}")
    if not correction.corrected_steps:
        raise ValueError(f"correction has no reviewed defensive steps: {correction.correction_id}")


def build_defense_reflex_examples(
    corrections: list[ReviewedCorrectionTrajectory],
) -> list[DefenseReflexTrainingExample]:
    if not corrections:
        raise ValueError("defense reflex corpus requires at least one reviewed correction")

    candidate_ids: set[str] = set()
    correction_hashes: set[str] = set()
    output: list[DefenseReflexTrainingExample] = []

    for correction in corrections:
        _assert_trainable(correction)
        if correction.candidate_id in candidate_ids:
            raise ValueError(f"duplicate candidate in defense reflex corpus: {correction.candidate_id}")
        if correction.correction_sha256 in correction_hashes:
            raise ValueError(
                f"duplicate correction digest in defense reflex corpus: {correction.correction_sha256}"
            )
        candidate_ids.add(correction.candidate_id)
        correction_hashes.add(correction.correction_sha256)

        sequence = [
            {
                "sequence": step.sequence,
                "expected_mode": step.expected_mode.value,
                "action": step.action.value,
                "target_entity_id": step.target_entity_id,
                "rationale": step.rationale,
                "supporting_evidence_ids": step.supporting_evidence_ids,
                "outcome_verification_ids": step.outcome_verification_ids,
            }
            for step in correction.corrected_steps
        ]
        output.append(
            DefenseReflexTrainingExample(
                example_id=f"defense-reflex:{correction.correction_sha256[:24]}",
                candidate_id=correction.candidate_id,
                scenario_id=correction.scenario_id,
                failure_type=correction.failure_type,
                source_report_sha256=correction.source_report_sha256,
                correction_sha256=correction.correction_sha256,
                corrected_interpretation=correction.corrected_interpretation,
                expected_sequence=sequence,
                review_evidence_ids=correction.review_evidence_ids,
                provenance={
                    "reviewer_id": correction.reviewer_id,
                    "correction_id": correction.correction_id,
                    "review_status": correction.review_status.value,
                },
            )
        )

    output.sort(key=lambda row: row.example_id)
    return output


def serialize_examples(examples: list[DefenseReflexTrainingExample]) -> str:
    return "".join(
        json.dumps(example.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n"
        for example in examples
    )


def build_manifest(examples: list[DefenseReflexTrainingExample]) -> DefenseReflexCorpusManifest:
    payload = serialize_examples(examples)
    return DefenseReflexCorpusManifest(
        example_count=len(examples),
        source_candidate_count=len({row.candidate_id for row in examples}),
        examples_sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        correction_sha256s=sorted(row.correction_sha256 for row in examples),
        ready_for_training_pipeline=bool(examples),
    )


def write_defense_reflex_release(
    corrections: list[ReviewedCorrectionTrajectory],
    output_dir: str | Path,
) -> DefenseReflexCorpusManifest:
    examples = build_defense_reflex_examples(corrections)
    payload = serialize_examples(examples)
    manifest = build_manifest(examples)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "examples.jsonl").write_text(payload, encoding="utf-8")
    (destination / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
