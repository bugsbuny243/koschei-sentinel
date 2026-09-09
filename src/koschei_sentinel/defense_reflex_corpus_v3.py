from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.cyber_range import CyberRangeScenario, run_cyber_range_scenario
from koschei_sentinel.defense_reflex_review import ReviewedCorrectionStep
from koschei_sentinel.models import StrictModel


class DefenseLessonKind(StrEnum):
    GOLD = "GOLD"
    CORRECTION = "CORRECTION"


class DefenseReviewMethod(StrEnum):
    HUMAN = "HUMAN"
    SYNTHETIC_POLICY = "SYNTHETIC_POLICY"


class ReviewedDefenseLesson(StrictModel):
    schema_version: Literal["sentinel.reviewed-defense-lesson.v1"] = (
        "sentinel.reviewed-defense-lesson.v1"
    )
    lesson_id: str = Field(min_length=3, max_length=256)
    lesson_kind: DefenseLessonKind
    scenario_id: str = Field(min_length=3, max_length=256)
    source_report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    reviewer_id: str = Field(min_length=3, max_length=256)
    review_method: DefenseReviewMethod
    interpretation: str = Field(min_length=16, max_length=8000)
    expected_steps: list[ReviewedCorrectionStep] = Field(min_length=1, max_length=64)
    review_evidence_ids: list[str] = Field(min_length=1, max_length=256)
    training_authorization: bool
    promotion_eligible: bool
    review_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def review_authority_is_fail_closed(self) -> ReviewedDefenseLesson:
        sequences = [row.sequence for row in self.expected_steps]
        if sequences != list(range(1, len(sequences) + 1)):
            raise ValueError("reviewed defense steps must be contiguous and ordered from 1")
        if self.promotion_eligible and not self.training_authorization:
            raise ValueError("promotion-eligible defense lessons must be training-authorized")
        if self.review_method is DefenseReviewMethod.SYNTHETIC_POLICY and self.promotion_eligible:
            raise ValueError("synthetic policy-reviewed lessons cannot be promotion-eligible")
        return self


class DefenseReflexTrainingExampleV3(StrictModel):
    schema_version: Literal["sentinel.defense-reflex-training-example.v3"] = (
        "sentinel.defense-reflex-training-example.v3"
    )
    example_id: str
    scenario_id: str
    source_report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    review_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    graph_snapshots: list[dict[str, object]] = Field(min_length=1)
    critical_entity_ids: list[str]
    observed_ticks: list[dict[str, object]] = Field(min_length=1)
    expected_interpretation: str
    expected_sequence: list[dict[str, object]] = Field(min_length=1)
    provenance: dict[str, str]
    promotion_eligible: bool


class DefenseReflexCorpusManifestV3(StrictModel):
    schema_version: Literal["sentinel.defense-reflex-corpus-manifest.v3"] = (
        "sentinel.defense-reflex-corpus-manifest.v3"
    )
    example_count: int = Field(gt=0)
    scenario_count: int = Field(gt=0)
    examples_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_report_sha256s: list[str]
    review_sha256s: list[str]
    human_reviewed_examples: int = Field(ge=0)
    synthetic_policy_reviewed_examples: int = Field(ge=0)
    promotion_eligible: bool
    ready_for_training_pipeline: Literal[True] = True


def cyber_range_report_sha256(scenario: CyberRangeScenario) -> str:
    report = run_cyber_range_scenario(scenario)
    return hashlib.sha256(report.model_dump_json().encode("utf-8")).hexdigest()


def _review_digest(
    *,
    lesson_kind: DefenseLessonKind,
    scenario_id: str,
    source_report_sha256: str,
    reviewer_id: str,
    review_method: DefenseReviewMethod,
    interpretation: str,
    expected_steps: list[ReviewedCorrectionStep],
    review_evidence_ids: list[str],
    training_authorization: bool,
    promotion_eligible: bool,
) -> str:
    payload = {
        "lesson_kind": lesson_kind.value,
        "scenario_id": scenario_id,
        "source_report_sha256": source_report_sha256,
        "reviewer_id": reviewer_id,
        "review_method": review_method.value,
        "interpretation": interpretation,
        "expected_steps": [row.model_dump(mode="json") for row in expected_steps],
        "review_evidence_ids": list(dict.fromkeys(review_evidence_ids)),
        "training_authorization": training_authorization,
        "promotion_eligible": promotion_eligible,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def create_reviewed_defense_lesson(
    *,
    lesson_kind: DefenseLessonKind,
    scenario: CyberRangeScenario,
    reviewer_id: str,
    review_method: DefenseReviewMethod,
    interpretation: str,
    expected_steps: list[ReviewedCorrectionStep],
    review_evidence_ids: list[str],
    training_authorization: bool,
    promotion_eligible: bool,
) -> ReviewedDefenseLesson:
    if not review_evidence_ids:
        raise ValueError("reviewed defense lesson requires review evidence")
    if not expected_steps:
        raise ValueError("reviewed defense lesson requires at least one expected step")
    source_report_sha = cyber_range_report_sha256(scenario)
    digest = _review_digest(
        lesson_kind=lesson_kind,
        scenario_id=scenario.scenario_id,
        source_report_sha256=source_report_sha,
        reviewer_id=reviewer_id,
        review_method=review_method,
        interpretation=interpretation,
        expected_steps=expected_steps,
        review_evidence_ids=review_evidence_ids,
        training_authorization=training_authorization,
        promotion_eligible=promotion_eligible,
    )
    return ReviewedDefenseLesson(
        lesson_id=f"defense-lesson:{scenario.scenario_id}:{digest[:16]}",
        lesson_kind=lesson_kind,
        scenario_id=scenario.scenario_id,
        source_report_sha256=source_report_sha,
        reviewer_id=reviewer_id,
        review_method=review_method,
        interpretation=interpretation,
        expected_steps=expected_steps,
        review_evidence_ids=list(dict.fromkeys(review_evidence_ids)),
        training_authorization=training_authorization,
        promotion_eligible=promotion_eligible,
        review_sha256=digest,
    )


def _observed_ticks(scenario: CyberRangeScenario) -> list[dict[str, object]]:
    report = run_cyber_range_scenario(scenario)
    return [
        {
            "tick": row.tick,
            "defense_mode": row.defense_mode.value,
            "current_stage": row.current_stage,
            "attack_confidence": row.attack_confidence,
            "authorized_actions": [action.value for action in row.authorized_actions],
            "reassessment_disposition": (
                row.reassessment_disposition.value
                if row.reassessment_disposition is not None
                else None
            ),
            "reroute_detected": row.reroute_detected,
        }
        for row in report.ticks
    ]


def _expected_sequence(lesson: ReviewedDefenseLesson) -> list[dict[str, object]]:
    return [
        {
            "sequence": row.sequence,
            "expected_mode": row.expected_mode.value,
            "action": row.action.value,
            "target_entity_id": row.target_entity_id,
            "rationale": row.rationale,
            "supporting_evidence_ids": row.supporting_evidence_ids,
            "outcome_verification_required": True,
        }
        for row in lesson.expected_steps
    ]


def build_defense_reflex_v3_example(
    scenario: CyberRangeScenario,
    lesson: ReviewedDefenseLesson,
) -> DefenseReflexTrainingExampleV3:
    if not lesson.training_authorization:
        raise ValueError("Defense Reflex v3 requires explicit training authorization")
    if lesson.scenario_id != scenario.scenario_id:
        raise ValueError("Defense Reflex v3 scenario and reviewed lesson IDs differ")
    report_sha = cyber_range_report_sha256(scenario)
    if report_sha != lesson.source_report_sha256:
        raise ValueError("Defense Reflex v3 scenario drifted after review")
    review_digest = _review_digest(
        lesson_kind=lesson.lesson_kind,
        scenario_id=lesson.scenario_id,
        source_report_sha256=lesson.source_report_sha256,
        reviewer_id=lesson.reviewer_id,
        review_method=lesson.review_method,
        interpretation=lesson.interpretation,
        expected_steps=lesson.expected_steps,
        review_evidence_ids=lesson.review_evidence_ids,
        training_authorization=lesson.training_authorization,
        promotion_eligible=lesson.promotion_eligible,
    )
    if review_digest != lesson.review_sha256:
        raise ValueError("Defense Reflex v3 reviewed lesson digest mismatch")

    graphs = [row.model_dump(mode="json") for row in scenario.graph_snapshots]
    seed = "|".join(
        [
            scenario.scenario_id,
            report_sha,
            lesson.review_sha256,
            hashlib.sha256(
                json.dumps(graphs, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        ]
    )
    example_sha = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return DefenseReflexTrainingExampleV3(
        example_id=f"defense-reflex-v3:{example_sha[:24]}",
        scenario_id=scenario.scenario_id,
        source_report_sha256=report_sha,
        review_sha256=lesson.review_sha256,
        graph_snapshots=graphs,
        critical_entity_ids=sorted(scenario.critical_entity_ids),
        observed_ticks=_observed_ticks(scenario),
        expected_interpretation=lesson.interpretation,
        expected_sequence=_expected_sequence(lesson),
        provenance={
            "lesson_id": lesson.lesson_id,
            "lesson_kind": lesson.lesson_kind.value,
            "reviewer_id": lesson.reviewer_id,
            "review_method": lesson.review_method.value,
            "input_policy": "graph-protected-scope-and-observable-state-only",
            "future_outcome_ids_in_target": "false",
        },
        promotion_eligible=lesson.promotion_eligible,
    )


def build_defense_reflex_v3_examples(
    pairs: list[tuple[CyberRangeScenario, ReviewedDefenseLesson]],
) -> list[DefenseReflexTrainingExampleV3]:
    if not pairs:
        raise ValueError("Defense Reflex v3 requires at least one scenario/review pair")
    examples: list[DefenseReflexTrainingExampleV3] = []
    scenario_ids: set[str] = set()
    review_ids: set[str] = set()
    for scenario, lesson in pairs:
        if scenario.scenario_id in scenario_ids:
            raise ValueError(f"duplicate Defense Reflex v3 scenario: {scenario.scenario_id}")
        if lesson.review_sha256 in review_ids:
            raise ValueError(f"duplicate Defense Reflex v3 review: {lesson.review_sha256}")
        scenario_ids.add(scenario.scenario_id)
        review_ids.add(lesson.review_sha256)
        examples.append(build_defense_reflex_v3_example(scenario, lesson))
    return sorted(examples, key=lambda row: row.example_id)


def serialize_defense_reflex_v3(examples: list[DefenseReflexTrainingExampleV3]) -> str:
    return "".join(
        json.dumps(row.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in examples
    )


def build_defense_reflex_v3_manifest(
    examples: list[DefenseReflexTrainingExampleV3],
) -> DefenseReflexCorpusManifestV3:
    if not examples:
        raise ValueError("Defense Reflex v3 manifest cannot be empty")
    payload = serialize_defense_reflex_v3(examples)
    human = sum(
        row.provenance.get("review_method") == DefenseReviewMethod.HUMAN.value
        for row in examples
    )
    synthetic = sum(
        row.provenance.get("review_method") == DefenseReviewMethod.SYNTHETIC_POLICY.value
        for row in examples
    )
    return DefenseReflexCorpusManifestV3(
        example_count=len(examples),
        scenario_count=len({row.scenario_id for row in examples}),
        examples_sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        source_report_sha256s=sorted(row.source_report_sha256 for row in examples),
        review_sha256s=sorted(row.review_sha256 for row in examples),
        human_reviewed_examples=human,
        synthetic_policy_reviewed_examples=synthetic,
        promotion_eligible=all(row.promotion_eligible for row in examples),
    )


def write_defense_reflex_v3_release(
    pairs: list[tuple[CyberRangeScenario, ReviewedDefenseLesson]],
    output_dir: str | Path,
) -> DefenseReflexCorpusManifestV3:
    examples = build_defense_reflex_v3_examples(pairs)
    payload = serialize_defense_reflex_v3(examples)
    manifest = build_defense_reflex_v3_manifest(examples)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "examples.jsonl").write_text(payload, encoding="utf-8")
    (destination / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
