from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.cyber_range import CyberRangeScenario, run_cyber_range_scenario
from koschei_sentinel.defense_reflex_candidates import DefenseReflexReviewStatus
from koschei_sentinel.defense_reflex_review import (
    CorrectionReviewDecision,
    ReviewedCorrectionTrajectory,
)
from koschei_sentinel.models import StrictModel


class DefenseReflexTrainingExampleV2(StrictModel):
    schema_version: Literal["sentinel.defense-reflex-training-example.v2"] = (
        "sentinel.defense-reflex-training-example.v2"
    )
    example_id: str
    scenario_id: str
    failure_type: str
    source_report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    correction_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    graph_snapshots: list[dict[str, object]]
    critical_entity_ids: list[str]
    observed_ticks: list[dict[str, object]]
    corrected_interpretation: str
    expected_sequence: list[dict[str, object]]
    review_evidence_ids: list[str]
    provenance: dict[str, str]


class DefenseReflexCorpusManifestV2(StrictModel):
    schema_version: Literal["sentinel.defense-reflex-corpus-manifest.v2"] = (
        "sentinel.defense-reflex-corpus-manifest.v2"
    )
    example_count: int = Field(gt=0)
    scenario_count: int = Field(gt=0)
    examples_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_report_sha256s: list[str]
    correction_sha256s: list[str]
    ready_for_training_pipeline: Literal[True] = True


def _report_sha256(report: StrictModel) -> str:
    return hashlib.sha256(report.model_dump_json().encode("utf-8")).hexdigest()


def _assert_reviewed(correction: ReviewedCorrectionTrajectory) -> None:
    if correction.review_decision is not CorrectionReviewDecision.APPROVE:
        raise ValueError("Defense Reflex v2 requires an approved correction")
    if correction.review_status is not DefenseReflexReviewStatus.APPROVED:
        raise ValueError("Defense Reflex v2 requires APPROVED review status")
    if not correction.outcome_verified:
        raise ValueError("Defense Reflex v2 requires verified correction outcome")
    if not correction.training_authorization:
        raise ValueError("Defense Reflex v2 requires explicit training authorization")
    if not correction.corrected_steps:
        raise ValueError("Defense Reflex v2 requires at least one corrected defense step")


def _observed_ticks(report: StrictModel) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for tick in report.ticks:
        rows.append(
            {
                "tick": tick.tick,
                "defense_mode": tick.defense_mode.value,
                "current_stage": tick.current_stage,
                "attack_confidence": tick.attack_confidence,
                "authorized_actions": [row.value for row in tick.authorized_actions],
                "attempted_actions": [row.value for row in tick.attempted_actions],
                "succeeded_actions": [row.value for row in tick.succeeded_actions],
                "contained": tick.contained,
                "reassessment_disposition": (
                    tick.reassessment_disposition.value
                    if tick.reassessment_disposition is not None
                    else None
                ),
                "reroute_detected": tick.reroute_detected,
            }
        )
    return rows


def _expected_sequence(
    correction: ReviewedCorrectionTrajectory,
) -> list[dict[str, object]]:
    return [
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


def build_defense_reflex_v2_example(
    scenario: CyberRangeScenario,
    correction: ReviewedCorrectionTrajectory,
) -> DefenseReflexTrainingExampleV2:
    _assert_reviewed(correction)
    if scenario.scenario_id != correction.scenario_id:
        raise ValueError("Defense Reflex scenario and correction IDs differ")

    report = run_cyber_range_scenario(scenario)
    report_sha = _report_sha256(report)
    if report_sha != correction.source_report_sha256:
        raise ValueError(
            "Defense Reflex scenario no longer reproduces the reviewed source report"
        )

    graph_snapshots = [
        graph.model_dump(mode="json") for graph in scenario.graph_snapshots
    ]
    digest_payload = "|".join(
        [
            scenario.scenario_id,
            report_sha,
            correction.correction_sha256,
            hashlib.sha256(
                json.dumps(
                    graph_snapshots,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        ]
    )
    example_digest = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()
    return DefenseReflexTrainingExampleV2(
        example_id=f"defense-reflex-v2:{example_digest[:24]}",
        scenario_id=scenario.scenario_id,
        failure_type=correction.failure_type,
        source_report_sha256=report_sha,
        correction_sha256=correction.correction_sha256,
        graph_snapshots=graph_snapshots,
        critical_entity_ids=sorted(scenario.critical_entity_ids),
        observed_ticks=_observed_ticks(report),
        corrected_interpretation=correction.corrected_interpretation,
        expected_sequence=_expected_sequence(correction),
        review_evidence_ids=correction.review_evidence_ids,
        provenance={
            "correction_id": correction.correction_id,
            "reviewer_id": correction.reviewer_id,
            "review_status": correction.review_status.value,
            "input_policy": "graph-and-observable-planner-state-only",
        },
    )


def build_defense_reflex_v2_examples(
    pairs: list[tuple[CyberRangeScenario, ReviewedCorrectionTrajectory]],
) -> list[DefenseReflexTrainingExampleV2]:
    if not pairs:
        raise ValueError("Defense Reflex v2 requires at least one scenario/correction pair")
    examples: list[DefenseReflexTrainingExampleV2] = []
    scenario_ids: set[str] = set()
    corrections: set[str] = set()
    for scenario, correction in pairs:
        if scenario.scenario_id in scenario_ids:
            raise ValueError(f"duplicate Defense Reflex scenario: {scenario.scenario_id}")
        if correction.correction_sha256 in corrections:
            raise ValueError(
                f"duplicate Defense Reflex correction: {correction.correction_sha256}"
            )
        scenario_ids.add(scenario.scenario_id)
        corrections.add(correction.correction_sha256)
        examples.append(build_defense_reflex_v2_example(scenario, correction))
    examples.sort(key=lambda row: row.example_id)
    return examples


def serialize_defense_reflex_v2(
    examples: list[DefenseReflexTrainingExampleV2],
) -> str:
    return "".join(
        json.dumps(
            row.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in examples
    )


def build_defense_reflex_v2_manifest(
    examples: list[DefenseReflexTrainingExampleV2],
) -> DefenseReflexCorpusManifestV2:
    if not examples:
        raise ValueError("Defense Reflex v2 manifest cannot be empty")
    payload = serialize_defense_reflex_v2(examples)
    return DefenseReflexCorpusManifestV2(
        example_count=len(examples),
        scenario_count=len({row.scenario_id for row in examples}),
        examples_sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        source_report_sha256s=sorted(row.source_report_sha256 for row in examples),
        correction_sha256s=sorted(row.correction_sha256 for row in examples),
    )


def write_defense_reflex_v2_release(
    pairs: list[tuple[CyberRangeScenario, ReviewedCorrectionTrajectory]],
    output_dir: str | Path,
) -> DefenseReflexCorpusManifestV2:
    examples = build_defense_reflex_v2_examples(pairs)
    payload = serialize_defense_reflex_v2(examples)
    manifest = build_defense_reflex_v2_manifest(examples)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "examples.jsonl").write_text(payload, encoding="utf-8")
    (destination / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
