from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.cyber_world_model_episode import CyberWorldModelEpisode
from koschei_sentinel.defense_reflex_candidates import DefenseReflexReviewStatus
from koschei_sentinel.defense_reflex_review import (
    CorrectionReviewDecision,
    ReviewedCorrectionTrajectory,
)
from koschei_sentinel.models import StrictModel


class CausalDefenseTrainingExample(StrictModel):
    schema_version: Literal["sentinel.causal-defense-training-example.v1"] = (
        "sentinel.causal-defense-training-example.v1"
    )
    example_id: str
    scenario_id: str
    truth: str
    source_report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    world_model_episode_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    attack_world_line_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    correction_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    temporal_snapshots: list[dict[str, object]]
    temporal_transitions: list[dict[str, object]]
    world_line_observations: list[dict[str, object]]
    world_line_transitions: list[dict[str, object]]
    corrected_interpretation: str
    expected_defense_sequence: list[dict[str, object]]
    provenance: dict[str, str]


class CausalDefenseCorpusManifest(StrictModel):
    schema_version: Literal["sentinel.causal-defense-corpus-manifest.v1"] = (
        "sentinel.causal-defense-corpus-manifest.v1"
    )
    example_count: int = Field(ge=0)
    examples_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    episode_sha256s: list[str]
    correction_sha256s: list[str]
    ready_for_training_pipeline: bool


def _assert_reviewed(correction: ReviewedCorrectionTrajectory) -> None:
    if correction.review_decision is not CorrectionReviewDecision.APPROVE:
        raise ValueError("causal-defense example requires an approved correction")
    if correction.review_status is not DefenseReflexReviewStatus.APPROVED:
        raise ValueError("causal-defense example requires APPROVED review status")
    if not correction.outcome_verified:
        raise ValueError("causal-defense example requires verified correction outcome")
    if not correction.training_authorization:
        raise ValueError("causal-defense example requires explicit training authorization")


def build_causal_defense_example(
    episode: CyberWorldModelEpisode,
    correction: ReviewedCorrectionTrajectory,
) -> CausalDefenseTrainingExample:
    _assert_reviewed(correction)
    if episode.training_authorization:
        raise ValueError("raw world-model episode must not self-authorize training")
    if episode.scenario_id != correction.scenario_id:
        raise ValueError("world-model episode and correction belong to different scenarios")
    if episode.source_report_sha256 != correction.source_report_sha256:
        raise ValueError("world-model episode and correction do not share the same source report digest")
    if episode.attack_world_lines is None:
        raise ValueError("causal-defense example requires attack world-line lineage")

    temporal_snapshots = [row.model_dump(mode="json") for row in episode.snapshots]
    temporal_transitions = [row.model_dump(mode="json") for row in episode.transitions]
    world_line_observations = [
        row.model_dump(mode="json") for row in episode.attack_world_lines.observations
    ]
    world_line_transitions = [
        row.model_dump(mode="json") for row in episode.attack_world_lines.transitions
    ]
    expected_sequence = [
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
    digest_payload = "|".join(
        [
            episode.episode_sha256,
            episode.attack_world_lines.timeline_sha256,
            correction.correction_sha256,
            episode.source_report_sha256,
        ]
    )
    digest = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()
    return CausalDefenseTrainingExample(
        example_id=f"causal-defense:{digest[:24]}",
        scenario_id=episode.scenario_id,
        truth=episode.truth,
        source_report_sha256=episode.source_report_sha256,
        world_model_episode_sha256=episode.episode_sha256,
        attack_world_line_sha256=episode.attack_world_lines.timeline_sha256,
        correction_sha256=correction.correction_sha256,
        temporal_snapshots=temporal_snapshots,
        temporal_transitions=temporal_transitions,
        world_line_observations=world_line_observations,
        world_line_transitions=world_line_transitions,
        corrected_interpretation=correction.corrected_interpretation,
        expected_defense_sequence=expected_sequence,
        provenance={
            "episode_id": episode.episode_id,
            "attack_world_line_sha256": episode.attack_world_lines.timeline_sha256,
            "correction_id": correction.correction_id,
            "reviewer_id": correction.reviewer_id,
        },
    )


def build_causal_defense_examples(
    pairs: list[tuple[CyberWorldModelEpisode, ReviewedCorrectionTrajectory]],
) -> list[CausalDefenseTrainingExample]:
    if not pairs:
        raise ValueError("causal-defense corpus requires at least one episode/correction pair")
    output: list[CausalDefenseTrainingExample] = []
    episode_hashes: set[str] = set()
    correction_hashes: set[str] = set()
    for episode, correction in pairs:
        if episode.episode_sha256 in episode_hashes:
            raise ValueError(f"duplicate world-model episode: {episode.episode_sha256}")
        if correction.correction_sha256 in correction_hashes:
            raise ValueError(f"duplicate reviewed correction: {correction.correction_sha256}")
        episode_hashes.add(episode.episode_sha256)
        correction_hashes.add(correction.correction_sha256)
        output.append(build_causal_defense_example(episode, correction))
    output.sort(key=lambda row: row.example_id)
    return output


def serialize_causal_examples(examples: list[CausalDefenseTrainingExample]) -> str:
    return "".join(
        json.dumps(row.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n"
        for row in examples
    )


def build_causal_manifest(
    examples: list[CausalDefenseTrainingExample],
) -> CausalDefenseCorpusManifest:
    payload = serialize_causal_examples(examples)
    return CausalDefenseCorpusManifest(
        example_count=len(examples),
        examples_sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        episode_sha256s=sorted(row.world_model_episode_sha256 for row in examples),
        correction_sha256s=sorted(row.correction_sha256 for row in examples),
        ready_for_training_pipeline=bool(examples),
    )


def write_causal_defense_release(
    pairs: list[tuple[CyberWorldModelEpisode, ReviewedCorrectionTrajectory]],
    output_dir: str | Path,
) -> CausalDefenseCorpusManifest:
    examples = build_causal_defense_examples(pairs)
    manifest = build_causal_manifest(examples)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "examples.jsonl").write_text(
        serialize_causal_examples(examples),
        encoding="utf-8",
    )
    (destination / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
