from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.cyber_range import CyberRangeScenario
from koschei_sentinel.defense_reflex_corpus_v3 import (
    DefenseLessonKind,
    DefenseReviewMethod,
    build_defense_reflex_v3_example,
    build_defense_reflex_v3_manifest,
    create_reviewed_defense_lesson,
    serialize_defense_reflex_v3,
)
from koschei_sentinel.defense_reflex_gold_queue import (
    GoldDefenseReviewPacket,
    GoldReviewPurpose,
    GoldReviewSplit,
    _packet_digest,
)
from koschei_sentinel.defense_reflex_gold_review import (
    GoldReviewedPacket,
    _review_digest,
)
from koschei_sentinel.defense_reflex_review import CorrectionReviewDecision
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import canonical_json


class GoldHoldoutEvaluationCase(StrictModel):
    schema_version: Literal["sentinel.gold-holdout-evaluation-case.v1"] = (
        "sentinel.gold-holdout-evaluation-case.v1"
    )
    case_id: str
    scenario_id: str
    source_report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    packet_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    review_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_context: dict[str, object]
    expected_interpretation: str
    expected_sequence: list[dict[str, object]] = Field(min_length=1)
    reviewer_id: str
    training_authorization: Literal[False] = False
    evaluation_authorization: Literal[True] = True
    case_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class GoldHoldoutManifest(StrictModel):
    schema_version: Literal["sentinel.gold-holdout-manifest.v1"] = (
        "sentinel.gold-holdout-manifest.v1"
    )
    case_count: int = Field(gt=0)
    cases_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    scenario_ids: list[str] = Field(min_length=1)
    review_sha256s: list[str] = Field(min_length=1)
    training_authorization: Literal[False] = False
    evaluation_ready: Literal[True] = True


class GoldDefenseReleaseManifest(StrictModel):
    schema_version: Literal["sentinel.gold-defense-release-manifest.v1"] = (
        "sentinel.gold-defense-release-manifest.v1"
    )
    train_examples: int = Field(gt=0)
    validation_examples: int = Field(gt=0)
    holdout_cases: int = Field(gt=0)
    train_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    validation_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    holdout_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    release_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    human_review_only: Literal[True] = True
    holdout_training_authorization: Literal[False] = False
    promotion_ready: Literal[True] = True


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(payload: str) -> str:
    return _sha256_bytes(payload.encode("utf-8"))


def _expected_sequence(reviewed: GoldReviewedPacket) -> list[dict[str, object]]:
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
        for row in reviewed.expected_steps
    ]


def _verify_review_binding(
    scenario: CyberRangeScenario,
    packet: GoldDefenseReviewPacket,
    reviewed: GoldReviewedPacket,
) -> None:
    if _packet_digest(packet.model_dump(mode="json")) != packet.packet_sha256:
        raise ValueError("Gold release packet self-hash does not verify")
    if _review_digest(reviewed.model_dump(mode="json")) != reviewed.review_sha256:
        raise ValueError("Gold release human review self-hash does not verify")
    if reviewed.decision is not CorrectionReviewDecision.APPROVE:
        raise ValueError("Gold release accepts only approved human reviews")
    bindings = (
        ("packet_id", reviewed.packet_id, packet.packet_id),
        ("packet_sha256", reviewed.packet_sha256, packet.packet_sha256),
        ("scenario_id", reviewed.scenario_id, scenario.scenario_id),
        ("packet scenario_id", packet.scenario_id, scenario.scenario_id),
        ("source_report_sha256", reviewed.source_report_sha256, packet.source_report_sha256),
        ("split", reviewed.split, packet.split),
    )
    for label, observed, expected in bindings:
        if observed != expected:
            raise ValueError(f"Gold release binding mismatch: {label}")
    if not reviewed.outcome_verified:
        raise ValueError("Gold release requires verified human-reviewed outcomes")
    if reviewed.split is GoldReviewSplit.HOLDOUT:
        if reviewed.training_authorization or reviewed.promotion_eligible:
            raise ValueError("Gold HOLDOUT review cannot enter training or promotion corpus")
        if not reviewed.evaluation_authorization:
            raise ValueError("Gold HOLDOUT review requires evaluation authorization")
    else:
        if not reviewed.training_authorization or not reviewed.promotion_eligible:
            raise ValueError("Gold TRAIN/VALIDATION review lacks training authorization")


def _training_example(
    scenario: CyberRangeScenario,
    packet: GoldDefenseReviewPacket,
    reviewed: GoldReviewedPacket,
):
    lesson = create_reviewed_defense_lesson(
        lesson_kind=(
            DefenseLessonKind.CORRECTION
            if packet.purpose is GoldReviewPurpose.CORRECTION
            else DefenseLessonKind.GOLD
        ),
        scenario=scenario,
        reviewer_id=reviewed.reviewer_id,
        review_method=DefenseReviewMethod.HUMAN,
        interpretation=reviewed.interpretation,
        expected_steps=reviewed.expected_steps,
        review_evidence_ids=reviewed.review_evidence_ids,
        training_authorization=True,
        promotion_eligible=True,
    )
    return build_defense_reflex_v3_example(scenario, lesson)


def _holdout_case(
    packet: GoldDefenseReviewPacket,
    reviewed: GoldReviewedPacket,
) -> GoldHoldoutEvaluationCase:
    payload: dict[str, object] = {
        "schema_version": "sentinel.gold-holdout-evaluation-case.v1",
        "case_id": f"gold-holdout:{packet.scenario_id}:{reviewed.review_sha256[:16]}",
        "scenario_id": packet.scenario_id,
        "source_report_sha256": packet.source_report_sha256,
        "packet_sha256": packet.packet_sha256,
        "review_sha256": reviewed.review_sha256,
        "input_context": packet.model_visible_context,
        "expected_interpretation": reviewed.interpretation,
        "expected_sequence": _expected_sequence(reviewed),
        "reviewer_id": reviewed.reviewer_id,
        "training_authorization": False,
        "evaluation_authorization": True,
    }
    payload["case_sha256"] = _sha256_text(canonical_json(payload))
    return GoldHoldoutEvaluationCase.model_validate(payload)


def _serialize_holdout(cases: list[GoldHoldoutEvaluationCase]) -> str:
    return "".join(
        json.dumps(row.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in sorted(cases, key=lambda item: item.case_id)
    )


def _write_training_split(examples: list[object], destination: Path) -> bytes:
    payload = serialize_defense_reflex_v3(examples)
    manifest = build_defense_reflex_v3_manifest(examples)
    if not manifest.promotion_eligible:
        raise ValueError("Gold human-reviewed training split must be promotion-eligible")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "examples.jsonl").write_text(payload, encoding="utf-8")
    manifest_raw = (
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    (destination / "manifest.json").write_bytes(manifest_raw)
    return manifest_raw


def write_gold_defense_release(
    rows: list[tuple[CyberRangeScenario, GoldDefenseReviewPacket, GoldReviewedPacket]],
    output_dir: str | Path,
) -> GoldDefenseReleaseManifest:
    if not rows:
        raise ValueError("Gold defense release requires reviewed packets")
    scenario_ids = [scenario.scenario_id for scenario, _packet, _reviewed in rows]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("Gold defense release scenario IDs must be unique")

    train_examples = []
    validation_examples = []
    holdout_cases: list[GoldHoldoutEvaluationCase] = []
    for scenario, packet, reviewed in rows:
        _verify_review_binding(scenario, packet, reviewed)
        if packet.split is GoldReviewSplit.TRAIN:
            train_examples.append(_training_example(scenario, packet, reviewed))
        elif packet.split is GoldReviewSplit.VALIDATION:
            validation_examples.append(_training_example(scenario, packet, reviewed))
        else:
            holdout_cases.append(_holdout_case(packet, reviewed))

    if not train_examples or not validation_examples or not holdout_cases:
        raise ValueError(
            "promotion-ready Gold release requires non-empty TRAIN, VALIDATION and HOLDOUT splits"
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    train_manifest_raw = _write_training_split(train_examples, destination / "train")
    validation_manifest_raw = _write_training_split(
        validation_examples,
        destination / "validation",
    )

    holdout_dir = destination / "holdout"
    holdout_dir.mkdir()
    holdout_payload = _serialize_holdout(holdout_cases)
    (holdout_dir / "cases.jsonl").write_text(holdout_payload, encoding="utf-8")
    holdout_manifest = GoldHoldoutManifest(
        case_count=len(holdout_cases),
        cases_sha256=_sha256_text(holdout_payload),
        scenario_ids=sorted(row.scenario_id for row in holdout_cases),
        review_sha256s=sorted(row.review_sha256 for row in holdout_cases),
    )
    holdout_manifest_raw = (
        json.dumps(
            holdout_manifest.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    (holdout_dir / "manifest.json").write_bytes(holdout_manifest_raw)

    release_payload = {
        "train_manifest_sha256": _sha256_bytes(train_manifest_raw),
        "validation_manifest_sha256": _sha256_bytes(validation_manifest_raw),
        "holdout_manifest_sha256": _sha256_bytes(holdout_manifest_raw),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "holdout_cases": len(holdout_cases),
    }
    release_sha = _sha256_text(canonical_json(release_payload))
    manifest = GoldDefenseReleaseManifest(
        **release_payload,
        release_sha256=release_sha,
    )
    (destination / "release-manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
