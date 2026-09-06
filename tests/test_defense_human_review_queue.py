from __future__ import annotations

import json
from pathlib import Path

import pytest

from koschei_sentinel.defense_authority import DefenseActionType, DefenseMode
from koschei_sentinel.defense_human_review_queue import (
    HumanDefenseReviewStep,
    build_human_review_queue,
    build_human_review_task,
    create_human_review_submission,
    verify_human_review_submission,
)
from koschei_sentinel.defense_reflex_corpus_v3 import DefenseReflexTrainingExampleV3


def _example() -> DefenseReflexTrainingExampleV3:
    return DefenseReflexTrainingExampleV3(
        example_id="reflex-v3:test-scenario:001",
        scenario_id="test-scenario",
        source_report_sha256="a" * 64,
        review_sha256="b" * 64,
        graph_snapshots=[
            {
                "tick": 0,
                "entities": [
                    {"entity_id": "endpoint:1", "entity_type": "DEVICE"},
                    {"entity_id": "process:1", "entity_type": "PROCESS"},
                ],
                "relations": [
                    {
                        "relation_id": "relation:1",
                        "source_entity_id": "endpoint:1",
                        "target_entity_id": "process:1",
                        "evidence": [
                            {
                                "evidence_id": "ev-1",
                                "source": "fixture",
                                "content_sha256": "c" * 64,
                            }
                        ],
                    }
                ],
            }
        ],
        critical_entity_ids=["endpoint:1"],
        observed_ticks=[
            {
                "tick": 0,
                "defense_mode": "GUARD",
                "current_stage": "INITIAL_ACCESS",
            }
        ],
        expected_interpretation="Synthetic answer key that must never enter the human task.",
        expected_sequence=[
            {
                "sequence": 1,
                "expected_mode": "GUARD",
                "action": "COLLECT_EVIDENCE",
                "target_entity_id": "endpoint:1",
                "rationale": "Synthetic target answer that must be blinded.",
                "supporting_evidence_ids": ["ev-1"],
                "outcome_verification_required": True,
            }
        ],
        provenance={"review_method": "SYNTHETIC_POLICY"},
        promotion_eligible=False,
    )


def test_human_review_task_removes_answer_key() -> None:
    task = build_human_review_task(_example())
    payload = task.model_dump(mode="json")

    assert task.allowed_evidence_ids == ["ev-1"]
    assert payload["status"] == "REVIEW_REQUIRED"
    assert "expected_interpretation" not in payload
    assert "expected_sequence" not in payload
    assert "observed_ticks" not in payload
    assert "review_sha256" not in payload
    assert "provenance" not in payload
    assert "Synthetic answer key" not in json.dumps(payload)


def test_human_review_submission_is_bound_to_existing_evidence() -> None:
    task = build_human_review_task(_example())
    submission = create_human_review_submission(
        task,
        reviewer_id="reviewer:test",
        interpretation=(
            "The observed graph supports evidence collection before any higher authority action."
        ),
        steps=[
            HumanDefenseReviewStep(
                sequence=1,
                expected_mode=DefenseMode.GUARD,
                action=DefenseActionType.COLLECT_EVIDENCE,
                target_entity_id="endpoint:1",
                rationale="Collect additional evidence while preserving the endpoint state.",
                supporting_evidence_ids=["ev-1"],
            )
        ],
        review_evidence_ids=["ev-1"],
    )

    verify_human_review_submission(task, submission)
    assert submission.training_authorization_requested is False


def test_human_review_rejects_invented_evidence() -> None:
    task = build_human_review_task(_example())

    with pytest.raises(ValueError, match="evidence not present"):
        create_human_review_submission(
            task,
            reviewer_id="reviewer:test",
            interpretation=(
                "The reviewer must not cite evidence that is not present in the blinded graph."
            ),
            steps=[
                HumanDefenseReviewStep(
                    sequence=1,
                    expected_mode=DefenseMode.GUARD,
                    action=DefenseActionType.COLLECT_EVIDENCE,
                    target_entity_id="endpoint:1",
                    rationale="Collect evidence without escalating authority prematurely.",
                    supporting_evidence_ids=["invented-evidence"],
                )
            ],
            review_evidence_ids=["ev-1"],
        )


def test_queue_manifest_is_explicitly_not_training_authorized(tmp_path: Path) -> None:
    source = tmp_path / "examples.jsonl"
    source.write_text(_example().model_dump_json() + "\n", encoding="utf-8")
    destination = tmp_path / "queue"

    manifest = build_human_review_queue(
        examples_path=source,
        output_dir=destination,
    )

    assert manifest.task_count == 1
    assert manifest.scenario_count == 1
    assert manifest.contains_answer_key is False
    assert manifest.training_authorization is False

    task_payload = json.loads(
        (destination / "tasks.jsonl").read_text(encoding="utf-8").strip()
    )
    assert "expected_interpretation" not in task_payload
    assert "expected_sequence" not in task_payload
