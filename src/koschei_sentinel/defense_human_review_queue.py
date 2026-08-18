from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.defense_authority import DefenseAction, DefenseMode
from koschei_sentinel.defense_reflex_corpus_v3 import DefenseReflexTrainingExampleV3
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import canonical_json


class HumanReviewTaskStatus(StrEnum):
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class HumanDefenseReviewStep(StrictModel):
    sequence: int = Field(gt=0)
    expected_mode: DefenseMode
    action: DefenseAction
    target_entity_id: str = Field(min_length=1, max_length=512)
    rationale: str = Field(min_length=16, max_length=4000)
    supporting_evidence_ids: list[str] = Field(min_length=1, max_length=256)


class HumanDefenseReviewTask(StrictModel):
    schema_version: Literal["sentinel.human-defense-review-task.v1"] = (
        "sentinel.human-defense-review-task.v1"
    )
    task_id: str
    status: Literal[HumanReviewTaskStatus.REVIEW_REQUIRED] = (
        HumanReviewTaskStatus.REVIEW_REQUIRED
    )
    scenario_id: str
    source_example_id: str
    source_report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    graph_snapshots: list[dict[str, object]] = Field(min_length=1)
    critical_entity_ids: list[str]
    allowed_evidence_ids: list[str]
    reviewer_instructions: list[str]
    task_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class HumanDefenseReviewSubmission(StrictModel):
    schema_version: Literal["sentinel.human-defense-review-submission.v1"] = (
        "sentinel.human-defense-review-submission.v1"
    )
    task_id: str
    task_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    reviewer_id: str = Field(min_length=3, max_length=256)
    interpretation: str = Field(min_length=24, max_length=8000)
    steps: list[HumanDefenseReviewStep] = Field(min_length=1, max_length=64)
    review_evidence_ids: list[str] = Field(min_length=1, max_length=256)
    training_authorization_requested: bool = False
    submission_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def ordered_steps(self) -> "HumanDefenseReviewSubmission":
        sequence = [row.sequence for row in self.steps]
        if sequence != list(range(1, len(sequence) + 1)):
            raise ValueError("human defense review steps must be contiguous from sequence 1")
        return self


class HumanDefenseReviewQueueManifest(StrictModel):
    schema_version: Literal["sentinel.human-defense-review-queue-manifest.v1"] = (
        "sentinel.human-defense-review-queue-manifest.v1"
    )
    task_count: int = Field(gt=0)
    scenario_count: int = Field(gt=0)
    tasks_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_example_sha256s: list[str]
    contains_answer_key: Literal[False] = False
    training_authorization: Literal[False] = False


def _digest(payload: dict[str, object], *, remove: str | None = None) -> str:
    unsigned = dict(payload)
    if remove is not None:
        unsigned.pop(remove, None)
    return hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest()


def _collect_evidence_ids(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.lower().replace("-", "_")
            if normalized == "evidence_id" and isinstance(child, str) and child:
                found.add(child)
            elif normalized == "evidence_ids" and isinstance(child, list):
                found.update(row for row in child if isinstance(row, str) and row)
            found.update(_collect_evidence_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_collect_evidence_ids(child))
    return found


def _source_example_digest(example: DefenseReflexTrainingExampleV3) -> str:
    return hashlib.sha256(example.model_dump_json().encode("utf-8")).hexdigest()


def build_human_review_task(
    example: DefenseReflexTrainingExampleV3,
) -> HumanDefenseReviewTask:
    evidence_ids = sorted(_collect_evidence_ids(example.graph_snapshots))
    instructions = [
        "Use only the graph snapshots, critical entity IDs, and evidence visible in this task.",
        "Do not assume hidden scenario truth, evaluator labels, future outcomes, or synthetic answers.",
        "Choose the least authority required by the evidence; uncertainty should remain GUARD.",
        "Every proposed action must cite supporting evidence IDs already present in the task.",
        "Do not invent future verification evidence IDs; state only the defensive step to review.",
        "This review is not training-authorized until independent adjudication verifies it.",
    ]
    payload: dict[str, object] = {
        "schema_version": "sentinel.human-defense-review-task.v1",
        "task_id": f"human-review:{example.scenario_id}:{example.example_id}",
        "status": HumanReviewTaskStatus.REVIEW_REQUIRED.value,
        "scenario_id": example.scenario_id,
        "source_example_id": example.example_id,
        "source_report_sha256": example.source_report_sha256,
        "graph_snapshots": example.graph_snapshots,
        "critical_entity_ids": example.critical_entity_ids,
        "allowed_evidence_ids": evidence_ids,
        "reviewer_instructions": instructions,
    }
    payload["task_sha256"] = _digest(payload)
    return HumanDefenseReviewTask.model_validate(payload)


def create_human_review_submission(
    task: HumanDefenseReviewTask,
    *,
    reviewer_id: str,
    interpretation: str,
    steps: list[HumanDefenseReviewStep],
    review_evidence_ids: list[str],
    training_authorization_requested: bool = False,
) -> HumanDefenseReviewSubmission:
    allowed = set(task.allowed_evidence_ids)
    cited = set(review_evidence_ids)
    for step in steps:
        cited.update(step.supporting_evidence_ids)
    unknown = sorted(cited - allowed)
    if unknown:
        raise ValueError(
            "human review cites evidence not present in the blinded task: "
            + ", ".join(unknown)
        )

    payload: dict[str, object] = {
        "schema_version": "sentinel.human-defense-review-submission.v1",
        "task_id": task.task_id,
        "task_sha256": task.task_sha256,
        "reviewer_id": reviewer_id,
        "interpretation": interpretation,
        "steps": [row.model_dump(mode="json") for row in steps],
        "review_evidence_ids": list(dict.fromkeys(review_evidence_ids)),
        "training_authorization_requested": training_authorization_requested,
    }
    payload["submission_sha256"] = _digest(payload)
    return HumanDefenseReviewSubmission.model_validate(payload)


def verify_human_review_submission(
    task: HumanDefenseReviewTask,
    submission: HumanDefenseReviewSubmission,
) -> None:
    if _digest(task.model_dump(mode="json"), remove="task_sha256") != task.task_sha256:
        raise ValueError("human review task self-hash does not verify")
    if submission.task_id != task.task_id or submission.task_sha256 != task.task_sha256:
        raise ValueError("human review submission is not bound to the supplied task")
    if (
        _digest(submission.model_dump(mode="json"), remove="submission_sha256")
        != submission.submission_sha256
    ):
        raise ValueError("human review submission self-hash does not verify")
    allowed = set(task.allowed_evidence_ids)
    cited = set(submission.review_evidence_ids)
    for step in submission.steps:
        cited.update(step.supporting_evidence_ids)
    unknown = sorted(cited - allowed)
    if unknown:
        raise ValueError(
            "human review submission references evidence outside the blinded task: "
            + ", ".join(unknown)
        )


def build_human_review_queue(
    *,
    examples_path: str | Path,
    output_dir: str | Path,
) -> HumanDefenseReviewQueueManifest:
    source = Path(examples_path)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"human review output already exists: {destination}")

    examples: list[DefenseReflexTrainingExampleV3] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            examples.append(DefenseReflexTrainingExampleV3.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"invalid Defense Reflex v3 example on line {line_number}") from exc
    if not examples:
        raise ValueError("human review queue requires at least one Defense Reflex v3 example")

    tasks = [build_human_review_task(example) for example in examples]
    task_ids = [row.task_id for row in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("human review queue contains duplicate task IDs")

    destination.mkdir(parents=True)
    tasks_path = destination / "tasks.jsonl"
    with tasks_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(task.model_dump_json() + "\n")

    source_hashes = sorted(_source_example_digest(example) for example in examples)
    manifest = HumanDefenseReviewQueueManifest(
        task_count=len(tasks),
        scenario_count=len({row.scenario_id for row in tasks}),
        tasks_sha256=hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        source_example_sha256s=source_hashes,
    )
    (destination / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
