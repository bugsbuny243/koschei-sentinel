from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.cyber_range import CyberRangeScenario, run_cyber_range_scenario
from koschei_sentinel.defense_reflex_candidates import mine_defense_reflex_candidates
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import canonical_json


class GoldReviewSplit(StrEnum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    HOLDOUT = "HOLDOUT"


class GoldReviewPurpose(StrEnum):
    GOLD_BEHAVIOR = "GOLD_BEHAVIOR"
    CORRECTION = "CORRECTION"


class GoldReviewSplitPolicy(StrictModel):
    schema_version: Literal["sentinel.gold-review-split-policy.v1"] = (
        "sentinel.gold-review-split-policy.v1"
    )
    seed: str = Field(default="koschei-gold-v1", min_length=3, max_length=128)
    train_basis_points: int = Field(default=8000, ge=1, le=9998)
    validation_basis_points: int = Field(default=1000, ge=1, le=9998)
    holdout_basis_points: int = Field(default=1000, ge=1, le=9998)

    @model_validator(mode="after")
    def ratios_sum_to_one(self) -> "GoldReviewSplitPolicy":
        total = (
            self.train_basis_points
            + self.validation_basis_points
            + self.holdout_basis_points
        )
        if total != 10000:
            raise ValueError("gold review split basis points must sum to 10000")
        return self


class GoldDefenseReviewPacket(StrictModel):
    schema_version: Literal["sentinel.gold-defense-review-packet.v1"] = (
        "sentinel.gold-defense-review-packet.v1"
    )
    packet_id: str
    scenario_id: str
    source_report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    split: GoldReviewSplit
    split_basis_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    split_policy_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    purpose: GoldReviewPurpose
    model_visible_context: dict[str, object]
    model_visible_context_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    review_only_context: dict[str, object]
    candidate_ids: list[str]
    training_authorization: Literal[False] = False
    promotion_eligible: Literal[False] = False
    packet_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class GoldReviewQueueManifest(StrictModel):
    schema_version: Literal["sentinel.gold-review-queue-manifest.v1"] = (
        "sentinel.gold-review-queue-manifest.v1"
    )
    packet_count: int = Field(gt=0)
    scenario_count: int = Field(gt=0)
    train_packets: int = Field(ge=0)
    validation_packets: int = Field(ge=0)
    holdout_packets: int = Field(ge=0)
    split_policy_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    train_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    validation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    holdout_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    queue_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    review_required: Literal[True] = True
    training_ready: Literal[False] = False


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _split_policy_sha256(policy: GoldReviewSplitPolicy) -> str:
    return _sha256_text(canonical_json(policy.model_dump(mode="json")))


def _report_sha256(scenario: CyberRangeScenario) -> tuple[object, str]:
    report = run_cyber_range_scenario(scenario)
    digest = hashlib.sha256(report.model_dump_json().encode("utf-8")).hexdigest()
    return report, digest


def _split_basis(
    scenario_id: str,
    report_sha256: str,
    policy: GoldReviewSplitPolicy,
) -> str:
    return _sha256_text(f"{policy.seed}|{scenario_id}|{report_sha256}")


def _split_for_basis(
    split_basis_sha256: str,
    policy: GoldReviewSplitPolicy,
) -> GoldReviewSplit:
    bucket = int(split_basis_sha256[:16], 16) % 10000
    if bucket < policy.train_basis_points:
        return GoldReviewSplit.TRAIN
    if bucket < policy.train_basis_points + policy.validation_basis_points:
        return GoldReviewSplit.VALIDATION
    return GoldReviewSplit.HOLDOUT


def _model_visible_ticks(report: object) -> list[dict[str, object]]:
    return [
        {
            "tick": row.tick,
            "graph_id": row.graph_id,
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


def _packet_digest(payload: dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("packet_sha256", None)
    return _sha256_text(canonical_json(unsigned))


def build_gold_review_packet(
    scenario: CyberRangeScenario,
    *,
    policy: GoldReviewSplitPolicy,
) -> GoldDefenseReviewPacket:
    report, report_sha = _report_sha256(scenario)
    candidates = mine_defense_reflex_candidates(report)
    split_basis = _split_basis(scenario.scenario_id, report_sha, policy)
    split = _split_for_basis(split_basis, policy)
    policy_sha = _split_policy_sha256(policy)
    purpose = (
        GoldReviewPurpose.CORRECTION
        if candidates
        else GoldReviewPurpose.GOLD_BEHAVIOR
    )
    model_visible_context: dict[str, object] = {
        "scenario_id": scenario.scenario_id,
        "critical_entity_ids": sorted(scenario.critical_entity_ids),
        "graph_snapshots": [
            graph.model_dump(mode="json") for graph in scenario.graph_snapshots
        ],
        "observed_ticks": _model_visible_ticks(report),
    }
    visible_sha = _sha256_text(canonical_json(model_visible_context))
    review_only_context: dict[str, object] = {
        "scenario_truth": scenario.truth.value,
        "expected_reroute_ticks": list(scenario.expected_reroute_ticks),
        "simulated_action_outcomes": [
            row.model_dump(mode="json") for row in scenario.action_outcomes
        ],
        "range_report": report.model_dump(mode="json"),
        "failure_candidates": [row.model_dump(mode="json") for row in candidates],
    }
    payload: dict[str, object] = {
        "schema_version": "sentinel.gold-defense-review-packet.v1",
        "packet_id": f"gold-review:{scenario.scenario_id}:{split_basis[:16]}",
        "scenario_id": scenario.scenario_id,
        "source_report_sha256": report_sha,
        "split": split.value,
        "split_basis_sha256": split_basis,
        "split_policy_sha256": policy_sha,
        "purpose": purpose.value,
        "model_visible_context": model_visible_context,
        "model_visible_context_sha256": visible_sha,
        "review_only_context": review_only_context,
        "candidate_ids": sorted(row.candidate_id for row in candidates),
        "training_authorization": False,
        "promotion_eligible": False,
    }
    payload["packet_sha256"] = _packet_digest(payload)
    return GoldDefenseReviewPacket.model_validate(payload)


def build_gold_review_queue(
    scenarios: list[CyberRangeScenario],
    *,
    policy: GoldReviewSplitPolicy | None = None,
) -> tuple[list[GoldDefenseReviewPacket], GoldReviewSplitPolicy]:
    if not scenarios:
        raise ValueError("gold review queue requires at least one scenario")
    selected_policy = policy or GoldReviewSplitPolicy()
    scenario_ids = [row.scenario_id for row in scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("gold review queue scenario IDs must be unique")
    packets = [
        build_gold_review_packet(scenario, policy=selected_policy)
        for scenario in scenarios
    ]
    return sorted(packets, key=lambda row: row.packet_id), selected_policy


def _serialize(rows: list[GoldDefenseReviewPacket]) -> str:
    return "".join(
        json.dumps(row.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in sorted(rows, key=lambda item: item.packet_id)
    )


def write_gold_review_queue(
    scenarios: list[CyberRangeScenario],
    output_dir: str | Path,
    *,
    policy: GoldReviewSplitPolicy | None = None,
) -> GoldReviewQueueManifest:
    packets, selected_policy = build_gold_review_queue(scenarios, policy=policy)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)

    by_split = {
        split: [row for row in packets if row.split is split]
        for split in GoldReviewSplit
    }
    serialized = {split: _serialize(rows) for split, rows in by_split.items()}
    filenames = {
        GoldReviewSplit.TRAIN: "train.jsonl",
        GoldReviewSplit.VALIDATION: "validation.jsonl",
        GoldReviewSplit.HOLDOUT: "holdout.jsonl",
    }
    for split, filename in filenames.items():
        (destination / filename).write_text(serialized[split], encoding="utf-8")

    policy_payload = (
        json.dumps(
            selected_policy.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (destination / "split-policy.json").write_text(policy_payload, encoding="utf-8")

    train_sha = _sha256_text(serialized[GoldReviewSplit.TRAIN])
    validation_sha = _sha256_text(serialized[GoldReviewSplit.VALIDATION])
    holdout_sha = _sha256_text(serialized[GoldReviewSplit.HOLDOUT])
    policy_sha = _split_policy_sha256(selected_policy)
    queue_sha = _sha256_text(
        canonical_json(
            {
                "split_policy_sha256": policy_sha,
                "train_sha256": train_sha,
                "validation_sha256": validation_sha,
                "holdout_sha256": holdout_sha,
            }
        )
    )
    manifest = GoldReviewQueueManifest(
        packet_count=len(packets),
        scenario_count=len({row.scenario_id for row in packets}),
        train_packets=len(by_split[GoldReviewSplit.TRAIN]),
        validation_packets=len(by_split[GoldReviewSplit.VALIDATION]),
        holdout_packets=len(by_split[GoldReviewSplit.HOLDOUT]),
        split_policy_sha256=policy_sha,
        train_sha256=train_sha,
        validation_sha256=validation_sha,
        holdout_sha256=holdout_sha,
        queue_sha256=queue_sha,
    )
    (destination / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
