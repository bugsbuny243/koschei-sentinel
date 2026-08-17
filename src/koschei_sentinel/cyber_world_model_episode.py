from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.adaptive_defense import graph_fingerprint
from koschei_sentinel.cyber_range import CyberRangeReport, CyberRangeScenario
from koschei_sentinel.cyber_state_graph import EvidenceStatus
from koschei_sentinel.models import StrictModel


class TemporalTransitionType(StrEnum):
    STABLE = "STABLE"
    STATE_CHANGE = "STATE_CHANGE"
    ATTACKER_REROUTE = "ATTACKER_REROUTE"
    CONTAINMENT = "CONTAINMENT"
    RECOVERY = "RECOVERY"


class WorldModelSnapshot(StrictModel):
    tick: int = Field(ge=0)
    graph_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    entity_count: int = Field(ge=0)
    relation_count: int = Field(ge=0)
    observed_relations: int = Field(ge=0)
    inferred_relations: int = Field(ge=0)
    predicted_relations: int = Field(ge=0)
    disproved_relations: int = Field(ge=0)
    current_stage: str | None
    defense_mode: str
    attack_confidence: float = Field(ge=0.0, le=1.0)
    attempted_actions: list[str]
    succeeded_actions: list[str]
    contained: bool


class WorldModelTransition(StrictModel):
    from_tick: int = Field(ge=0)
    to_tick: int = Field(ge=1)
    transition_type: TemporalTransitionType
    graph_changed: bool
    previous_stage: str | None
    current_stage: str | None
    containment_changed: bool
    reroute_expected: bool
    reroute_detected: bool


class CyberWorldModelEpisode(StrictModel):
    schema_version: Literal["sentinel.cyber-world-model-episode.v1"] = (
        "sentinel.cyber-world-model-episode.v1"
    )
    episode_id: str = Field(min_length=3, max_length=256)
    scenario_id: str
    truth: str
    graph_id: str
    snapshots: list[WorldModelSnapshot] = Field(min_length=1, max_length=128)
    transitions: list[WorldModelTransition] = Field(default_factory=list, max_length=127)
    episode_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    training_authorization: bool = False

    @model_validator(mode="after")
    def temporal_integrity(self) -> "CyberWorldModelEpisode":
        ticks = [row.tick for row in self.snapshots]
        if ticks != list(range(len(ticks))):
            raise ValueError("world-model snapshots must use contiguous ticks from zero")
        if len(self.transitions) != max(0, len(self.snapshots) - 1):
            raise ValueError("world-model episode requires exactly one transition per snapshot edge")
        for index, transition in enumerate(self.transitions, 1):
            if transition.from_tick != index - 1 or transition.to_tick != index:
                raise ValueError("world-model transition ticks must align with snapshot order")
        if self.training_authorization:
            raise ValueError(
                "raw world-model episodes are observational records and cannot self-authorize training"
            )
        return self


def _snapshot_counts(graph: object) -> dict[EvidenceStatus, int]:
    counts = {status: 0 for status in EvidenceStatus}
    for relation in graph.relations:
        counts[relation.status] += 1
    return counts


def _transition_type(
    previous: WorldModelSnapshot,
    current: WorldModelSnapshot,
    *,
    reroute_expected: bool,
    reroute_detected: bool,
) -> TemporalTransitionType:
    if reroute_expected and reroute_detected:
        return TemporalTransitionType.ATTACKER_REROUTE
    if not previous.contained and current.contained:
        return TemporalTransitionType.CONTAINMENT
    if previous.current_stage is not None and current.current_stage is None:
        return TemporalTransitionType.RECOVERY
    if previous.graph_sha256 == current.graph_sha256:
        return TemporalTransitionType.STABLE
    return TemporalTransitionType.STATE_CHANGE


def _episode_digest(
    *,
    scenario_id: str,
    graph_id: str,
    snapshots: list[WorldModelSnapshot],
    transitions: list[WorldModelTransition],
) -> str:
    payload = {
        "scenario_id": scenario_id,
        "graph_id": graph_id,
        "snapshots": [row.model_dump(mode="json") for row in snapshots],
        "transitions": [row.model_dump(mode="json") for row in transitions],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_world_model_episode(
    scenario: CyberRangeScenario,
    report: CyberRangeReport,
) -> CyberWorldModelEpisode:
    if scenario.scenario_id != report.scenario_id:
        raise ValueError("scenario/report identifiers do not match")
    if scenario.truth is not report.truth:
        raise ValueError("scenario/report truth labels do not match")
    if len(scenario.graph_snapshots) != len(report.ticks):
        raise ValueError("scenario/report tick counts do not match")

    snapshots: list[WorldModelSnapshot] = []
    graph_ids = {graph.graph_id for graph in scenario.graph_snapshots}
    if len(graph_ids) != 1:
        raise ValueError("world-model episode requires one stable graph_id")
    graph_id = next(iter(graph_ids))

    for index, (graph, tick) in enumerate(zip(scenario.graph_snapshots, report.ticks, strict=True)):
        if tick.tick != index:
            raise ValueError("range report ticks are not contiguous")
        if tick.graph_id != graph.graph_id:
            raise ValueError("range report tick references a different graph")
        counts = _snapshot_counts(graph)
        snapshots.append(
            WorldModelSnapshot(
                tick=index,
                graph_sha256=graph_fingerprint(graph),
                entity_count=len(graph.entities),
                relation_count=len(graph.relations),
                observed_relations=counts[EvidenceStatus.OBSERVED],
                inferred_relations=counts[EvidenceStatus.INFERRED],
                predicted_relations=counts[EvidenceStatus.PREDICTED],
                disproved_relations=counts[EvidenceStatus.DISPROVED],
                current_stage=tick.current_stage,
                defense_mode=tick.defense_mode.value,
                attack_confidence=tick.attack_confidence,
                attempted_actions=[action.value for action in tick.attempted_actions],
                succeeded_actions=[action.value for action in tick.succeeded_actions],
                contained=tick.contained,
            )
        )

    transitions: list[WorldModelTransition] = []
    for index in range(1, len(snapshots)):
        previous = snapshots[index - 1]
        current = snapshots[index]
        tick = report.ticks[index]
        transitions.append(
            WorldModelTransition(
                from_tick=index - 1,
                to_tick=index,
                transition_type=_transition_type(
                    previous,
                    current,
                    reroute_expected=tick.reroute_expected,
                    reroute_detected=tick.reroute_detected,
                ),
                graph_changed=previous.graph_sha256 != current.graph_sha256,
                previous_stage=previous.current_stage,
                current_stage=current.current_stage,
                containment_changed=previous.contained != current.contained,
                reroute_expected=tick.reroute_expected,
                reroute_detected=tick.reroute_detected,
            )
        )

    digest = _episode_digest(
        scenario_id=scenario.scenario_id,
        graph_id=graph_id,
        snapshots=snapshots,
        transitions=transitions,
    )
    return CyberWorldModelEpisode(
        episode_id=f"world:{scenario.scenario_id}:{digest[:16]}",
        scenario_id=scenario.scenario_id,
        truth=scenario.truth.value,
        graph_id=graph_id,
        snapshots=snapshots,
        transitions=transitions,
        episode_sha256=digest,
        training_authorization=False,
    )
