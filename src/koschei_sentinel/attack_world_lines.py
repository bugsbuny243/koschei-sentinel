from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from enum import StrEnum
from typing import Literal

from pydantic import Field

from koschei_sentinel.attack_progression import AttackComponentReport, analyze_attack_progression
from koschei_sentinel.cyber_state_graph import CyberStateGraph
from koschei_sentinel.models import StrictModel


class WorldLineTransitionType(StrEnum):
    NEW = "NEW"
    CONTINUED = "CONTINUED"
    SPLIT = "SPLIT"
    MERGED = "MERGED"
    RECONFIGURED = "RECONFIGURED"
    ENDED = "ENDED"


class AttackWorldLinePolicy(StrictModel):
    schema_version: Literal["sentinel.attack-world-line-policy.v1"] = (
        "sentinel.attack-world-line-policy.v1"
    )
    min_overlap_score: float = Field(default=0.25, ge=0.0, le=1.0)
    entity_weight: float = Field(default=0.8, ge=0.0, le=1.0)
    relation_weight: float = Field(default=0.2, ge=0.0, le=1.0)


class WorldLineComponentObservation(StrictModel):
    tick: int = Field(ge=0)
    world_line_id: str
    component_id: str
    entity_ids: list[str]
    relation_ids: list[str]
    current_stage: str | None
    risk_score: float = Field(ge=0.0, le=1.0)


class AttackWorldLineTransition(StrictModel):
    from_tick: int | None = Field(default=None, ge=0)
    to_tick: int = Field(ge=0)
    transition_type: WorldLineTransitionType
    predecessor_world_line_ids: list[str]
    successor_world_line_ids: list[str]
    predecessor_component_ids: list[str]
    successor_component_ids: list[str]
    max_overlap_score: float = Field(ge=0.0, le=1.0)


class AttackWorldLineTimeline(StrictModel):
    schema_version: Literal["sentinel.attack-world-line-timeline.v1"] = (
        "sentinel.attack-world-line-timeline.v1"
    )
    stream_id: str = Field(min_length=3, max_length=256)
    ticks: int = Field(ge=1)
    policy: AttackWorldLinePolicy
    observations: list[WorldLineComponentObservation]
    transitions: list[AttackWorldLineTransition]
    timeline_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def component_overlap_score(
    previous: AttackComponentReport,
    current: AttackComponentReport,
    policy: AttackWorldLinePolicy,
) -> float:
    entity_score = _jaccard(set(previous.entity_ids), set(current.entity_ids))
    relation_score = _jaccard(set(previous.active_relation_ids), set(current.active_relation_ids))
    total_weight = policy.entity_weight + policy.relation_weight
    if total_weight <= 0.0:
        raise ValueError("attack world-line policy requires positive overlap weights")
    return min(
        1.0,
        (
            policy.entity_weight * entity_score
            + policy.relation_weight * relation_score
        )
        / total_weight,
    )


def _new_world_line_id(
    *,
    stream_id: str,
    tick: int,
    component_id: str,
    predecessors: list[str],
) -> str:
    payload = "|".join(
        [stream_id, str(tick), component_id, ",".join(sorted(predecessors))]
    )
    return "world-line:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _bipartite_groups(
    previous: list[AttackComponentReport],
    current: list[AttackComponentReport],
    policy: AttackWorldLinePolicy,
) -> tuple[list[tuple[set[int], set[int]]], dict[tuple[int, int], float]]:
    edge_scores: dict[tuple[int, int], float] = {}
    prev_to_curr: dict[int, set[int]] = defaultdict(set)
    curr_to_prev: dict[int, set[int]] = defaultdict(set)
    for p_index, p_component in enumerate(previous):
        for c_index, c_component in enumerate(current):
            score = component_overlap_score(p_component, c_component, policy)
            if score < policy.min_overlap_score:
                continue
            edge_scores[(p_index, c_index)] = score
            prev_to_curr[p_index].add(c_index)
            curr_to_prev[c_index].add(p_index)

    groups: list[tuple[set[int], set[int]]] = []
    seen_prev: set[int] = set()
    seen_curr: set[int] = set()
    for start in sorted(set(prev_to_curr) | set(curr_to_prev)):
        if start in seen_prev and start in seen_curr:
            continue
        p_group: set[int] = set()
        c_group: set[int] = set()
        queue: deque[tuple[str, int]] = deque()
        if start in prev_to_curr and start not in seen_prev:
            queue.append(("p", start))
        elif start in curr_to_prev and start not in seen_curr:
            queue.append(("c", start))
        else:
            continue
        while queue:
            side, index = queue.popleft()
            if side == "p":
                if index in p_group:
                    continue
                p_group.add(index)
                seen_prev.add(index)
                for neighbor in sorted(prev_to_curr.get(index, set())):
                    queue.append(("c", neighbor))
            else:
                if index in c_group:
                    continue
                c_group.add(index)
                seen_curr.add(index)
                for neighbor in sorted(curr_to_prev.get(index, set())):
                    queue.append(("p", neighbor))
        if p_group or c_group:
            groups.append((p_group, c_group))
    return groups, edge_scores


def _transition_type(previous_count: int, current_count: int) -> WorldLineTransitionType:
    if previous_count == 1 and current_count == 1:
        return WorldLineTransitionType.CONTINUED
    if previous_count == 1 and current_count > 1:
        return WorldLineTransitionType.SPLIT
    if previous_count > 1 and current_count == 1:
        return WorldLineTransitionType.MERGED
    return WorldLineTransitionType.RECONFIGURED


def _timeline_digest(
    *,
    stream_id: str,
    policy: AttackWorldLinePolicy,
    observations: list[WorldLineComponentObservation],
    transitions: list[AttackWorldLineTransition],
) -> str:
    payload = {
        "stream_id": stream_id,
        "policy": policy.model_dump(mode="json"),
        "observations": [row.model_dump(mode="json") for row in observations],
        "transitions": [row.model_dump(mode="json") for row in transitions],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_attack_world_line_timeline(
    graph_snapshots: list[CyberStateGraph],
    *,
    stream_id: str,
    policy: AttackWorldLinePolicy | None = None,
) -> AttackWorldLineTimeline:
    if not graph_snapshots:
        raise ValueError("attack world-line timeline requires at least one graph snapshot")
    gate = policy or AttackWorldLinePolicy()
    if gate.entity_weight + gate.relation_weight <= 0.0:
        raise ValueError("attack world-line policy requires positive overlap weights")

    component_ticks = [
        analyze_attack_progression(graph).components for graph in graph_snapshots
    ]
    observations: list[WorldLineComponentObservation] = []
    transitions: list[AttackWorldLineTransition] = []
    previous_line_by_component: dict[str, str] = {}

    for component in component_ticks[0]:
        line_id = _new_world_line_id(
            stream_id=stream_id,
            tick=0,
            component_id=component.component_id,
            predecessors=[],
        )
        previous_line_by_component[component.component_id] = line_id
        observations.append(
            WorldLineComponentObservation(
                tick=0,
                world_line_id=line_id,
                component_id=component.component_id,
                entity_ids=component.entity_ids,
                relation_ids=component.active_relation_ids,
                current_stage=component.current_stage.value if component.current_stage else None,
                risk_score=component.risk_score,
            )
        )
        transitions.append(
            AttackWorldLineTransition(
                from_tick=None,
                to_tick=0,
                transition_type=WorldLineTransitionType.NEW,
                predecessor_world_line_ids=[],
                successor_world_line_ids=[line_id],
                predecessor_component_ids=[],
                successor_component_ids=[component.component_id],
                max_overlap_score=0.0,
            )
        )

    for tick in range(1, len(component_ticks)):
        previous = component_ticks[tick - 1]
        current = component_ticks[tick]
        groups, scores = _bipartite_groups(previous, current, gate)
        linked_prev = {index for group, _ in groups for index in group}
        linked_curr = {index for _, group in groups for index in group}
        current_line_by_component: dict[str, str] = {}

        for p_group, c_group in groups:
            predecessors = sorted(
                previous_line_by_component[previous[index].component_id]
                for index in p_group
            )
            transition_type = _transition_type(len(p_group), len(c_group))
            successors: list[str] = []
            successor_components: list[str] = []
            for c_index in sorted(c_group):
                component = current[c_index]
                if transition_type is WorldLineTransitionType.CONTINUED:
                    line_id = predecessors[0]
                else:
                    line_id = _new_world_line_id(
                        stream_id=stream_id,
                        tick=tick,
                        component_id=component.component_id,
                        predecessors=predecessors,
                    )
                current_line_by_component[component.component_id] = line_id
                successors.append(line_id)
                successor_components.append(component.component_id)
                observations.append(
                    WorldLineComponentObservation(
                        tick=tick,
                        world_line_id=line_id,
                        component_id=component.component_id,
                        entity_ids=component.entity_ids,
                        relation_ids=component.active_relation_ids,
                        current_stage=(
                            component.current_stage.value if component.current_stage else None
                        ),
                        risk_score=component.risk_score,
                    )
                )
            group_scores = [
                scores[(p_index, c_index)]
                for p_index in p_group
                for c_index in c_group
                if (p_index, c_index) in scores
            ]
            transitions.append(
                AttackWorldLineTransition(
                    from_tick=tick - 1,
                    to_tick=tick,
                    transition_type=transition_type,
                    predecessor_world_line_ids=predecessors,
                    successor_world_line_ids=sorted(successors),
                    predecessor_component_ids=sorted(
                        previous[index].component_id for index in p_group
                    ),
                    successor_component_ids=sorted(successor_components),
                    max_overlap_score=max(group_scores, default=0.0),
                )
            )

        for p_index, component in enumerate(previous):
            if p_index in linked_prev:
                continue
            line_id = previous_line_by_component[component.component_id]
            transitions.append(
                AttackWorldLineTransition(
                    from_tick=tick - 1,
                    to_tick=tick,
                    transition_type=WorldLineTransitionType.ENDED,
                    predecessor_world_line_ids=[line_id],
                    successor_world_line_ids=[],
                    predecessor_component_ids=[component.component_id],
                    successor_component_ids=[],
                    max_overlap_score=0.0,
                )
            )

        for c_index, component in enumerate(current):
            if c_index in linked_curr:
                continue
            line_id = _new_world_line_id(
                stream_id=stream_id,
                tick=tick,
                component_id=component.component_id,
                predecessors=[],
            )
            current_line_by_component[component.component_id] = line_id
            observations.append(
                WorldLineComponentObservation(
                    tick=tick,
                    world_line_id=line_id,
                    component_id=component.component_id,
                    entity_ids=component.entity_ids,
                    relation_ids=component.active_relation_ids,
                    current_stage=component.current_stage.value if component.current_stage else None,
                    risk_score=component.risk_score,
                )
            )
            transitions.append(
                AttackWorldLineTransition(
                    from_tick=tick - 1,
                    to_tick=tick,
                    transition_type=WorldLineTransitionType.NEW,
                    predecessor_world_line_ids=[],
                    successor_world_line_ids=[line_id],
                    predecessor_component_ids=[],
                    successor_component_ids=[component.component_id],
                    max_overlap_score=0.0,
                )
            )

        previous_line_by_component = current_line_by_component

    observations.sort(key=lambda row: (row.tick, row.world_line_id, row.component_id))
    transitions.sort(
        key=lambda row: (
            row.to_tick,
            row.transition_type.value,
            tuple(row.predecessor_world_line_ids),
            tuple(row.successor_world_line_ids),
        )
    )
    digest = _timeline_digest(
        stream_id=stream_id,
        policy=gate,
        observations=observations,
        transitions=transitions,
    )
    return AttackWorldLineTimeline(
        stream_id=stream_id,
        ticks=len(graph_snapshots),
        policy=gate,
        observations=observations,
        transitions=transitions,
        timeline_sha256=digest,
    )
