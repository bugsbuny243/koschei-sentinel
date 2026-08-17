from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from enum import StrEnum
from typing import Literal

from pydantic import Field

from koschei_sentinel.attack_progression import AttackComponentReport, analyze_attack_progression
from koschei_sentinel.cyber_state_graph import CyberStateGraph, EvidenceStatus
from koschei_sentinel.models import StrictModel


class WorldLineTransitionType(StrEnum):
    NEW = "NEW"
    CONTINUED = "CONTINUED"
    REROUTED = "REROUTED"
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
    protected_anchor_bonus: float = Field(default=0.35, ge=0.0, le=1.0)


class WorldLineComponentObservation(StrictModel):
    tick: int = Field(ge=0)
    world_line_id: str
    component_id: str
    entity_ids: list[str]
    relation_ids: list[str]
    protected_anchor_ids: list[str]
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
    shared_protected_anchor_ids: list[str]
    max_overlap_score: float = Field(ge=0.0, le=1.0)


class AttackWorldLineTimeline(StrictModel):
    schema_version: Literal["sentinel.attack-world-line-timeline.v1"] = (
        "sentinel.attack-world-line-timeline.v1"
    )
    stream_id: str = Field(min_length=3, max_length=256)
    ticks: int = Field(ge=1)
    protected_anchor_entity_ids: list[str]
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


def _component_protected_anchors(
    graph: CyberStateGraph,
    component: AttackComponentReport,
    protected_anchor_entity_ids: set[str],
) -> set[str]:
    if not protected_anchor_entity_ids:
        return set()
    component_entities = set(component.entity_ids)
    anchors = component_entities & protected_anchor_entity_ids
    for relation in graph.relations:
        if relation.status is EvidenceStatus.DISPROVED or relation.confidence < 0.5:
            continue
        if relation.source_entity_id in component_entities:
            if relation.target_entity_id in protected_anchor_entity_ids:
                anchors.add(relation.target_entity_id)
        if relation.target_entity_id in component_entities:
            if relation.source_entity_id in protected_anchor_entity_ids:
                anchors.add(relation.source_entity_id)
    return anchors


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
    previous_anchors: dict[str, set[str]],
    current_anchors: dict[str, set[str]],
    policy: AttackWorldLinePolicy,
) -> tuple[
    list[tuple[set[int], set[int]]],
    dict[tuple[int, int], float],
    set[tuple[int, int]],
]:
    edge_scores: dict[tuple[int, int], float] = {}
    anchor_only_edges: set[tuple[int, int]] = set()
    adjacency: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)

    for p_index, p_component in enumerate(previous):
        for c_index, c_component in enumerate(current):
            base_score = component_overlap_score(p_component, c_component, policy)
            shared_anchors = (
                previous_anchors.get(p_component.component_id, set())
                & current_anchors.get(c_component.component_id, set())
            )
            score = min(
                1.0,
                base_score + (policy.protected_anchor_bonus if shared_anchors else 0.0),
            )
            if score < policy.min_overlap_score:
                continue
            edge = (p_index, c_index)
            edge_scores[edge] = score
            if base_score < policy.min_overlap_score and shared_anchors:
                anchor_only_edges.add(edge)
            p_node = ("p", p_index)
            c_node = ("c", c_index)
            adjacency[p_node].add(c_node)
            adjacency[c_node].add(p_node)

    groups: list[tuple[set[int], set[int]]] = []
    seen: set[tuple[str, int]] = set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        p_group: set[int] = set()
        c_group: set[int] = set()
        queue: deque[tuple[str, int]] = deque([start])
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            side, index = node
            if side == "p":
                p_group.add(index)
            else:
                c_group.add(index)
            for neighbor in sorted(adjacency.get(node, set())):
                if neighbor not in seen:
                    queue.append(neighbor)
        groups.append((p_group, c_group))
    return groups, edge_scores, anchor_only_edges


def _transition_type(
    p_group: set[int],
    c_group: set[int],
    anchor_only_edges: set[tuple[int, int]],
) -> WorldLineTransitionType:
    if len(p_group) == 1 and len(c_group) == 1:
        edge = (next(iter(p_group)), next(iter(c_group)))
        if edge in anchor_only_edges:
            return WorldLineTransitionType.REROUTED
        return WorldLineTransitionType.CONTINUED
    if len(p_group) == 1 and len(c_group) > 1:
        return WorldLineTransitionType.SPLIT
    if len(p_group) > 1 and len(c_group) == 1:
        return WorldLineTransitionType.MERGED
    return WorldLineTransitionType.RECONFIGURED


def _timeline_digest(
    *,
    stream_id: str,
    protected_anchor_entity_ids: list[str],
    policy: AttackWorldLinePolicy,
    observations: list[WorldLineComponentObservation],
    transitions: list[AttackWorldLineTransition],
) -> str:
    payload = {
        "stream_id": stream_id,
        "protected_anchor_entity_ids": protected_anchor_entity_ids,
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
    protected_anchor_entity_ids: list[str] | None = None,
    policy: AttackWorldLinePolicy | None = None,
) -> AttackWorldLineTimeline:
    if not graph_snapshots:
        raise ValueError("attack world-line timeline requires at least one graph snapshot")
    gate = policy or AttackWorldLinePolicy()
    if gate.entity_weight + gate.relation_weight <= 0.0:
        raise ValueError("attack world-line policy requires positive overlap weights")
    anchors = set(protected_anchor_entity_ids or [])

    component_ticks = [
        analyze_attack_progression(graph).components for graph in graph_snapshots
    ]
    anchor_ticks = [
        {
            component.component_id: _component_protected_anchors(graph, component, anchors)
            for component in components
        }
        for graph, components in zip(graph_snapshots, component_ticks, strict=True)
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
        component_anchors = sorted(anchor_ticks[0].get(component.component_id, set()))
        observations.append(
            WorldLineComponentObservation(
                tick=0,
                world_line_id=line_id,
                component_id=component.component_id,
                entity_ids=component.entity_ids,
                relation_ids=component.active_relation_ids,
                protected_anchor_ids=component_anchors,
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
                shared_protected_anchor_ids=component_anchors,
                max_overlap_score=0.0,
            )
        )

    for tick in range(1, len(component_ticks)):
        previous = component_ticks[tick - 1]
        current = component_ticks[tick]
        groups, scores, anchor_only_edges = _bipartite_groups(
            previous,
            current,
            anchor_ticks[tick - 1],
            anchor_ticks[tick],
            gate,
        )
        linked_prev = {index for group, _ in groups for index in group}
        linked_curr = {index for _, group in groups for index in group}
        current_line_by_component: dict[str, str] = {}

        for p_group, c_group in groups:
            predecessors = sorted(
                previous_line_by_component[previous[index].component_id]
                for index in p_group
            )
            transition_type = _transition_type(p_group, c_group, anchor_only_edges)
            successors: list[str] = []
            successor_components: list[str] = []
            group_shared_anchors: set[str] = set()
            for p_index in p_group:
                for c_index in c_group:
                    group_shared_anchors.update(
                        anchor_ticks[tick - 1].get(previous[p_index].component_id, set())
                        & anchor_ticks[tick].get(current[c_index].component_id, set())
                    )
            for c_index in sorted(c_group):
                component = current[c_index]
                if transition_type in {
                    WorldLineTransitionType.CONTINUED,
                    WorldLineTransitionType.REROUTED,
                }:
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
                        protected_anchor_ids=sorted(
                            anchor_ticks[tick].get(component.component_id, set())
                        ),
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
                    shared_protected_anchor_ids=sorted(group_shared_anchors),
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
                    shared_protected_anchor_ids=[],
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
            component_anchors = sorted(anchor_ticks[tick].get(component.component_id, set()))
            observations.append(
                WorldLineComponentObservation(
                    tick=tick,
                    world_line_id=line_id,
                    component_id=component.component_id,
                    entity_ids=component.entity_ids,
                    relation_ids=component.active_relation_ids,
                    protected_anchor_ids=component_anchors,
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
                    shared_protected_anchor_ids=component_anchors,
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
    sorted_anchors = sorted(anchors)
    digest = _timeline_digest(
        stream_id=stream_id,
        protected_anchor_entity_ids=sorted_anchors,
        policy=gate,
        observations=observations,
        transitions=transitions,
    )
    return AttackWorldLineTimeline(
        stream_id=stream_id,
        ticks=len(graph_snapshots),
        protected_anchor_entity_ids=sorted_anchors,
        policy=gate,
        observations=observations,
        transitions=transitions,
        timeline_sha256=digest,
    )
