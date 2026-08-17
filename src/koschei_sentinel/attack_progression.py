from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from enum import StrEnum
from typing import Literal

from pydantic import Field

from koschei_sentinel.cyber_state_graph import (
    CyberEntity,
    CyberEntityType,
    CyberRelation,
    CyberStateGraph,
    EvidenceStatus,
)
from koschei_sentinel.defense_authority import DefenseActionType
from koschei_sentinel.models import StrictModel


class AttackStage(StrEnum):
    RECONNAISSANCE = "RECONNAISSANCE"
    INITIAL_ACCESS = "INITIAL_ACCESS"
    EXECUTION = "EXECUTION"
    PERSISTENCE = "PERSISTENCE"
    PRIVILEGE_ESCALATION = "PRIVILEGE_ESCALATION"
    CREDENTIAL_ACCESS = "CREDENTIAL_ACCESS"
    DISCOVERY = "DISCOVERY"
    LATERAL_MOVEMENT = "LATERAL_MOVEMENT"
    COLLECTION = "COLLECTION"
    COMMAND_AND_CONTROL = "COMMAND_AND_CONTROL"
    SUPPLY_CHAIN = "SUPPLY_CHAIN"
    SIGNER_OR_WALLET_ACCESS = "SIGNER_OR_WALLET_ACCESS"
    EXFILTRATION = "EXFILTRATION"
    IMPACT = "IMPACT"


_STAGE_ORDER = [
    AttackStage.RECONNAISSANCE,
    AttackStage.INITIAL_ACCESS,
    AttackStage.EXECUTION,
    AttackStage.PERSISTENCE,
    AttackStage.PRIVILEGE_ESCALATION,
    AttackStage.CREDENTIAL_ACCESS,
    AttackStage.DISCOVERY,
    AttackStage.LATERAL_MOVEMENT,
    AttackStage.COLLECTION,
    AttackStage.COMMAND_AND_CONTROL,
    AttackStage.SUPPLY_CHAIN,
    AttackStage.SIGNER_OR_WALLET_ACCESS,
    AttackStage.EXFILTRATION,
    AttackStage.IMPACT,
]
_STAGE_RANK = {stage: index for index, stage in enumerate(_STAGE_ORDER)}

_STAGE_SUCCESSORS: dict[AttackStage, tuple[AttackStage, ...]] = {
    AttackStage.RECONNAISSANCE: (AttackStage.INITIAL_ACCESS,),
    AttackStage.INITIAL_ACCESS: (AttackStage.EXECUTION, AttackStage.CREDENTIAL_ACCESS),
    AttackStage.EXECUTION: (AttackStage.PERSISTENCE, AttackStage.DISCOVERY),
    AttackStage.PERSISTENCE: (AttackStage.PRIVILEGE_ESCALATION, AttackStage.CREDENTIAL_ACCESS),
    AttackStage.PRIVILEGE_ESCALATION: (AttackStage.CREDENTIAL_ACCESS, AttackStage.LATERAL_MOVEMENT),
    AttackStage.CREDENTIAL_ACCESS: (AttackStage.DISCOVERY, AttackStage.LATERAL_MOVEMENT),
    AttackStage.DISCOVERY: (AttackStage.LATERAL_MOVEMENT, AttackStage.COLLECTION),
    AttackStage.LATERAL_MOVEMENT: (
        AttackStage.COLLECTION,
        AttackStage.SUPPLY_CHAIN,
        AttackStage.SIGNER_OR_WALLET_ACCESS,
    ),
    AttackStage.COLLECTION: (AttackStage.EXFILTRATION, AttackStage.IMPACT),
    AttackStage.COMMAND_AND_CONTROL: (AttackStage.LATERAL_MOVEMENT, AttackStage.IMPACT),
    AttackStage.SUPPLY_CHAIN: (AttackStage.SIGNER_OR_WALLET_ACCESS, AttackStage.IMPACT),
    AttackStage.SIGNER_OR_WALLET_ACCESS: (AttackStage.IMPACT,),
    AttackStage.EXFILTRATION: (AttackStage.IMPACT,),
    AttackStage.IMPACT: (),
}

_RELATION_STAGE_HINTS: dict[str, AttackStage] = {
    "scans": AttackStage.RECONNAISSANCE,
    "probes": AttackStage.RECONNAISSANCE,
    "initial_access": AttackStage.INITIAL_ACCESS,
    "authenticates_to": AttackStage.INITIAL_ACCESS,
    "executes": AttackStage.EXECUTION,
    "spawns": AttackStage.EXECUTION,
    "persists_via": AttackStage.PERSISTENCE,
    "creates_persistence": AttackStage.PERSISTENCE,
    "elevates_to": AttackStage.PRIVILEGE_ESCALATION,
    "assumes_privilege": AttackStage.PRIVILEGE_ESCALATION,
    "steals_credential": AttackStage.CREDENTIAL_ACCESS,
    "uses_credential": AttackStage.CREDENTIAL_ACCESS,
    "reads_secret": AttackStage.CREDENTIAL_ACCESS,
    "discovers": AttackStage.DISCOVERY,
    "enumerates": AttackStage.DISCOVERY,
    "moves_to": AttackStage.LATERAL_MOVEMENT,
    "remote_executes": AttackStage.LATERAL_MOVEMENT,
    "collects": AttackStage.COLLECTION,
    "stages_data": AttackStage.COLLECTION,
    "beacons_to": AttackStage.COMMAND_AND_CONTROL,
    "controlled_by": AttackStage.COMMAND_AND_CONTROL,
    "modifies_repository": AttackStage.SUPPLY_CHAIN,
    "modifies_pipeline": AttackStage.SUPPLY_CHAIN,
    "modifies_artifact": AttackStage.SUPPLY_CHAIN,
    "reaches_signer": AttackStage.SIGNER_OR_WALLET_ACCESS,
    "uses_wallet": AttackStage.SIGNER_OR_WALLET_ACCESS,
    "prepares_transaction": AttackStage.SIGNER_OR_WALLET_ACCESS,
    "exfiltrates_to": AttackStage.EXFILTRATION,
    "encrypts": AttackStage.IMPACT,
    "destroys": AttackStage.IMPACT,
    "drains": AttackStage.IMPACT,
    "executes_transaction": AttackStage.IMPACT,
}


class StageEvidence(StrictModel):
    stage: AttackStage
    relation_ids: list[str]
    score: float = Field(ge=0.0, le=1.0)
    observed_relations: int = Field(ge=0)
    inferred_relations: int = Field(ge=0)


class PredictedTransition(StrictModel):
    from_stage: AttackStage
    to_stage: AttackStage
    probability: float = Field(ge=0.0, le=1.0)
    rationale: str


class DefensiveCutPoint(StrictModel):
    entity_id: str
    entity_type: CyberEntityType
    action: DefenseActionType
    effect_score: float = Field(ge=0.0, le=1.0)
    downstream_entities: int = Field(ge=0)
    supporting_relation_ids: list[str]
    rationale: str


class AttackComponentReport(StrictModel):
    schema_version: Literal["sentinel.attack-component.v1"] = "sentinel.attack-component.v1"
    component_id: str
    entity_ids: list[str]
    relation_ids: list[str]
    active_stages: list[StageEvidence]
    current_stage: AttackStage | None
    progression_confidence: float = Field(ge=0.0, le=1.0)
    predicted_transitions: list[PredictedTransition]
    defensive_cut_points: list[DefensiveCutPoint]
    active_relation_ids: list[str]
    disproved_relation_ids: list[str]
    risk_score: float = Field(ge=0.0, le=1.0)


class AttackProgressionReport(StrictModel):
    schema_version: Literal["sentinel.attack-progression.v1"] = (
        "sentinel.attack-progression.v1"
    )
    graph_id: str
    active_stages: list[StageEvidence]
    current_stage: AttackStage | None
    progression_confidence: float = Field(ge=0.0, le=1.0)
    predicted_transitions: list[PredictedTransition]
    defensive_cut_points: list[DefensiveCutPoint]
    active_relation_ids: list[str]
    disproved_relation_ids: list[str]
    primary_component_id: str | None = None
    components: list[AttackComponentReport] = Field(default_factory=list)


def _relation_stage(relation: CyberRelation) -> AttackStage | None:
    normalized = relation.relation_type.strip().lower().replace("-", "_").replace(" ", "_")
    direct = _RELATION_STAGE_HINTS.get(normalized)
    if direct is not None:
        return direct
    for hint, stage in _RELATION_STAGE_HINTS.items():
        if hint in normalized:
            return stage
    return None


def _relation_weight(relation: CyberRelation) -> float:
    if relation.status is EvidenceStatus.OBSERVED:
        status_weight = 1.0
    elif relation.status is EvidenceStatus.INFERRED:
        status_weight = 0.65
    elif relation.status is EvidenceStatus.PREDICTED:
        status_weight = 0.25
    else:
        return 0.0
    evidence_bonus = min(0.15, 0.03 * len(relation.evidence))
    return min(1.0, relation.confidence * status_weight + evidence_bonus)


def _stage_evidence(graph: CyberStateGraph) -> list[StageEvidence]:
    grouped: dict[AttackStage, list[CyberRelation]] = defaultdict(list)
    for relation in graph.relations:
        if relation.status is EvidenceStatus.DISPROVED:
            continue
        stage = _relation_stage(relation)
        if stage is not None:
            grouped[stage].append(relation)

    output: list[StageEvidence] = []
    for stage in _STAGE_ORDER:
        relations = grouped.get(stage, [])
        if not relations:
            continue
        weights = [_relation_weight(relation) for relation in relations]
        strongest = max(weights)
        corroboration = min(0.2, 0.04 * max(0, len(relations) - 1))
        score = min(1.0, strongest + corroboration)
        output.append(
            StageEvidence(
                stage=stage,
                relation_ids=sorted(relation.relation_id for relation in relations),
                score=score,
                observed_relations=sum(
                    relation.status is EvidenceStatus.OBSERVED for relation in relations
                ),
                inferred_relations=sum(
                    relation.status is EvidenceStatus.INFERRED for relation in relations
                ),
            )
        )
    return output


def _current_stage(stages: list[StageEvidence]) -> AttackStage | None:
    credible = [row for row in stages if row.score >= 0.5]
    if not credible:
        return None
    return max(credible, key=lambda row: (_STAGE_RANK[row.stage], row.score)).stage


def _predictions(current: AttackStage | None, stages: list[StageEvidence]) -> list[PredictedTransition]:
    if current is None:
        return []
    present = {row.stage for row in stages if row.score >= 0.5}
    successors = [stage for stage in _STAGE_SUCCESSORS[current] if stage not in present]
    if not successors:
        return []
    base = 0.72 if len(successors) == 1 else 0.58
    return [
        PredictedTransition(
            from_stage=current,
            to_stage=stage,
            probability=max(0.25, base - index * 0.08),
            rationale="deterministic progression prior; prediction is not treated as observed evidence",
        )
        for index, stage in enumerate(successors[:3])
    ]


def _active_relations(graph: CyberStateGraph) -> list[CyberRelation]:
    return [
        relation
        for relation in graph.relations
        if relation.status in {EvidenceStatus.OBSERVED, EvidenceStatus.INFERRED}
        and relation.confidence >= 0.5
    ]


def _action_for_entity(entity: CyberEntity) -> DefenseActionType | None:
    return {
        CyberEntityType.CREDENTIAL: DefenseActionType.REVOKE_CREDENTIAL,
        CyberEntityType.DEVICE: DefenseActionType.ISOLATE_ENDPOINT,
        CyberEntityType.PROCESS: DefenseActionType.KILL_PROCESS,
        CyberEntityType.PIPELINE: DefenseActionType.PAUSE_PIPELINE,
        CyberEntityType.CLOUD_RESOURCE: DefenseActionType.QUARANTINE_WORKLOAD,
        CyberEntityType.WALLET: DefenseActionType.FREEZE_SIGNER,
        CyberEntityType.TRANSACTION: DefenseActionType.HOLD_TRANSACTION,
        CyberEntityType.NETWORK_ENDPOINT: DefenseActionType.BLOCK_IOC,
    }.get(entity.entity_type)


def _reachable_count(start: str, adjacency: dict[str, set[str]]) -> int:
    seen: set[str] = {start}
    queue: deque[str] = deque([start])
    while queue:
        current = queue.popleft()
        for target in adjacency.get(current, set()):
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return max(0, len(seen) - 1)


def _cut_points(graph: CyberStateGraph) -> list[DefensiveCutPoint]:
    entities = {entity.entity_id: entity for entity in graph.entities}
    active = _active_relations(graph)
    adjacency: dict[str, set[str]] = defaultdict(set)
    relations_by_entity: dict[str, list[CyberRelation]] = defaultdict(list)
    for relation in active:
        adjacency[relation.source_entity_id].add(relation.target_entity_id)
        relations_by_entity[relation.source_entity_id].append(relation)
        relations_by_entity[relation.target_entity_id].append(relation)

    total = max(1, len(entities) - 1)
    candidates: list[DefensiveCutPoint] = []
    for entity in graph.entities:
        action = _action_for_entity(entity)
        if action is None:
            continue
        touching = relations_by_entity.get(entity.entity_id, [])
        if not touching:
            continue
        downstream = _reachable_count(entity.entity_id, adjacency)
        max_confidence = max(relation.confidence for relation in touching)
        observed_bonus = 0.15 if any(
            relation.status is EvidenceStatus.OBSERVED for relation in touching
        ) else 0.0
        centrality = min(1.0, downstream / total)
        effect = min(1.0, 0.5 * centrality + 0.35 * max_confidence + observed_bonus)
        candidates.append(
            DefensiveCutPoint(
                entity_id=entity.entity_id,
                entity_type=entity.entity_type,
                action=action,
                effect_score=effect,
                downstream_entities=downstream,
                supporting_relation_ids=sorted(
                    relation.relation_id for relation in touching
                ),
                rationale=(
                    "defensive control point inside one attack component; score combines "
                    "component-local downstream reach, relation confidence, and observed-evidence support"
                ),
            )
        )
    candidates.sort(key=lambda row: (-row.effect_score, row.entity_id))
    return candidates[:8]


def _active_components(graph: CyberStateGraph) -> list[set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for relation in _active_relations(graph):
        adjacency[relation.source_entity_id].add(relation.target_entity_id)
        adjacency[relation.target_entity_id].add(relation.source_entity_id)

    remaining = set(adjacency)
    components: list[set[str]] = []
    while remaining:
        start = min(remaining)
        seen = {start}
        queue: deque[str] = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in sorted(adjacency.get(current, set())):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        remaining -= seen
        components.append(seen)
    components.sort(key=lambda row: tuple(sorted(row)))
    return components


def _component_id(entity_ids: set[str], active_relation_ids: list[str]) -> str:
    payload = "|".join([",".join(sorted(entity_ids)), ",".join(sorted(active_relation_ids))])
    return "attack-component:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _component_subgraph(graph: CyberStateGraph, entity_ids: set[str]) -> CyberStateGraph:
    return CyberStateGraph(
        graph_id=f"{graph.graph_id}:component",
        entities=[entity for entity in graph.entities if entity.entity_id in entity_ids],
        relations=[
            relation
            for relation in graph.relations
            if relation.source_entity_id in entity_ids and relation.target_entity_id in entity_ids
        ],
    )


def _component_risk_score(
    *,
    current_stage: AttackStage | None,
    confidence: float,
    cut_points: list[DefensiveCutPoint],
    entity_count: int,
) -> float:
    stage_score = (
        0.0
        if current_stage is None
        else _STAGE_RANK[current_stage] / max(1, len(_STAGE_ORDER) - 1)
    )
    cut_score = max((row.effect_score for row in cut_points), default=0.0)
    span_score = min(1.0, entity_count / 8.0)
    return min(1.0, 0.45 * stage_score + 0.35 * confidence + 0.15 * cut_score + 0.05 * span_score)


def _analyze_component(graph: CyberStateGraph, entity_ids: set[str]) -> AttackComponentReport:
    subgraph = _component_subgraph(graph, entity_ids)
    stages = _stage_evidence(subgraph)
    current = _current_stage(stages)
    confidence = max((row.score for row in stages), default=0.0)
    active = _active_relations(subgraph)
    active_ids = sorted(relation.relation_id for relation in active)
    cut_points = _cut_points(subgraph)
    relation_ids = sorted(relation.relation_id for relation in subgraph.relations)
    disproved = sorted(
        relation.relation_id
        for relation in subgraph.relations
        if relation.status is EvidenceStatus.DISPROVED
    )
    return AttackComponentReport(
        component_id=_component_id(entity_ids, active_ids),
        entity_ids=sorted(entity_ids),
        relation_ids=relation_ids,
        active_stages=stages,
        current_stage=current,
        progression_confidence=confidence,
        predicted_transitions=_predictions(current, stages),
        defensive_cut_points=cut_points,
        active_relation_ids=active_ids,
        disproved_relation_ids=disproved,
        risk_score=_component_risk_score(
            current_stage=current,
            confidence=confidence,
            cut_points=cut_points,
            entity_count=len(entity_ids),
        ),
    )


def _primary_component(
    components: list[AttackComponentReport],
    focus_entity_ids: set[str],
) -> AttackComponentReport | None:
    if not components:
        return None
    focused = [
        component
        for component in components
        if focus_entity_ids.intersection(component.entity_ids)
    ]
    candidates = focused or components
    return max(
        candidates,
        key=lambda row: (
            row.risk_score,
            _STAGE_RANK.get(row.current_stage, -1),
            row.progression_confidence,
            row.component_id,
        ),
    )


def analyze_attack_progression(
    graph: CyberStateGraph,
    *,
    focus_entity_ids: list[str] | None = None,
) -> AttackProgressionReport:
    components = [_analyze_component(graph, ids) for ids in _active_components(graph)]
    components.sort(key=lambda row: (-row.risk_score, row.component_id))
    primary = _primary_component(components, set(focus_entity_ids or []))

    if primary is None:
        return AttackProgressionReport(
            graph_id=graph.graph_id,
            active_stages=[],
            current_stage=None,
            progression_confidence=0.0,
            predicted_transitions=[],
            defensive_cut_points=[],
            active_relation_ids=[],
            disproved_relation_ids=sorted(
                relation.relation_id
                for relation in graph.relations
                if relation.status is EvidenceStatus.DISPROVED
            ),
            primary_component_id=None,
            components=[],
        )

    return AttackProgressionReport(
        graph_id=graph.graph_id,
        active_stages=primary.active_stages,
        current_stage=primary.current_stage,
        progression_confidence=primary.progression_confidence,
        predicted_transitions=primary.predicted_transitions,
        defensive_cut_points=primary.defensive_cut_points,
        active_relation_ids=primary.active_relation_ids,
        disproved_relation_ids=primary.disproved_relation_ids,
        primary_component_id=primary.component_id,
        components=components,
    )
