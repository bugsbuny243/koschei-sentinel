from koschei_sentinel.attack_world_lines import (
    WorldLineTransitionType,
    build_attack_world_line_timeline,
)
from koschei_sentinel.cyber_state_graph import (
    CyberEntity,
    CyberEntityType,
    CyberRelation,
    CyberStateGraph,
    EvidenceRef,
    EvidenceStatus,
)


SHA = "d" * 64


def _evidence(name: str) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=f"evidence:{name}",
        source="ENDPOINT:edr:test",
        content_sha256=SHA,
    )


def _graph(graph_id: str, edges: list[tuple[str, str, str]]) -> CyberStateGraph:
    entity_ids = sorted({value for source, target, _ in edges for value in (source, target)})
    entities = [
        CyberEntity(
            entity_id=entity_id,
            entity_type=(
                CyberEntityType.DEVICE
                if entity_id.startswith("device:")
                else CyberEntityType.PROCESS
            ),
        )
        for entity_id in entity_ids
    ]
    relations = [
        CyberRelation(
            relation_id=relation_id,
            source_entity_id=source,
            target_entity_id=target,
            relation_type="executes",
            status=EvidenceStatus.OBSERVED,
            confidence=0.95,
            evidence=[_evidence(relation_id)],
        )
        for source, target, relation_id in edges
    ]
    return CyberStateGraph(graph_id=graph_id, entities=entities, relations=relations)


def test_growing_component_keeps_the_same_world_line() -> None:
    first = _graph(
        "incident:timeline",
        [("device:a", "process:b", "r1")],
    )
    second = _graph(
        "incident:timeline",
        [
            ("device:a", "process:b", "r1"),
            ("process:b", "process:c", "r2"),
        ],
    )

    timeline = build_attack_world_line_timeline(
        [first, second],
        stream_id="stream:continued",
    )

    observations = sorted(timeline.observations, key=lambda row: row.tick)
    assert len(observations) == 2
    assert observations[0].world_line_id == observations[1].world_line_id
    assert any(
        row.transition_type is WorldLineTransitionType.CONTINUED
        for row in timeline.transitions
    )


def test_one_attack_component_can_split_into_two_tracked_world_lines() -> None:
    first = _graph(
        "incident:split",
        [
            ("device:a", "process:b", "r1"),
            ("process:b", "process:c", "r2"),
            ("process:c", "process:d", "r3"),
        ],
    )
    second = _graph(
        "incident:split",
        [
            ("device:a", "process:b", "r1"),
            ("process:c", "process:d", "r3"),
        ],
    )

    timeline = build_attack_world_line_timeline(
        [first, second],
        stream_id="stream:split",
    )

    split = next(
        row
        for row in timeline.transitions
        if row.transition_type is WorldLineTransitionType.SPLIT
    )
    assert len(split.predecessor_world_line_ids) == 1
    assert len(split.successor_world_line_ids) == 2
    assert split.predecessor_world_line_ids[0] not in split.successor_world_line_ids


def test_two_world_lines_can_merge_into_one_new_lineage() -> None:
    first = _graph(
        "incident:merge",
        [
            ("device:a", "process:b", "r1"),
            ("process:c", "process:d", "r3"),
        ],
    )
    second = _graph(
        "incident:merge",
        [
            ("device:a", "process:b", "r1"),
            ("process:b", "process:c", "r2"),
            ("process:c", "process:d", "r3"),
        ],
    )

    timeline = build_attack_world_line_timeline(
        [first, second],
        stream_id="stream:merge",
    )

    merged = next(
        row
        for row in timeline.transitions
        if row.transition_type is WorldLineTransitionType.MERGED
    )
    assert len(merged.predecessor_world_line_ids) == 2
    assert len(merged.successor_world_line_ids) == 1
    assert merged.successor_world_line_ids[0] not in merged.predecessor_world_line_ids


def test_world_line_timeline_digest_is_deterministic() -> None:
    snapshots = [
        _graph("incident:stable", [("device:a", "process:b", "r1")]),
        _graph(
            "incident:stable",
            [
                ("device:a", "process:b", "r1"),
                ("process:b", "process:c", "r2"),
            ],
        ),
    ]

    first = build_attack_world_line_timeline(snapshots, stream_id="stream:stable")
    second = build_attack_world_line_timeline(snapshots, stream_id="stream:stable")

    assert first.timeline_sha256 == second.timeline_sha256
    assert first.model_dump() == second.model_dump()
