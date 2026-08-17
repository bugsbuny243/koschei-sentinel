from koschei_sentinel.assured_multi_incident_defense import (
    build_assured_multi_incident_defense_plan,
)
from koschei_sentinel.cyber_perception import TelemetrySourceType
from koschei_sentinel.cyber_state_graph import CyberEntityType
from koschei_sentinel.defense_resource_scheduler import (
    DefenseResourceClass,
    DefenseResourcePolicy,
    DefenseSchedulerState,
    SchedulingDisposition,
    build_defense_resource_schedule,
)
from koschei_sentinel.perception_adapter_sdk import (
    DeclarativeAdapterSpec,
    DeclarativePerceptionAdapter,
    EntityMappingSpec,
    PerceptionAdapterDescriptor,
    SanitizedTelemetryEvent,
    run_perception_adapter_batch,
)
from koschei_sentinel.perception_assurance import fuse_admitted_perception_batches
from koschei_sentinel.perception_graph_binding import compile_bound_perception_graph
from koschei_sentinel.perception_source_registry import (
    PerceptionSourceEnrollment,
    PerceptionSourceRegistry,
    admit_perception_adapter_result,
)


def _adapter(
    adapter_id: str,
    source_type: TelemetrySourceType,
    relation_type: str,
    source_field: str,
    source_entity_type: CyberEntityType,
    target_field: str,
    target_entity_type: CyberEntityType,
) -> DeclarativePerceptionAdapter:
    return DeclarativePerceptionAdapter(
        DeclarativeAdapterSpec(
            descriptor=PerceptionAdapterDescriptor(
                adapter_id=adapter_id,
                vendor="test",
                product="scheduler",
                adapter_version="1",
                supported_source_types=[source_type],
            ),
            source=EntityMappingSpec(id_field=source_field, entity_type=source_entity_type),
            target=EntityMappingSpec(id_field=target_field, entity_type=target_entity_type),
            relation_type=relation_type,
        )
    )


def _result(
    *,
    suffix: str,
    source_type: TelemetrySourceType,
    source_instance: str,
    relation_type: str,
    source_field: str,
    source_entity_type: CyberEntityType,
    target_field: str,
    target_entity_type: CyberEntityType,
    fields: dict[str, str],
    payload_char: str,
):
    adapter_id = f"scheduler.{suffix}"
    adapter = _adapter(
        adapter_id,
        source_type,
        relation_type,
        source_field,
        source_entity_type,
        target_field,
        target_entity_type,
    )
    event = SanitizedTelemetryEvent(
        event_id=f"event:{suffix}",
        source_type=source_type,
        source_instance=source_instance,
        payload_sha256=payload_char * 64,
        fields=fields,
    )
    return adapter_id, run_perception_adapter_batch(
        adapter,
        [event],
        batch_id=f"batch:{suffix}",
    )


def _component(prefix: str, chars: tuple[str, str, str]):
    return [
        _result(
            suffix=f"{prefix}-credential",
            source_type=TelemetrySourceType.ENDPOINT,
            source_instance=f"sensor:{prefix}:endpoint",
            relation_type="uses_credential",
            source_field="credential_id",
            source_entity_type=CyberEntityType.CREDENTIAL,
            target_field="device_id",
            target_entity_type=CyberEntityType.DEVICE,
            fields={"credential_id": f"credential:{prefix}", "device_id": f"device:{prefix}"},
            payload_char=chars[0],
        ),
        _result(
            suffix=f"{prefix}-pipeline",
            source_type=TelemetrySourceType.CICD,
            source_instance=f"sensor:{prefix}:cicd",
            relation_type="modifies_pipeline",
            source_field="device_id",
            source_entity_type=CyberEntityType.DEVICE,
            target_field="pipeline_id",
            target_entity_type=CyberEntityType.PIPELINE,
            fields={"device_id": f"device:{prefix}", "pipeline_id": f"pipeline:{prefix}"},
            payload_char=chars[1],
        ),
        _result(
            suffix=f"{prefix}-signer",
            source_type=TelemetrySourceType.SIGNER_WALLET,
            source_instance=f"sensor:{prefix}:signer",
            relation_type="reaches_signer",
            source_field="pipeline_id",
            source_entity_type=CyberEntityType.PIPELINE,
            target_field="wallet_id",
            target_entity_type=CyberEntityType.WALLET,
            fields={"pipeline_id": f"pipeline:{prefix}", "wallet_id": f"wallet:{prefix}"},
            payload_char=chars[2],
        ),
    ]


def _plan(*, critical: list[str], b_single_domain: bool = False):
    component_a = _component("a", ("a", "b", "c"))
    component_b = _component("b", ("d", "e", "f"))
    rows = component_a + component_b
    enrollments = []
    source_types = [
        TelemetrySourceType.ENDPOINT,
        TelemetrySourceType.CICD,
        TelemetrySourceType.SIGNER_WALLET,
    ]
    for prefix, component_rows in (("a", component_a), ("b", component_b)):
        for index, ((adapter_id, _), source_type) in enumerate(
            zip(component_rows, source_types, strict=True)
        ):
            domain = (
                "b-single-domain"
                if prefix == "b" and b_single_domain
                else f"{prefix}-domain-{index}"
            )
            enrollments.append(
                PerceptionSourceEnrollment(
                    source_type=source_type,
                    source_instance=f"sensor:{prefix}:{['endpoint', 'cicd', 'signer'][index]}",
                    allowed_adapter_ids=[adapter_id],
                    independence_domain=domain,
                )
            )
    registry = PerceptionSourceRegistry(
        registry_id="registry:scheduler",
        enrollments=enrollments,
    )
    admitted = [admit_perception_adapter_result(result, registry) for _, result in rows]
    fused = fuse_admitted_perception_batches(
        admitted,
        registry=registry,
        fused_batch_id="fusion:scheduler",
    )
    bound = compile_bound_perception_graph(
        fused.perception_batch,
        graph_id="incident:scheduler",
    )
    return build_assured_multi_incident_defense_plan(
        bound.graph,
        graph_receipt=bound.receipt,
        perception_assurance=fused.assurance,
        registry=registry,
        critical_entity_ids=critical,
    )


def _policy(*, total: int, signer: int, reserved: int = 0) -> DefenseResourcePolicy:
    capacities = DefenseResourcePolicy().resource_capacities.copy()
    capacities[DefenseResourceClass.SIGNER] = signer
    return DefenseResourcePolicy(
        max_parallel_total=total,
        reserved_critical_slots=reserved,
        resource_capacities=capacities,
    )


def test_reserved_critical_slot_prioritizes_critical_signer_component() -> None:
    plan = _plan(critical=["wallet:a"])
    schedule = build_defense_resource_schedule(
        plan,
        policy=_policy(total=1, signer=1, reserved=1),
    )

    assert len(schedule.scheduled) == 1
    assert "wallet:a" in schedule.scheduled[0].critical_entity_ids
    assert schedule.scheduled[0].resource_class is DefenseResourceClass.SIGNER
    assert len(schedule.deferred) == 1
    assert schedule.deferred[0].disposition is SchedulingDisposition.DEFERRED_GLOBAL_CAPACITY


def test_signer_capacity_prevents_control_plane_overcommit() -> None:
    plan = _plan(critical=["wallet:a", "wallet:b"])
    schedule = build_defense_resource_schedule(
        plan,
        policy=_policy(total=2, signer=1, reserved=0),
    )

    assert len(schedule.scheduled) == 1
    assert schedule.resource_usage[DefenseResourceClass.SIGNER] == 1
    assert len(schedule.deferred) == 1
    assert schedule.deferred[0].disposition is SchedulingDisposition.DEFERRED_RESOURCE_CAPACITY


def test_deferred_component_ages_and_wins_next_equal_capacity_wave() -> None:
    plan = _plan(critical=[])
    policy = _policy(total=1, signer=1, reserved=0)
    first = build_defense_resource_schedule(plan, policy=policy)
    assert len(first.scheduled) == 1
    assert len(first.deferred) == 1
    deferred_component = first.deferred[0].component_id

    second = build_defense_resource_schedule(
        plan,
        policy=policy,
        state=first.next_state,
    )
    assert second.scheduled[0].component_id == deferred_component
    assert second.scheduled[0].wait_cycles == 1
    assert second.next_state.wait_cycles_by_component[deferred_component] == 0


def test_guard_component_without_authorized_cut_point_consumes_no_resource_slot() -> None:
    plan = _plan(critical=["wallet:a", "wallet:b"], b_single_domain=True)
    schedule = build_defense_resource_schedule(
        plan,
        policy=_policy(total=2, signer=2, reserved=1),
    )

    guard_components = {
        row.component_id
        for row in plan.component_plans
        if row.assured_plan.assurance.effective_mode.value == "GUARD"
    }
    assert guard_components
    assert guard_components.issubset(set(schedule.no_action_component_ids))
    assert not guard_components & {row.component_id for row in schedule.scheduled}


def test_schedule_is_deterministic_for_same_plan_policy_and_state() -> None:
    plan = _plan(critical=["wallet:a", "wallet:b"])
    policy = _policy(total=1, signer=1, reserved=1)
    state = DefenseSchedulerState(wait_cycles_by_component={})
    first = build_defense_resource_schedule(plan, policy=policy, state=state)
    second = build_defense_resource_schedule(plan, policy=policy, state=state)

    assert first.model_dump() == second.model_dump()
    assert first.schedule_sha256 == second.schedule_sha256
