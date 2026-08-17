from koschei_sentinel.assured_multi_incident_defense import (
    build_assured_multi_incident_defense_plan,
)
from koschei_sentinel.cyber_perception import TelemetrySourceType
from koschei_sentinel.cyber_state_graph import CyberEntityType
from koschei_sentinel.defense_authority import DefenseMode
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
                product="multi-assurance",
                adapter_version="1",
                supported_source_types=[source_type],
            ),
            source=EntityMappingSpec(
                id_field=source_field,
                entity_type=source_entity_type,
            ),
            target=EntityMappingSpec(
                id_field=target_field,
                entity_type=target_entity_type,
            ),
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
    adapter_id = f"test.{suffix}"
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


def _component_results(prefix: str, chars: tuple[str, str, str]):
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
            fields={
                "credential_id": f"credential:{prefix}",
                "device_id": f"device:{prefix}",
            },
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
            fields={
                "device_id": f"device:{prefix}",
                "pipeline_id": f"pipeline:{prefix}",
            },
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
            fields={
                "pipeline_id": f"pipeline:{prefix}",
                "wallet_id": f"wallet:{prefix}",
            },
            payload_char=chars[2],
        ),
    ]


def test_assurance_domains_do_not_transfer_between_parallel_attack_components() -> None:
    component_a = _component_results("a", ("a", "b", "c"))
    component_b = _component_results("b", ("d", "e", "f"))
    rows = component_a + component_b

    enrollments = []
    for prefix, component_rows in (("a", component_a), ("b", component_b)):
        source_types = [
            TelemetrySourceType.ENDPOINT,
            TelemetrySourceType.CICD,
            TelemetrySourceType.SIGNER_WALLET,
        ]
        source_instances = [
            f"sensor:{prefix}:endpoint",
            f"sensor:{prefix}:cicd",
            f"sensor:{prefix}:signer",
        ]
        for index, ((adapter_id, _), source_type, source_instance) in enumerate(
            zip(component_rows, source_types, source_instances, strict=True)
        ):
            domain = f"a-domain-{index}" if prefix == "a" else "b-single-domain"
            enrollments.append(
                PerceptionSourceEnrollment(
                    source_type=source_type,
                    source_instance=source_instance,
                    allowed_adapter_ids=[adapter_id],
                    independence_domain=domain,
                )
            )

    registry = PerceptionSourceRegistry(
        registry_id="registry:parallel",
        enrollments=enrollments,
    )
    admitted = [
        admit_perception_adapter_result(result, registry)
        for _, result in rows
    ]
    fused = fuse_admitted_perception_batches(
        admitted,
        registry=registry,
        fused_batch_id="fusion:parallel",
    )
    bound = compile_bound_perception_graph(
        fused.perception_batch,
        graph_id="incident:parallel",
    )

    plan = build_assured_multi_incident_defense_plan(
        bound.graph,
        graph_receipt=bound.receipt,
        perception_assurance=fused.assurance,
        registry=registry,
        critical_entity_ids=["wallet:a", "wallet:b"],
    )

    assert plan.active_component_count == 2
    by_wallet = {}
    for component in plan.component_plans:
        for wallet in ("wallet:a", "wallet:b"):
            if wallet in component.entity_ids:
                by_wallet[wallet] = component

    assert by_wallet["wallet:a"].assured_plan.assurance.effective_mode is DefenseMode.SIEGE
    assert by_wallet["wallet:a"].assured_plan.assurance.active_independent_domain_count == 3
    assert by_wallet["wallet:b"].assured_plan.assurance.effective_mode is DefenseMode.GUARD
    assert by_wallet["wallet:b"].assured_plan.assurance.active_independent_domain_count == 1
