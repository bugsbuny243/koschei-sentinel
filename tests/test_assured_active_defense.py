from koschei_sentinel.assured_active_defense import build_assured_active_defense_plan
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
    *,
    adapter_id: str,
    source_type: TelemetrySourceType,
    relation_type: str,
    source_id_field: str,
    source_entity_type: CyberEntityType,
    target_id_field: str,
    target_entity_type: CyberEntityType,
) -> DeclarativePerceptionAdapter:
    return DeclarativePerceptionAdapter(
        DeclarativeAdapterSpec(
            descriptor=PerceptionAdapterDescriptor(
                adapter_id=adapter_id,
                vendor="test",
                product="assurance-test",
                adapter_version="1",
                supported_source_types=[source_type],
            ),
            source=EntityMappingSpec(
                id_field=source_id_field,
                entity_type=source_entity_type,
            ),
            target=EntityMappingSpec(
                id_field=target_id_field,
                entity_type=target_entity_type,
            ),
            relation_type=relation_type,
        )
    )


def _result(
    *,
    adapter: DeclarativePerceptionAdapter,
    event_id: str,
    source_type: TelemetrySourceType,
    source_instance: str,
    payload_char: str,
    fields: dict[str, str],
):
    event = SanitizedTelemetryEvent(
        event_id=event_id,
        source_type=source_type,
        source_instance=source_instance,
        payload_sha256=payload_char * 64,
        fields=fields,
    )
    return run_perception_adapter_batch(
        adapter,
        [event],
        batch_id=f"batch:{event_id}",
    )


def _attack_results(include_unrelated: bool = False):
    credential = _result(
        adapter=_adapter(
            adapter_id="test.credential-path",
            source_type=TelemetrySourceType.ENDPOINT,
            relation_type="uses_credential",
            source_id_field="credential_id",
            source_entity_type=CyberEntityType.CREDENTIAL,
            target_id_field="device_id",
            target_entity_type=CyberEntityType.DEVICE,
        ),
        event_id="credential",
        source_type=TelemetrySourceType.ENDPOINT,
        source_instance="sensor:credential",
        payload_char="a",
        fields={"credential_id": "credential:prod", "device_id": "device:runner"},
    )
    pipeline = _result(
        adapter=_adapter(
            adapter_id="test.pipeline-path",
            source_type=TelemetrySourceType.CICD,
            relation_type="modifies_pipeline",
            source_id_field="device_id",
            source_entity_type=CyberEntityType.DEVICE,
            target_id_field="pipeline_id",
            target_entity_type=CyberEntityType.PIPELINE,
        ),
        event_id="pipeline",
        source_type=TelemetrySourceType.CICD,
        source_instance="sensor:pipeline",
        payload_char="b",
        fields={"device_id": "device:runner", "pipeline_id": "pipeline:prod"},
    )
    signer = _result(
        adapter=_adapter(
            adapter_id="test.signer-path",
            source_type=TelemetrySourceType.SIGNER_WALLET,
            relation_type="reaches_signer",
            source_id_field="pipeline_id",
            source_entity_type=CyberEntityType.PIPELINE,
            target_id_field="wallet_id",
            target_entity_type=CyberEntityType.WALLET,
        ),
        event_id="signer",
        source_type=TelemetrySourceType.SIGNER_WALLET,
        source_instance="sensor:signer",
        payload_char="c",
        fields={"pipeline_id": "pipeline:prod", "wallet_id": "wallet:treasury"},
    )
    rows = [credential, pipeline, signer]
    if include_unrelated:
        rows.append(
            _result(
                adapter=_adapter(
                    adapter_id="test.unrelated-cloud",
                    source_type=TelemetrySourceType.CLOUD_IAM,
                    relation_type="authenticates_to",
                    source_id_field="identity_id",
                    source_entity_type=CyberEntityType.IDENTITY,
                    target_id_field="resource_id",
                    target_entity_type=CyberEntityType.CLOUD_RESOURCE,
                ),
                event_id="unrelated",
                source_type=TelemetrySourceType.CLOUD_IAM,
                source_instance="sensor:unrelated",
                payload_char="d",
                fields={"identity_id": "identity:other", "resource_id": "cloud:other"},
            )
        )
    return rows


def _registry(domains: dict[str, str], *, include_unrelated: bool = False):
    enrollments = [
        PerceptionSourceEnrollment(
            source_type=TelemetrySourceType.ENDPOINT,
            source_instance="sensor:credential",
            allowed_adapter_ids=["test.credential-path"],
            independence_domain=domains["credential"],
        ),
        PerceptionSourceEnrollment(
            source_type=TelemetrySourceType.CICD,
            source_instance="sensor:pipeline",
            allowed_adapter_ids=["test.pipeline-path"],
            independence_domain=domains["pipeline"],
        ),
        PerceptionSourceEnrollment(
            source_type=TelemetrySourceType.SIGNER_WALLET,
            source_instance="sensor:signer",
            allowed_adapter_ids=["test.signer-path"],
            independence_domain=domains["signer"],
        ),
    ]
    if include_unrelated:
        enrollments.append(
            PerceptionSourceEnrollment(
                source_type=TelemetrySourceType.CLOUD_IAM,
                source_instance="sensor:unrelated",
                allowed_adapter_ids=["test.unrelated-cloud"],
                independence_domain=domains["unrelated"],
            )
        )
    return PerceptionSourceRegistry(registry_id="registry:assurance", enrollments=enrollments)


def _plan(domains: dict[str, str], *, include_unrelated: bool = False):
    registry = _registry(domains, include_unrelated=include_unrelated)
    admitted = [
        admit_perception_adapter_result(row, registry)
        for row in _attack_results(include_unrelated=include_unrelated)
    ]
    fused = fuse_admitted_perception_batches(
        admitted,
        registry=registry,
        fused_batch_id="fusion:attack",
    )
    bound = compile_bound_perception_graph(
        fused.perception_batch,
        graph_id="incident:attack",
    )
    return build_assured_active_defense_plan(
        bound.graph,
        graph_receipt=bound.receipt,
        perception_assurance=fused.assurance,
        registry=registry,
        critical_entity_ids=["wallet:treasury"],
    )


def test_three_independent_domains_can_preserve_siege() -> None:
    result = _plan(
        {
            "credential": "endpoint-domain",
            "pipeline": "cicd-domain",
            "signer": "signer-domain",
        }
    )

    assert result.assurance.base_mode is DefenseMode.SIEGE
    assert result.assurance.effective_mode is DefenseMode.SIEGE
    assert result.assurance.active_independent_domain_count == 3
    assert result.assurance.downgraded is False


def test_two_independent_domains_downgrade_siege_to_combat() -> None:
    result = _plan(
        {
            "credential": "core-domain",
            "pipeline": "core-domain",
            "signer": "signer-domain",
        }
    )

    assert result.assurance.base_mode is DefenseMode.SIEGE
    assert result.assurance.effective_mode is DefenseMode.COMBAT
    assert result.assurance.active_independent_domain_count == 2
    assert result.assurance.downgraded is True


def test_unrelated_second_domain_cannot_unlock_combat_for_single_domain_attack_path() -> None:
    result = _plan(
        {
            "credential": "attack-domain",
            "pipeline": "attack-domain",
            "signer": "attack-domain",
            "unrelated": "unrelated-domain",
        },
        include_unrelated=True,
    )

    assert result.assurance.base_mode is DefenseMode.SIEGE
    assert result.assurance.effective_mode is DefenseMode.GUARD
    assert result.assurance.active_independence_domains == ["attack-domain"]
    assert result.assurance.active_independent_domain_count == 1
    assert result.assurance.downgraded is True
