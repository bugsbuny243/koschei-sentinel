import pytest

from tests.test_defense_resource_scheduler import _policy, _result

from koschei_sentinel.assured_multi_incident_defense import (
    build_assured_multi_incident_defense_plan,
)
from koschei_sentinel.cyber_perception import TelemetrySourceType
from koschei_sentinel.cyber_state_graph import CyberEntityType
from koschei_sentinel.defense_resource_scheduler import build_defense_resource_schedule
from koschei_sentinel.defense_wave_execution import (
    authorize_wave_component,
    record_wave_component_execution,
    start_defense_wave_execution,
    verify_wave_component_outcome,
)
from koschei_sentinel.defense_wave_reassessment import reassess_after_defense_wave
from koschei_sentinel.perception_assurance import fuse_admitted_perception_batches
from koschei_sentinel.perception_graph_binding import compile_bound_perception_graph
from koschei_sentinel.perception_source_registry import (
    PerceptionSourceEnrollment,
    PerceptionSourceRegistry,
    admit_perception_adapter_result,
)


def _results(version: str, chars: tuple[str, str, str]):
    return [
        _result(
            suffix=f"reassess-{version}-credential",
            source_type=TelemetrySourceType.ENDPOINT,
            source_instance="sensor:reassess:endpoint",
            relation_type="uses_credential",
            source_field="credential_id",
            source_entity_type=CyberEntityType.CREDENTIAL,
            target_field="device_id",
            target_entity_type=CyberEntityType.DEVICE,
            fields={"credential_id": "credential:prod", "device_id": "device:runner"},
            payload_char=chars[0],
        ),
        _result(
            suffix=f"reassess-{version}-pipeline",
            source_type=TelemetrySourceType.CICD,
            source_instance="sensor:reassess:cicd",
            relation_type="modifies_pipeline",
            source_field="device_id",
            source_entity_type=CyberEntityType.DEVICE,
            target_field="pipeline_id",
            target_entity_type=CyberEntityType.PIPELINE,
            fields={"device_id": "device:runner", "pipeline_id": "pipeline:prod"},
            payload_char=chars[1],
        ),
        _result(
            suffix=f"reassess-{version}-signer",
            source_type=TelemetrySourceType.SIGNER_WALLET,
            source_instance="sensor:reassess:signer",
            relation_type="reaches_signer",
            source_field="pipeline_id",
            source_entity_type=CyberEntityType.PIPELINE,
            target_field="wallet_id",
            target_entity_type=CyberEntityType.WALLET,
            fields={"pipeline_id": "pipeline:prod", "wallet_id": "wallet:treasury"},
            payload_char=chars[2],
        ),
    ]


def _registry() -> PerceptionSourceRegistry:
    return PerceptionSourceRegistry(
        registry_id="registry:wave-reassessment",
        enrollments=[
            PerceptionSourceEnrollment(
                source_type=TelemetrySourceType.ENDPOINT,
                source_instance="sensor:reassess:endpoint",
                allowed_adapter_ids=[
                    "scheduler.reassess-v1-credential",
                    "scheduler.reassess-v2-credential",
                ],
                independence_domain="reassess-endpoint",
            ),
            PerceptionSourceEnrollment(
                source_type=TelemetrySourceType.CICD,
                source_instance="sensor:reassess:cicd",
                allowed_adapter_ids=[
                    "scheduler.reassess-v1-pipeline",
                    "scheduler.reassess-v2-pipeline",
                ],
                independence_domain="reassess-cicd",
            ),
            PerceptionSourceEnrollment(
                source_type=TelemetrySourceType.SIGNER_WALLET,
                source_instance="sensor:reassess:signer",
                allowed_adapter_ids=[
                    "scheduler.reassess-v1-signer",
                    "scheduler.reassess-v2-signer",
                ],
                independence_domain="reassess-signer",
            ),
        ],
    )


def _snapshot(version: str, *, fused_batch_id: str):
    registry = _registry()
    rows = _results(version, ("a", "b", "c") if version == "v1" else ("d", "e", "f"))
    admitted = [
        admit_perception_adapter_result(result, registry) for _, result in rows
    ]
    fused = fuse_admitted_perception_batches(
        admitted,
        registry=registry,
        fused_batch_id=fused_batch_id,
    )
    bound = compile_bound_perception_graph(
        fused.perception_batch,
        graph_id="incident:wave-reassessment",
    )
    plan = build_assured_multi_incident_defense_plan(
        bound.graph,
        graph_receipt=bound.receipt,
        perception_assurance=fused.assurance,
        registry=registry,
        critical_entity_ids=["wallet:treasury"],
    )
    return registry, fused, bound, plan


def _completed_wave():
    registry, fused, bound, plan = _snapshot("v1", fused_batch_id="fusion:wave:v1")
    schedule = build_defense_resource_schedule(
        plan,
        policy=_policy(total=1, signer=1, reserved=1),
    )
    wave = start_defense_wave_execution(schedule, plan)
    component_id = wave.components[0].component_id
    wave = authorize_wave_component(
        wave,
        component_id=component_id,
        precondition_evidence_ids=["evidence:wave:precondition"],
    )
    wave = record_wave_component_execution(
        wave,
        component_id=component_id,
        execution_receipt_ids=["receipt:wave:execution"],
    )
    wave = verify_wave_component_outcome(
        wave,
        component_id=component_id,
        succeeded=True,
        outcome_evidence_ids=["evidence:wave:outcome"],
    )
    assert wave.complete is True
    return registry, fused, bound, plan, schedule, wave


def test_same_perception_batch_cannot_be_reused_after_completed_wave() -> None:
    registry, fused, bound, plan, schedule, wave = _completed_wave()

    with pytest.raises(ValueError, match="fresh fused perception batch"):
        reassess_after_defense_wave(
            completed_wave=wave,
            previous_schedule=schedule,
            previous_plan=plan,
            next_bound_graph=bound,
            next_perception_assurance=fused.assurance,
            registry=registry,
            critical_entity_ids=["wallet:treasury"],
        )


def test_renaming_batch_without_new_graph_evidence_is_rejected() -> None:
    registry, _, _, plan, schedule, wave = _completed_wave()
    _, renamed_fused, renamed_bound, _ = _snapshot(
        "v1",
        fused_batch_id="fusion:wave:v1-renamed-only",
    )

    with pytest.raises(ValueError, match="materially new graph evidence"):
        reassess_after_defense_wave(
            completed_wave=wave,
            previous_schedule=schedule,
            previous_plan=plan,
            next_bound_graph=renamed_bound,
            next_perception_assurance=renamed_fused.assurance,
            registry=registry,
            critical_entity_ids=["wallet:treasury"],
        )


def test_new_evidence_produces_fresh_reassessment_and_new_plan() -> None:
    registry, _, previous_bound, plan, schedule, wave = _completed_wave()
    _, next_fused, next_bound, _ = _snapshot("v2", fused_batch_id="fusion:wave:v2")

    result = reassess_after_defense_wave(
        completed_wave=wave,
        previous_schedule=schedule,
        previous_plan=plan,
        next_bound_graph=next_bound,
        next_perception_assurance=next_fused.assurance,
        registry=registry,
        critical_entity_ids=["wallet:treasury"],
    )

    assert result.receipt.fresh_perception is True
    assert result.receipt.graph_changed is True
    assert result.receipt.plan_changed is True
    assert result.receipt.previous_graph_sha256 == previous_bound.receipt.graph_sha256
    assert result.receipt.next_graph_sha256 == next_bound.receipt.graph_sha256
    assert result.receipt.previous_plan_sha256 != result.receipt.next_plan_sha256
