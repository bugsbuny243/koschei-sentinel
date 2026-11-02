"""Regression coverage for the five-layer defense extensions and read-only API."""
from __future__ import annotations

from fastapi.testclient import TestClient

from koschei_sentinel.assured_active_defense import AssuredActiveDefensePlan
from koschei_sentinel.attack_progression import analyze_attack_progression
from koschei_sentinel.cyber_perception import TelemetrySourceType
from koschei_sentinel.cyber_state_graph import (
    CyberEntity,
    CyberEntityType,
    CyberRelation,
    CyberStateGraph,
    EvidenceRef,
    EvidenceStatus,
)
from koschei_sentinel.defense_authority import (
    AttackAssessment,
    DefenseActionType,
    DefenseMode,
    decide_defense_mode,
)
from koschei_sentinel.interception_planner import build_interception_plan
from koschei_sentinel.perception_adapter_profiles import (
    generic_network_flowspec_adapter,
    generic_protocol_circuit_adapter,
)
from koschei_sentinel.perception_adapter_sdk import (
    DeclarativeAdapterSpec,
    DeclarativePerceptionAdapter,
    EntityMappingSpec,
    PerceptionAdapterDescriptor,
    SanitizedTelemetryEvent,
    run_perception_adapter,
    run_perception_adapter_batch,
)
from koschei_sentinel.perception_assurance import fuse_admitted_perception_batches
from koschei_sentinel.perception_graph_binding import compile_bound_perception_graph
from koschei_sentinel.perception_source_registry import (
    PerceptionSourceEnrollment,
    PerceptionSourceRegistry,
    admit_perception_adapter_result,
)
from koschei_sentinel.service import app

client = TestClient(app)


def _graph() -> CyberStateGraph:
    return CyberStateGraph(
        graph_id="graph:v5",
        entities=[
            CyberEntity(entity_id="identity:one", entity_type=CyberEntityType.IDENTITY),
            CyberEntity(entity_id="credential:one", entity_type=CyberEntityType.CREDENTIAL),
            CyberEntity(entity_id="protocol:one", entity_type=CyberEntityType.PROTOCOL),
            CyberEntity(entity_id="transaction:one", entity_type=CyberEntityType.TRANSACTION),
        ],
        relations=[
            CyberRelation(
                relation_id="relation:identity",
                source_entity_id="identity:one",
                target_entity_id="credential:one",
                relation_type="uses_credential",
                status=EvidenceStatus.OBSERVED,
                confidence=0.99,
                evidence=[
                    EvidenceRef(
                        evidence_id="evidence:identity",
                        source="ENDPOINT:sensor-1",
                        content_sha256="a" * 64,
                    )
                ],
            ),
            CyberRelation(
                relation_id="relation:protocol",
                source_entity_id="credential:one",
                target_entity_id="protocol:one",
                relation_type="reaches_signer",
                status=EvidenceStatus.OBSERVED,
                confidence=0.99,
                evidence=[
                    EvidenceRef(
                        evidence_id="evidence:protocol",
                        source="BLOCKCHAIN:sensor-2",
                        content_sha256="b" * 64,
                    )
                ],
            ),
            CyberRelation(
                relation_id="relation:transaction",
                source_entity_id="protocol:one",
                target_entity_id="transaction:one",
                relation_type="prepares_transaction",
                status=EvidenceStatus.OBSERVED,
                confidence=0.99,
                evidence=[
                    EvidenceRef(
                        evidence_id="evidence:transaction",
                        source="BLOCKCHAIN:sensor-3",
                        content_sha256="c" * 64,
                    )
                ],
            ),
        ],
    )


def _assured_api_payload() -> tuple[dict, CyberStateGraph]:
    definitions = [
        (
            TelemetrySourceType.ENDPOINT,
            "credential:prod",
            CyberEntityType.CREDENTIAL,
            "device:runner",
            CyberEntityType.DEVICE,
            "uses_credential",
        ),
        (
            TelemetrySourceType.CICD,
            "device:runner",
            CyberEntityType.DEVICE,
            "pipeline:prod",
            CyberEntityType.PIPELINE,
            "modifies_pipeline",
        ),
        (
            TelemetrySourceType.SIGNER_WALLET,
            "pipeline:prod",
            CyberEntityType.PIPELINE,
            "wallet:treasury",
            CyberEntityType.WALLET,
            "reaches_signer",
        ),
    ]
    enrollments = []
    adapter_results = []
    for index, (source_type, source_id, source_entity_type, target_id, target_entity_type, relation_type) in enumerate(definitions):
        adapter_id = f"test.v5-{index}"
        source_instance = f"sensor:v5-{index}"
        adapter = DeclarativePerceptionAdapter(
            DeclarativeAdapterSpec(
                descriptor=PerceptionAdapterDescriptor(
                    adapter_id=adapter_id,
                    vendor="test",
                    product="v5-test",
                    adapter_version="1",
                    supported_source_types=[source_type],
                ),
                source=EntityMappingSpec(id_field="source_id", entity_type=source_entity_type),
                target=EntityMappingSpec(id_field="target_id", entity_type=target_entity_type),
                relation_type=relation_type,
            )
        )
        adapter_results.append(
            run_perception_adapter_batch(
                adapter,
                [
                    SanitizedTelemetryEvent(
                        event_id=f"event:v5-{index}",
                        source_type=source_type,
                        source_instance=source_instance,
                        payload_sha256=("abc"[index]) * 64,
                        fields={"source_id": source_id, "target_id": target_id},
                    )
                ],
                batch_id=f"batch:v5-{index}",
            )
        )
        enrollments.append(
            PerceptionSourceEnrollment(
                source_type=source_type,
                source_instance=source_instance,
                allowed_adapter_ids=[adapter_id],
                independence_domain=f"domain:v5-{index}",
            )
        )
    registry = PerceptionSourceRegistry(registry_id="registry:v5", enrollments=enrollments)
    admitted = [admit_perception_adapter_result(row, registry) for row in adapter_results]
    fused = fuse_admitted_perception_batches(
        admitted, registry=registry, fused_batch_id="fusion:v5"
    )
    bound = compile_bound_perception_graph(fused.perception_batch, graph_id="graph:v5-assured")
    request = {
        "graph": bound.graph.model_dump(mode="json"),
        "graph_receipt": bound.receipt.model_dump(mode="json"),
        "perception_assurance": fused.assurance.model_dump(mode="json"),
        "registry": registry.model_dump(mode="json"),
        "critical_entity_ids": ["wallet:treasury"],
    }
    return request, bound.graph


def test_identity_and_protocol_generate_cut_points() -> None:
    progression = analyze_attack_progression(_graph())
    actions_by_entity = {
        row.entity_id: row.action for row in progression.defensive_cut_points
    }
    assert actions_by_entity["identity:one"] is DefenseActionType.TERMINATE_SESSION
    assert actions_by_entity["protocol:one"] is DefenseActionType.PAUSE_PROTOCOL_CONTRACT


def test_combat_and_siege_layer_permissions() -> None:
    combat = decide_defense_mode(
        AttackAssessment(
            assessment_id="assessment:combat",
            attack_confidence=0.90,
            corroborating_evidence_count=2,
            active_progression=True,
            critical_asset_at_risk=True,
            blast_radius_score=0.5,
        )
    )
    siege = decide_defense_mode(
        AttackAssessment(
            assessment_id="assessment:siege",
            attack_confidence=0.97,
            corroborating_evidence_count=3,
            active_progression=True,
            critical_asset_at_risk=True,
            blast_radius_score=0.8,
        )
    )
    assert combat.mode is DefenseMode.COMBAT
    assert siege.mode is DefenseMode.SIEGE
    combat_actions = set(combat.permitted_actions)
    siege_actions = set(siege.permitted_actions)
    assert {
        DefenseActionType.ISOLATE_MICROSEGMENT,
        DefenseActionType.BGP_BLACKHOLE_ROUTE,
        DefenseActionType.DIVERT_FLOWSPEC_SCRUBBING,
        DefenseActionType.PAUSE_PROTOCOL_CONTRACT,
    } <= combat_actions
    hardware_actions = {
        DefenseActionType.LOCK_HSM_PARTITION,
        DefenseActionType.TRIGGER_AIRGAP_ISOLATION,
    }
    assert not combat_actions.intersection(hardware_actions)
    assert hardware_actions <= siege_actions


def test_new_telemetry_adapters_only_normalize_evidence() -> None:
    network = SanitizedTelemetryEvent(
        event_id="event:network",
        source_type=TelemetrySourceType.NETWORK,
        source_instance="network:monitor",
        payload_sha256="d" * 64,
        fields={
            "network_endpoint_id": "network:upstream",
            "destination_endpoint_id": "network:destination",
            "source_prefix": "192.0.2.0/24",
            "destination_prefix": "198.51.100.0/24",
        },
    )
    rows, receipt = run_perception_adapter(generic_network_flowspec_adapter(), network)
    assert receipt.observation_count == 1
    assert rows[0].source_entity.entity_type is CyberEntityType.NETWORK_ENDPOINT
    assert rows[0].target_entity.entity_type is CyberEntityType.NETWORK_ENDPOINT

    protocol = SanitizedTelemetryEvent(
        event_id="event:protocol",
        source_type=TelemetrySourceType.BLOCKCHAIN,
        source_instance="chain:monitor",
        payload_sha256="e" * 64,
        fields={
            "protocol_id": "protocol:bridge",
            "transaction_id": "transaction:sample",
            "chain_id": "1",
        },
    )
    observations, _ = run_perception_adapter(generic_protocol_circuit_adapter(), protocol)
    assert observations[0].source_entity.entity_type is CyberEntityType.PROTOCOL
    assert observations[0].target_entity.entity_type is CyberEntityType.TRANSACTION


def test_plan_endpoint_preserves_existing_routes_and_initializes_execution() -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200
    response = client.post(
        "/v1/active-defense/plan",
        json={
            "graph": _graph().model_dump(mode="json"),
            "critical_entity_ids": ["protocol:one"],
        },
    )
    assert response.status_code == 200, response.text
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Sentinel-Execution-Mode"] == "simulation-only"
    assert response.headers["X-Sentinel-Evidence-Trust"] == "caller-supplied"
    body = response.json()
    assert body["active_defense_plan"]["graph_id"] == "graph:v5"
    assert body["interception_plan"]["graph_id"] == "graph:v5"
    assert body["interception_execution"]["graph_id"] == "graph:v5"
    assert client.post(
        "/v1/active-defense/plan",
        json={"graph": _graph().model_dump(mode="json"), "critical_entity_ids": ["unknown"]},
    ).status_code == 422


def test_assured_plan_and_connector_api_are_fail_closed() -> None:
    payload, graph = _assured_api_payload()
    response = client.post("/v1/active-defense/assured-plan", json=payload)
    assert response.status_code == 200, response.text
    assert response.headers["X-Sentinel-Execution-Mode"] == "simulation-only"
    assert response.headers["X-Sentinel-Evidence-Trust"] == "caller-supplied"
    assured = AssuredActiveDefensePlan.model_validate(response.json())
    assert assured.assurance.active_independent_domain_count == 3
    assert assured.assurance.effective_mode is DefenseMode.SIEGE

    interception = build_interception_plan(assured.active_defense_plan)
    assert interception.steps
    step = interception.steps[0]
    evidence_ids = sorted({
        row.evidence_id for relation in graph.relations for row in relation.evidence
    })
    request = {
        "scope": {
            "scope_id": "scope:v5",
            "entity_ids": [step.target_entity_id],
            "permitted_actions": [step.action.value],
        },
        "assured_plan": assured.model_dump(mode="json"),
        "interception_plan": interception.model_dump(mode="json"),
        "step_id": step.step_id,
        "precondition_evidence_ids": evidence_ids,
    }
    dry_run = client.post("/v1/active-defense/connector-envelope", json=request)
    assert dry_run.status_code == 200, dry_run.text
    assert dry_run.headers["X-Sentinel-Production-Authorization"] == "not-granted"
    assert dry_run.json()["command"]["dry_run"] is True
    assert dry_run.json()["production_authorized"] is False
    live_command = client.post(
        "/v1/active-defense/connector-envelope",
        json={**request, "dry_run": False},
    )
    assert live_command.status_code == 403
    assert live_command.headers["X-Sentinel-Production-Authorization"] == "not-granted"
    assert live_command.headers["X-Sentinel-Evidence-Trust"] == "caller-supplied"

    tampered = dict(payload)
    tampered["graph_receipt"] = {
        **payload["graph_receipt"],
        "graph_sha256": "0" * 64,
    }
    assert client.post(
        "/v1/active-defense/assured-plan", json=tampered
    ).status_code == 422


def test_active_defense_api_limits_untrusted_graph_size(monkeypatch) -> None:
    monkeypatch.setattr("koschei_sentinel.service._MAX_API_GRAPH_ENTITIES", 3)
    response = client.post(
        "/v1/active-defense/plan",
        json={"graph": _graph().model_dump(mode="json")},
    )
    assert response.status_code == 422
    assert "graph entity limit exceeded" in response.text


def test_assured_api_limits_untrusted_registry_size(monkeypatch) -> None:
    payload, _ = _assured_api_payload()
    monkeypatch.setattr("koschei_sentinel.service._MAX_API_SOURCE_ENROLLMENTS", 2)
    response = client.post("/v1/active-defense/assured-plan", json=payload)
    assert response.status_code == 422
    assert "source enrollment limit exceeded" in response.text


def test_active_defense_ingress_rejects_declared_oversize_body(monkeypatch) -> None:
    monkeypatch.setattr(
        "koschei_sentinel.request_body_limit._MAX_ACTIVE_DEFENSE_BODY_BYTES", 128
    )
    response = client.post(
        "/v1/active-defense/plan",
        json={"graph": _graph().model_dump(mode="json")},
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "active-defense request body limit exceeded"
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Sentinel-Execution-Mode"] == "simulation-only"


def test_active_defense_ingress_rejects_undeclared_streaming_body(monkeypatch) -> None:
    monkeypatch.setattr(
        "koschei_sentinel.request_body_limit._MAX_ACTIVE_DEFENSE_BODY_BYTES", 128
    )
    response = client.post(
        "/v1/active-defense/plan",
        content=iter([b'{"graph":"', b"x" * 150, b'"}']),
        headers={"Content-Type": "application/json", "Transfer-Encoding": "chunked"},
    )
    assert response.status_code == 413
    assert "body limit exceeded" in response.text


def test_body_guard_does_not_change_legacy_opinions_route(monkeypatch) -> None:
    monkeypatch.setattr(
        "koschei_sentinel.request_body_limit._MAX_ACTIVE_DEFENSE_BODY_BYTES", 128
    )
    response = client.post("/v1/opinions", json={"extra": "a" * 512})
    assert response.status_code == 422


def test_handled_api_errors_carry_nonproduction_trust_headers(monkeypatch) -> None:
    responses = [
        client.post("/v1/active-defense/plan", json={}),
        client.post(
            "/v1/active-defense/plan",
            content=b"{",
            headers={"Content-Type": "application/json"},
        ),
        client.get("/v1/active-defense/plan"),
        client.get("/v1/active-defense/unknown"),
    ]
    monkeypatch.setattr(
        "koschei_sentinel.request_body_limit._MAX_ACTIVE_DEFENSE_BODY_BYTES", 10
    )
    responses.append(
        client.post(
            "/v1/active-defense/plan",
            json={"graph": _graph().model_dump(mode="json")},
        )
    )
    assert [response.status_code for response in responses] == [422, 422, 405, 404, 413]
    expected = {
        "cache-control": "no-store",
        "x-sentinel-execution-mode": "simulation-only",
        "x-sentinel-evidence-trust": "caller-supplied",
        "x-sentinel-production-authorization": "not-granted",
    }
    for response in responses:
        for header, value in expected.items():
            assert response.headers[header] == value
            assert sum(key.lower() == header.encode() for key, _ in response.headers.raw) == 1

    legacy = client.post("/v1/opinions", json={})
    assert legacy.status_code == 422
    assert "X-Sentinel-Execution-Mode" not in legacy.headers


def test_api_rejects_deep_json_before_framework_parsing(monkeypatch) -> None:
    monkeypatch.setattr(
        "koschei_sentinel.request_body_limit._MAX_ACTIVE_DEFENSE_JSON_DEPTH", 4
    )
    response = client.post(
        "/v1/active-defense/plan",
        content=b"[" * 5 + b"0" + b"]" * 5,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "active-defense JSON nesting limit exceeded"
    assert response.headers["X-Sentinel-Execution-Mode"] == "simulation-only"
    assert response.headers["X-Sentinel-Production-Authorization"] == "not-granted"


def test_api_rejects_overwide_json_before_framework_parsing(monkeypatch) -> None:
    monkeypatch.setattr(
        "koschei_sentinel.request_body_limit._MAX_ACTIVE_DEFENSE_JSON_STRUCTURE_TOKENS",
        3,
    )
    response = client.post(
        "/v1/active-defense/plan",
        content=b"[0,1,2,3,4]",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "active-defense JSON structure limit exceeded"
    assert response.headers["Cache-Control"] == "no-store"
