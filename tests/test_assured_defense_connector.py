import pytest

from koschei_sentinel.active_defense_planner import ActiveDefensePlan
from koschei_sentinel.assured_active_defense import (
    ActiveDefenseAssurancePolicy,
    ActiveDefenseAssuranceReceipt,
    AssuredActiveDefensePlan,
)
from koschei_sentinel.assured_defense_connector import build_assured_connector_envelope
from koschei_sentinel.attack_progression import (
    AttackProgressionReport,
    DefensiveCutPoint,
)
from koschei_sentinel.cyber_state_graph import CyberEntityType
from koschei_sentinel.defense_authority import (
    AttackAssessment,
    DefenseActionType,
    DefenseDecision,
    DefenseMode,
)
from koschei_sentinel.defense_connector_contract import ProtectedScope
from koschei_sentinel.interception_planner import build_interception_plan


def _assured_plan(domain_count: int = 2) -> AssuredActiveDefensePlan:
    cut = DefensiveCutPoint(
        entity_id="device:prod",
        entity_type=CyberEntityType.DEVICE,
        action=DefenseActionType.ISOLATE_ENDPOINT,
        effect_score=0.85,
        downstream_entities=2,
        supporting_relation_ids=["r1"],
        rationale="test defensive cut point",
    )
    progression = AttackProgressionReport(
        graph_id="incident:prod",
        active_stages=[],
        current_stage=None,
        progression_confidence=0.9,
        predicted_transitions=[],
        defensive_cut_points=[cut],
        active_relation_ids=["r1"],
        disproved_relation_ids=[],
    )
    active = ActiveDefensePlan(
        graph_id="incident:prod",
        assessment=AttackAssessment(
            assessment_id="assessment:prod",
            attack_confidence=0.9,
            corroborating_evidence_count=2,
            active_progression=True,
        ),
        decision=DefenseDecision(
            mode=DefenseMode.COMBAT,
            permitted_actions=[DefenseActionType.ISOLATE_ENDPOINT],
            rationale=["test combat decision"],
        ),
        progression=progression,
        authorized_cut_points=[cut],
        withheld_cut_points=[],
        critical_entity_ids=["device:prod"],
        rationale=["test plan"],
    )
    domains = [f"domain-{index}" for index in range(domain_count)]
    assurance = ActiveDefenseAssuranceReceipt(
        graph_id="incident:prod",
        graph_sha256="a" * 64,
        source_batch_sha256="b" * 64,
        registry_sha256="c" * 64,
        base_mode=DefenseMode.COMBAT,
        effective_mode=DefenseMode.COMBAT,
        assurance_relation_ids=["r1"],
        active_source_principals=[f"ENDPOINT:sensor-{index}" for index in range(domain_count)],
        active_independence_domains=domains,
        active_independent_domain_count=domain_count,
        unknown_active_evidence_sources=[],
        downgraded=False,
        high_impact_authorized=True,
    )
    return AssuredActiveDefensePlan(
        active_defense_plan=active,
        assurance=assurance,
        policy=ActiveDefenseAssurancePolicy(),
    )


def _scope(entity_id: str = "device:prod") -> ProtectedScope:
    return ProtectedScope(
        scope_id="scope:prod",
        entity_ids=[entity_id],
        permitted_actions=[DefenseActionType.ISOLATE_ENDPOINT],
    )


def test_assured_connector_envelope_allows_bound_combat_step() -> None:
    assured = _assured_plan(2)
    interception = build_interception_plan(assured.active_defense_plan)
    envelope = build_assured_connector_envelope(
        scope=_scope(),
        assured_plan=assured,
        interception_plan=interception,
        step_id=interception.steps[0].step_id,
        precondition_evidence_ids=["evidence:one", "evidence:two"],
        dry_run=False,
    )

    assert envelope.production_authorized is True
    assert envelope.command.dry_run is False
    assert envelope.command.action is DefenseActionType.ISOLATE_ENDPOINT


def test_combat_connector_rejects_insufficient_independent_domains() -> None:
    assured = _assured_plan(1)
    interception = build_interception_plan(assured.active_defense_plan)
    with pytest.raises(ValueError, match="lacks required independent evidence domains"):
        build_assured_connector_envelope(
            scope=_scope(),
            assured_plan=assured,
            interception_plan=interception,
            step_id=interception.steps[0].step_id,
            precondition_evidence_ids=["evidence:one"],
            dry_run=False,
        )


def test_connector_rejects_target_outside_protected_scope() -> None:
    assured = _assured_plan(2)
    interception = build_interception_plan(assured.active_defense_plan)
    with pytest.raises(ValueError, match="outside the authorized protected scope"):
        build_assured_connector_envelope(
            scope=_scope("device:other"),
            assured_plan=assured,
            interception_plan=interception,
            step_id=interception.steps[0].step_id,
            precondition_evidence_ids=["evidence:one", "evidence:two"],
            dry_run=False,
        )


def test_connector_rejects_unknown_interception_step() -> None:
    assured = _assured_plan(2)
    interception = build_interception_plan(assured.active_defense_plan)
    with pytest.raises(ValueError, match="does not exist"):
        build_assured_connector_envelope(
            scope=_scope(),
            assured_plan=assured,
            interception_plan=interception,
            step_id="intercept:missing",
            precondition_evidence_ids=["evidence:one", "evidence:two"],
            dry_run=False,
        )
