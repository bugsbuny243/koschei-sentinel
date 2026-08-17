from koschei_sentinel.active_defense_planner import ActiveDefensePlan
from koschei_sentinel.attack_progression import (
    AttackProgressionReport,
    AttackStage,
    DefensiveCutPoint,
    PredictedTransition,
)
from koschei_sentinel.cyber_state_graph import CyberEntityType
from koschei_sentinel.defense_authority import (
    AttackAssessment,
    DefenseActionType,
    DefenseDecision,
    DefenseMode,
)
from koschei_sentinel.interception_planner import (
    InterceptionUrgency,
    build_interception_plan,
)


def test_predicted_impact_prioritizes_transaction_and_signer_controls() -> None:
    progression = AttackProgressionReport(
        graph_id="intercept-001",
        active_stages=[],
        current_stage=AttackStage.SIGNER_OR_WALLET_ACCESS,
        progression_confidence=0.99,
        predicted_transitions=[
            PredictedTransition(
                from_stage=AttackStage.SIGNER_OR_WALLET_ACCESS,
                to_stage=AttackStage.IMPACT,
                probability=0.72,
                rationale="test",
            )
        ],
        defensive_cut_points=[],
        active_relation_ids=["r1"],
        disproved_relation_ids=[],
    )
    cuts = [
        DefensiveCutPoint(
            entity_id="credential:root",
            entity_type=CyberEntityType.CREDENTIAL,
            action=DefenseActionType.REVOKE_CREDENTIAL,
            effect_score=0.99,
            downstream_entities=8,
            supporting_relation_ids=["r1"],
            rationale="test",
        ),
        DefensiveCutPoint(
            entity_id="transaction:pending",
            entity_type=CyberEntityType.TRANSACTION,
            action=DefenseActionType.HOLD_TRANSACTION,
            effect_score=0.45,
            downstream_entities=0,
            supporting_relation_ids=["r2"],
            rationale="test",
        ),
        DefensiveCutPoint(
            entity_id="wallet:signer",
            entity_type=CyberEntityType.WALLET,
            action=DefenseActionType.FREEZE_SIGNER,
            effect_score=0.50,
            downstream_entities=1,
            supporting_relation_ids=["r3"],
            rationale="test",
        ),
    ]
    plan = ActiveDefensePlan(
        graph_id="intercept-001",
        assessment=AttackAssessment(
            assessment_id="assessment:intercept-001",
            attack_confidence=0.99,
            corroborating_evidence_count=3,
            critical_asset_at_risk=True,
            active_progression=True,
            blast_radius_score=0.99,
            evidence_gap=False,
        ),
        decision=DefenseDecision(
            mode=DefenseMode.SIEGE,
            permitted_actions=list(DefenseActionType),
            rationale=["test"],
        ),
        progression=progression.model_copy(update={"defensive_cut_points": cuts}),
        authorized_cut_points=cuts,
        withheld_cut_points=[],
        critical_entity_ids=["wallet:signer"],
        rationale=["test"],
    )

    interception = build_interception_plan(plan)

    assert interception.predicted_impact_present is True
    assert interception.steps[0].action is DefenseActionType.FREEZE_SIGNER
    assert interception.steps[1].action is DefenseActionType.HOLD_TRANSACTION
    assert interception.steps[0].urgency is InterceptionUrgency.IMMEDIATE
    assert interception.steps[1].urgency is InterceptionUrgency.IMMEDIATE
    assert interception.steps[2].action is DefenseActionType.REVOKE_CREDENTIAL


def test_no_authorized_cut_points_produces_no_interception_steps() -> None:
    progression = AttackProgressionReport(
        graph_id="guard-empty",
        active_stages=[],
        current_stage=None,
        progression_confidence=0.2,
        predicted_transitions=[],
        defensive_cut_points=[],
        active_relation_ids=[],
        disproved_relation_ids=[],
    )
    plan = ActiveDefensePlan(
        graph_id="guard-empty",
        assessment=AttackAssessment(
            assessment_id="assessment:guard-empty",
            attack_confidence=0.2,
            corroborating_evidence_count=0,
            critical_asset_at_risk=False,
            active_progression=False,
            blast_radius_score=0.0,
            evidence_gap=True,
        ),
        decision=DefenseDecision(
            mode=DefenseMode.GUARD,
            permitted_actions=[DefenseActionType.OBSERVE, DefenseActionType.COLLECT_EVIDENCE],
            rationale=["test"],
        ),
        progression=progression,
        authorized_cut_points=[],
        withheld_cut_points=[],
        critical_entity_ids=[],
        rationale=["test"],
    )

    interception = build_interception_plan(plan)

    assert interception.steps == []
    assert interception.predicted_impact_present is False
