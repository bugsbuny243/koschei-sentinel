from collections import Counter

from koschei_sentinel.cyber_seed_curriculum import (
    SEED_POLICY_ID,
    build_seed_curriculum,
)
from koschei_sentinel.defense_reflex_corpus_v3 import (
    DefenseReviewMethod,
    build_defense_reflex_v3_examples,
    build_defense_reflex_v3_manifest,
)


def _graph_evidence_ids(scenario) -> set[str]:
    return {
        evidence.evidence_id
        for graph in scenario.graph_snapshots
        for relation in graph.relations
        for evidence in relation.evidence
    }


def test_seed_curriculum_has_eight_balanced_families_and_32_scenarios() -> None:
    rows = build_seed_curriculum()
    counts = Counter(family for family, _, _ in rows)

    assert len(rows) == 32
    assert len(counts) == 8
    assert set(counts.values()) == {4}
    assert len({scenario.scenario_id for _, scenario, _ in rows}) == 32


def test_seed_curriculum_is_training_authorized_but_never_promotion_eligible() -> None:
    rows = build_seed_curriculum()
    for _, _, lesson in rows:
        assert lesson.reviewer_id == SEED_POLICY_ID
        assert lesson.review_method is DefenseReviewMethod.SYNTHETIC_POLICY
        assert lesson.training_authorization is True
        assert lesson.promotion_eligible is False

    examples = build_defense_reflex_v3_examples(
        [(scenario, lesson) for _, scenario, lesson in rows]
    )
    manifest = build_defense_reflex_v3_manifest(examples)
    assert manifest.example_count == 32
    assert manifest.synthetic_policy_reviewed_examples == 32
    assert manifest.human_reviewed_examples == 0
    assert manifest.promotion_eligible is False


def test_every_seed_target_cites_only_evidence_already_present_in_graph() -> None:
    rows = build_seed_curriculum()
    examples = build_defense_reflex_v3_examples(
        [(scenario, lesson) for _, scenario, lesson in rows]
    )
    scenario_by_id = {scenario.scenario_id: scenario for _, scenario, _ in rows}

    for example in examples:
        present = _graph_evidence_ids(scenario_by_id[example.scenario_id])
        cited = {
            evidence_id
            for step in example.expected_sequence
            for evidence_id in step["supporting_evidence_ids"]
        }
        assert cited
        assert cited.issubset(present)
        assert all("outcome_verification_ids" not in step for step in example.expected_sequence)
        assert all(step["outcome_verification_required"] is True for step in example.expected_sequence)
