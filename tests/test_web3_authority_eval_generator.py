from __future__ import annotations

from collections import Counter
from pathlib import Path

from koschei_sentinel.web3_authority_eval_contract import load_and_validate_eval_bundle
from koschei_sentinel.web3_authority_eval_generator import generate_eval_bundle

CONFIG = Path("configs/training/web3-authority-evals.v1.json")
SEEDS = Path("configs/training/web3-authority-eval-seeds.v1.json")


def _bundle():
    config, seeds = load_and_validate_eval_bundle(CONFIG, SEEDS)
    return config, seeds, generate_eval_bundle(config, seeds)


def test_generator_creates_required_variants_for_every_seed() -> None:
    _config, seeds, generated = _bundle()
    assert len(generated.cases) == len(seeds.cases) * 4
    by_parent: dict[str, set[str]] = {}
    for case in generated.cases:
        by_parent.setdefault(case.parent_seed_id, set()).add(case.variant_kind)
    assert set(by_parent) == {seed.id for seed in seeds.cases}
    assert all(
        variants == {"seed", "counterfactual", "evidence_removed", "remediation_regression"}
        for variants in by_parent.values()
    )


def test_evidence_removed_variants_fail_closed() -> None:
    _config, _seeds, generated = _bundle()
    removed = [case for case in generated.cases if case.variant_kind == "evidence_removed"]
    assert removed
    assert all(case.expected_confidence == "low" for case in removed)
    assert all(case.must_abstain_from for case in removed)


def test_lineage_is_deterministic_and_unique_per_variant() -> None:
    config, seeds, first = _bundle()
    second = generate_eval_bundle(config, seeds)
    assert [case.lineage_sha256 for case in first.cases] == [
        case.lineage_sha256 for case in second.cases
    ]
    assert len({case.lineage_sha256 for case in first.cases}) == len(first.cases)


def test_generated_bundle_does_not_fake_minimum_reviewed_case_gate() -> None:
    config, seeds, generated = _bundle()
    generated_counts = Counter(case.family for case in generated.cases)
    reviewed_seed_counts = Counter(seed.family for seed in seeds.cases)

    # Generated variants improve adversarial coverage, but they are not substitutes
    # for the curriculum's minimum number of independently reviewed cases.
    assert all(generated_counts[family.id] >= 4 for family in config.families)
    assert any(
        reviewed_seed_counts[family.id] < config.eval_generation.minimum_cases_per_family
        for family in config.families
    )
