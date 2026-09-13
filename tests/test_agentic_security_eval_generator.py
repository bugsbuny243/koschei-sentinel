from collections import Counter
from pathlib import Path

from pydantic import ValidationError
import pytest

from koschei_sentinel.agentic_security_eval_generator import GeneratedCase, generate_from_paths

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/training/agentic-security-protocol-state-evals.v1.json"
SEEDS = ROOT / "configs/training/agentic-security-protocol-state-seeds.v1.json"


def test_generator_builds_exactly_128_unreviewed_candidates() -> None:
    bundle = generate_from_paths(CONFIG, SEEDS)
    assert bundle.total_cases == 128
    assert len(bundle.cases) == 128
    assert bundle.review_status == "unreviewed"
    assert bundle.gold_eligible is False

    family_counts = Counter(case.family for case in bundle.cases)
    assert set(family_counts.values()) == {32}

    mode_counts = Counter((case.family, case.requested_mode) for case in bundle.cases)
    for family in family_counts:
        assert mode_counts[(family, "positive")] == 8
        assert mode_counts[(family, "hard_negative")] == 8
        assert mode_counts[(family, "ambiguous")] == 8
        assert mode_counts[(family, "abstention")] == 8


def test_generated_case_cannot_be_promoted_to_reviewed_or_gold_by_payload() -> None:
    bundle = generate_from_paths(CONFIG, SEEDS)
    payload = bundle.cases[0].model_dump(mode="json")
    payload["review_status"] = "reviewed"
    payload["gold_eligible"] = True
    with pytest.raises(ValidationError):
        GeneratedCase.model_validate(payload)


def test_generation_is_deterministic() -> None:
    first = generate_from_paths(CONFIG, SEEDS)
    second = generate_from_paths(CONFIG, SEEDS)
    assert [case.lineage_sha256 for case in first.cases] == [case.lineage_sha256 for case in second.cases]
    assert [case.case_id for case in first.cases] == [case.case_id for case in second.cases]
