from pathlib import Path

from koschei_sentinel.benchmark import load_benchmark_suite
from koschei_sentinel.matrix import _digest_cases
from koschei_sentinel.promotion import load_promotion_policy

ROOT = Path(__file__).parents[1]


def test_shadow_promotion_policy_pins_exact_canonical_suite() -> None:
    suite = load_benchmark_suite(ROOT / "fixtures" / "evals" / "suite.safe.jsonl")
    policy = load_promotion_policy(
        ROOT / "configs" / "promotion" / "shadow-research.v1.json"
    )
    assert policy.required_benchmark_suite_digest == _digest_cases(suite)
