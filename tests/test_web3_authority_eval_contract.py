from __future__ import annotations

import json
from pathlib import Path

import pytest

from koschei_sentinel.web3_authority_eval_contract import (
    Web3AuthorityEvalConfig,
    Web3AuthorityEvalSeeds,
    load_and_validate_eval_bundle,
    validate_eval_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/training/web3-authority-evals.v1.json"
SEEDS = ROOT / "configs/training/web3-authority-eval-seeds.v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_repository_bundle_is_valid() -> None:
    config, seeds = load_and_validate_eval_bundle(CONFIG, SEEDS)
    assert config.target_architecture.total_parameters == "397B"
    assert config.target_architecture.active_parameters == "35B"
    assert len(config.families) == 8
    assert len(seeds.cases) == 8


def test_unknown_family_fails_closed() -> None:
    config = Web3AuthorityEvalConfig.model_validate(_load(CONFIG))
    raw_seeds = _load(SEEDS)
    raw_seeds["cases"][0]["family"] = "unknown_family"
    seeds = Web3AuthorityEvalSeeds.model_validate(raw_seeds)
    with pytest.raises(ValueError, match="unknown families"):
        validate_eval_bundle(config, seeds)


def test_missing_family_seed_fails_closed() -> None:
    config = Web3AuthorityEvalConfig.model_validate(_load(CONFIG))
    raw_seeds = _load(SEEDS)
    raw_seeds["cases"] = [
        case for case in raw_seeds["cases"] if case["family"] != "authority_graph"
    ]
    seeds = Web3AuthorityEvalSeeds.model_validate(raw_seeds)
    with pytest.raises(ValueError, match="requires at least one reviewed seed"):
        validate_eval_bundle(config, seeds)


def test_abstention_without_boundary_is_rejected() -> None:
    raw_seeds = _load(SEEDS)
    abstention = next(case for case in raw_seeds["cases"] if case["mode"] == "abstention")
    abstention["must_abstain_from"] = []
    with pytest.raises(ValueError, match="must define must_abstain_from"):
        Web3AuthorityEvalSeeds.model_validate(raw_seeds)


def test_architecture_drift_is_rejected() -> None:
    raw_config = _load(CONFIG)
    raw_config["target_architecture"]["active_parameters"] = "17B"
    with pytest.raises(ValueError):
        Web3AuthorityEvalConfig.model_validate(raw_config)


def test_scoring_weight_drift_is_rejected() -> None:
    raw_config = _load(CONFIG)
    raw_config["scoring"]["dimensions"]["authority_reconstruction"] = 0.25
    with pytest.raises(ValueError, match="must sum to 1.0"):
        Web3AuthorityEvalConfig.model_validate(raw_config)


def test_hard_fail_authority_guard_cannot_be_removed() -> None:
    raw_config = _load(CONFIG)
    raw_config["scoring"]["hard_fail_conditions"].remove("invented_authority")
    config = Web3AuthorityEvalConfig.model_validate(raw_config)
    seeds = Web3AuthorityEvalSeeds.model_validate(_load(SEEDS))
    with pytest.raises(ValueError, match="invented_authority"):
        validate_eval_bundle(config, seeds)
