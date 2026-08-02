from __future__ import annotations

from pathlib import Path

import pytest

from koschei_sentinel.adapters import CandidateRegistry, load_candidate_registry
from koschei_sentinel.benchmark import load_benchmark_suite
from koschei_sentinel.matrix import run_comparison, write_comparison

FIXTURES = Path(__file__).parents[1] / "fixtures"


def safe_run():
    suite = load_benchmark_suite(FIXTURES / "evals/suite.safe.jsonl")
    registry_path = FIXTURES / "models/candidates.safe.json"
    registry = load_candidate_registry(registry_path)
    return run_comparison(
        suite,
        registry,
        registry_base_dir=registry_path.parent,
    )


def test_safe_comparison_passes_and_ranks_deterministically() -> None:
    first = safe_run()
    second = safe_run()
    assert first.matrix == second.matrix
    assert first.matrix.all_candidates_passed
    assert first.matrix.passed_candidates == 2
    assert first.matrix.ranking == [
        "sentinel-baseline-v0.4",
        "sentinel-replay-safe",
    ]


def test_network_error_is_recorded_without_aborting_matrix() -> None:
    suite = load_benchmark_suite(FIXTURES / "evals/suite.safe.jsonl")
    registry = CandidateRegistry.model_validate(
        {
            "candidates": [
                {
                    "candidate_id": "remote-model",
                    "adapter": "openai-compatible",
                    "model": "remote/model",
                    "base_url": "https://models.example/v1",
                    "api_key_env": "MODEL_API_KEY",
                }
            ]
        }
    )
    run = run_comparison(
        suite,
        registry,
        registry_base_dir=Path("."),
    )
    outcome = run.matrix.candidates[0]
    assert outcome.status == "error"
    assert outcome.error_code == "network_disabled"
    assert not run.matrix.any_candidate_passed


def test_comparison_artifacts_are_deterministic(tmp_path: Path) -> None:
    first = safe_run()
    second = safe_run()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    write_comparison(first, first_dir)
    write_comparison(second, second_dir)
    first_files = sorted(
        path.relative_to(first_dir) for path in first_dir.rglob("*") if path.is_file()
    )
    second_files = sorted(
        path.relative_to(second_dir) for path in second_dir.rglob("*") if path.is_file()
    )
    assert first_files == second_files
    for relative in first_files:
        assert (first_dir / relative).read_bytes() == (second_dir / relative).read_bytes()


def test_existing_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "matrix"
    output.mkdir()
    marker = output / "do-not-touch"
    marker.write_text("safe")
    with pytest.raises(FileExistsError):
        write_comparison(safe_run(), output)
    assert marker.read_text() == "safe"
