from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from koschei_sentinel.adapters import (
    AdapterError,
    CandidateRegistry,
    CandidateSpec,
    build_adapter,
    load_candidate_registry,
    parse_chat_completion,
)
from koschei_sentinel.benchmark import load_benchmark_suite


FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_candidate_registry_rejects_duplicate_ids() -> None:
    candidate = {
        "candidate_id": "same",
        "adapter": "baseline",
    }
    with pytest.raises(ValidationError, match="candidate_id values must be unique"):
        CandidateRegistry.model_validate({"candidates": [candidate, candidate]})


def test_candidate_config_cannot_contain_literal_secret() -> None:
    with pytest.raises(ValidationError):
        CandidateSpec.model_validate(
            {
                "candidate_id": "unsafe",
                "adapter": "openai-compatible",
                "model": "model",
                "base_url": "https://models.example/v1",
                "api_key": "sk-this-must-never-be-accepted",
            }
        )


def test_network_adapter_is_disabled_by_default() -> None:
    spec = CandidateSpec(
        candidate_id="remote-model",
        adapter="openai-compatible",
        model="remote/model",
        base_url="https://models.example/v1",
        api_key_env="MODEL_API_KEY",
    )
    adapter = build_adapter(spec, base_dir=Path("."))
    suite = load_benchmark_suite(FIXTURES / "evals/suite.safe.jsonl")
    with pytest.raises(AdapterError) as error:
        adapter.predict(suite)
    assert error.value.code == "network_disabled"


def test_private_endpoint_is_rejected_before_request() -> None:
    spec = CandidateSpec(
        candidate_id="private-model",
        adapter="openai-compatible",
        model="local/model",
        base_url="https://127.0.0.1:8000/v1",
    )
    adapter = build_adapter(
        spec,
        base_dir=Path("."),
        allow_network=True,
    )
    suite = load_benchmark_suite(FIXTURES / "evals/suite.safe.jsonl")
    with pytest.raises(AdapterError) as error:
        adapter.predict(suite)
    assert error.value.code == "unsafe_endpoint"


def test_replay_candidate_mismatch_fails_closed(tmp_path: Path) -> None:
    source = FIXTURES / "models/replay.safe.jsonl"
    rows = []
    for line in source.read_text().splitlines():
        item = json.loads(line)
        item["candidate"] = "different-candidate"
        rows.append(json.dumps(item))
    replay = tmp_path / "replay.jsonl"
    replay.write_text("\n".join(rows) + "\n")
    spec = CandidateSpec(
        candidate_id="expected-candidate",
        adapter="replay",
        predictions_path="replay.jsonl",
    )
    adapter = build_adapter(spec, base_dir=tmp_path)
    with pytest.raises(AdapterError) as error:
        adapter.predict([])
    assert error.value.code == "candidate_mismatch"


def test_replay_path_cannot_escape_registry_directory(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.jsonl"
    outside.write_text("")
    spec = CandidateSpec(
        candidate_id="escape",
        adapter="replay",
        predictions_path="../outside.jsonl",
    )
    adapter = build_adapter(spec, base_dir=tmp_path)
    with pytest.raises(AdapterError) as error:
        adapter.predict([])
    assert error.value.code == "unsafe_replay_path"


def test_malformed_provider_response_is_rejected() -> None:
    with pytest.raises(AdapterError) as error:
        parse_chat_completion('{"choices":[{"message":{"content":"```json\\n{}\\n```"}}]}')
    assert error.value.code == "malformed_provider_response"


def test_safe_registry_loads() -> None:
    registry = load_candidate_registry(FIXTURES / "models/candidates.safe.json")
    assert [item.candidate_id for item in registry.candidates] == [
        "sentinel-baseline-v0.4",
        "sentinel-replay-safe",
    ]
