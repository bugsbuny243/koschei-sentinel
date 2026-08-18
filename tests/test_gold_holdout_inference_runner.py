import hashlib
import json

import pytest

from koschei_sentinel.gold_holdout_evaluation import export_gold_holdout_inference_pack
from koschei_sentinel.gold_holdout_inference_runner import (
    GoldHoldoutGenerationPolicy,
    _load_inference_pack,
    _prediction_from_generated_text,
    _prompt_messages,
)
from koschei_sentinel.training import canonical_json
from tests.test_defense_reflex_gold_release import _release_rows
from koschei_sentinel.defense_reflex_gold_release import write_gold_defense_release


def _pack(tmp_path):
    _policy, rows = _release_rows()
    release = tmp_path / "gold-release"
    write_gold_defense_release(rows, release)
    pack = tmp_path / "holdout-pack"
    export_gold_holdout_inference_pack(release, pack)
    return pack


def test_inference_pack_contains_only_answer_key_isolated_input_contract(tmp_path) -> None:
    pack = _pack(tmp_path)

    rows, manifest, _manifest_raw = _load_inference_pack(pack)

    assert len(rows) == manifest.case_count
    assert manifest.answer_key_excluded is True
    for row in rows:
        assert set(row.input_context) == {
            "scenario_id",
            "critical_entity_ids",
            "graph_snapshots",
        }
        serialized = canonical_json(row.input_context)
        assert "expected_interpretation" not in serialized
        assert "expected_sequence" not in serialized
        assert "scenario_truth" not in serialized
        assert "range_report" not in serialized


def test_inference_pack_rejects_answer_key_field_even_if_hashes_are_recomputed(tmp_path) -> None:
    pack = _pack(tmp_path)
    inputs_path = pack / "inputs.jsonl"
    manifest_path = pack / "manifest.json"
    rows = [json.loads(line) for line in inputs_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["input_context"]["expected_sequence"] = []
    rows[0]["input_context_sha256"] = hashlib.sha256(
        canonical_json(rows[0]["input_context"]).encode("utf-8")
    ).hexdigest()
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    )
    inputs_path.write_text(payload, encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside the answer-key-isolated contract"):
        _load_inference_pack(pack)


def test_generated_prediction_is_strict_json_and_revision_is_adapter_digest(tmp_path) -> None:
    pack = _pack(tmp_path)
    cases, _manifest, _manifest_raw = _load_inference_pack(pack)
    adapter_digest = "a" * 64
    generated = json.dumps(
        {
            "interpretation": "Evidence remains limited, so collect more evidence before escalation.",
            "defense_sequence": [
                {
                    "sequence": 1,
                    "expected_mode": "GUARD",
                    "action": "COLLECT_EVIDENCE",
                    "target_entity_id": "entity:test",
                    "rationale": "Collect corroborating telemetry before active containment.",
                    "supporting_evidence_ids": ["evidence:test"],
                    "outcome_verification_required": True,
                }
            ],
        }
    )

    prediction = _prediction_from_generated_text(
        case=cases[0],
        generated_text=generated,
        model_ref="koschei-sentinel:test",
        adapter_digest=adapter_digest,
    )

    assert prediction.model_revision == adapter_digest
    assert prediction.adapter_digest == adapter_digest
    assert prediction.model_ref == "koschei-sentinel:test"
    assert prediction.prediction_sha256

    with pytest.raises(ValueError, match="not exactly one JSON object"):
        _prediction_from_generated_text(
            case=cases[0],
            generated_text=f"```json\n{generated}\n```",
            model_ref="koschei-sentinel:test",
            adapter_digest=adapter_digest,
        )


def test_inference_prompt_matches_training_input_contract_without_answer_key(tmp_path) -> None:
    pack = _pack(tmp_path)
    cases, _manifest, _manifest_raw = _load_inference_pack(pack)

    messages = _prompt_messages(cases[0])
    user_payload = json.loads(messages[1]["content"])

    assert set(user_payload) == {
        "task",
        "scenario_id",
        "critical_entity_ids",
        "graph_snapshots",
    }
    assert user_payload["task"] == "derive an evidence-grounded defensive plan"
    assert "expected_sequence" not in messages[1]["content"]
    assert "expected_interpretation" not in messages[1]["content"]


def test_gold_holdout_generation_policy_is_deterministic() -> None:
    policy = GoldHoldoutGenerationPolicy()

    assert policy.do_sample is False
    assert policy.num_beams == 1
    assert policy.max_new_tokens == 1024
