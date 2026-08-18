import hashlib
import json

from koschei_sentinel.defense_reflex_gold_release import write_gold_defense_release
from koschei_sentinel.gold_holdout_evaluation import export_gold_holdout_inference_pack
from koschei_sentinel.gold_holdout_inference_runner import (
    GoldHoldoutGenerationPolicy,
    GoldHoldoutInferencePlan,
    GoldHoldoutInferenceRunReceipt,
    _digest_without,
    _load_inference_pack,
    _policy_sha256,
    _prediction_from_generated_text,
)
from koschei_sentinel.gold_holdout_inference_verify import (
    verify_gold_holdout_inference_output,
)
from tests.test_defense_reflex_gold_release import _release_rows


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _prediction_text() -> str:
    return json.dumps(
        {
            "interpretation": "Collect corroborating evidence before escalating containment.",
            "defense_sequence": [
                {
                    "sequence": 1,
                    "expected_mode": "GUARD",
                    "action": "COLLECT_EVIDENCE",
                    "target_entity_id": "entity:test",
                    "rationale": "Evidence remains incomplete.",
                    "supporting_evidence_ids": ["evidence:test"],
                    "outcome_verification_required": True,
                }
            ],
        }
    )


def _fixture(tmp_path):
    _policy, rows = _release_rows()
    release = tmp_path / "gold-release"
    write_gold_defense_release(rows, release)
    pack = tmp_path / "holdout-pack"
    export_gold_holdout_inference_pack(release, pack)
    cases, inference_manifest, manifest_raw = _load_inference_pack(pack)

    generation_policy = GoldHoldoutGenerationPolicy()
    adapter_digest = "a" * 64
    plan_payload = {
        "schema_version": "sentinel.gold-holdout-inference-plan.v1",
        "model_ref": "koschei-sentinel:test",
        "model_revision": adapter_digest,
        "adapter_digest": adapter_digest,
        "base_model": "Qwen/Qwen3.5-9B-Base",
        "base_revision": "b" * 40,
        "run_id": "gold-holdout-test",
        "case_count": len(cases),
        "inputs_sha256": inference_manifest.inputs_sha256,
        "inference_manifest_sha256": _sha256_bytes(manifest_raw),
        "generation_policy_sha256": _policy_sha256(generation_policy),
        "answer_key_isolated": True,
        "deterministic_generation": True,
    }
    plan_payload["plan_sha256"] = _digest_without(plan_payload, "plan_sha256")
    plan = GoldHoldoutInferencePlan.model_validate(plan_payload)

    predictions = [
        _prediction_from_generated_text(
            case=case,
            generated_text=_prediction_text(),
            model_ref=plan.model_ref,
            adapter_digest=adapter_digest,
        )
        for case in cases
    ]
    prediction_payload = "".join(
        json.dumps(row.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in predictions
    )
    failure_payload = ""
    receipt_payload = {
        "schema_version": "sentinel.gold-holdout-inference-run-receipt.v1",
        "plan_sha256": plan.plan_sha256,
        "model_ref": plan.model_ref,
        "model_revision": adapter_digest,
        "adapter_digest": adapter_digest,
        "base_model": plan.base_model,
        "base_revision": plan.base_revision,
        "case_count": len(cases),
        "prediction_count": len(predictions),
        "failure_count": 0,
        "failed_case_ids": [],
        "predictions_sha256": _sha256_bytes(prediction_payload.encode("utf-8")),
        "failures_sha256": _sha256_bytes(failure_payload.encode("utf-8")),
        "cuda_device_index": 0,
        "cuda_device_name": "fixture-gpu",
        "runtime_versions": {"torch": "fixture"},
    }
    receipt_payload["receipt_sha256"] = _digest_without(receipt_payload, "receipt_sha256")
    receipt = GoldHoldoutInferenceRunReceipt.model_validate(receipt_payload)

    output = tmp_path / "inference-output"
    output.mkdir()
    (output / "plan.json").write_text(
        json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "receipt.json").write_text(
        json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "generation-policy.json").write_text(
        json.dumps(generation_policy.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "predictions.jsonl").write_text(prediction_payload, encoding="utf-8")
    (output / "failures.jsonl").write_text(failure_payload, encoding="utf-8")
    return pack, output, cases, plan


def test_offline_inference_verifier_accepts_complete_bound_output(tmp_path) -> None:
    pack, output, cases, _plan = _fixture(tmp_path)

    report = verify_gold_holdout_inference_output(output, pack)

    assert report.valid is True
    assert report.case_count == len(cases)
    assert report.prediction_count == len(cases)
    assert report.failure_count == 0
    assert report.plan_verified is True
    assert report.receipt_verified is True
    assert report.input_binding_verified is True
    assert report.generation_policy_verified is True
    assert report.prediction_hashes_verified is True
    assert report.identity_verified is True
    assert report.complete_case_accounting is True
    assert report.violations == []


def test_offline_inference_verifier_rejects_generation_policy_drift(tmp_path) -> None:
    pack, output, _cases, _plan = _fixture(tmp_path)
    policy_path = output / "generation-policy.json"
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    payload["max_new_tokens"] = 2048
    policy_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_gold_holdout_inference_output(output, pack)

    assert report.valid is False
    assert report.generation_policy_verified is False
    assert any("generation policy SHA differs" in row for row in report.violations)


def test_offline_inference_verifier_rejects_missing_case_even_with_rewritten_receipt(tmp_path) -> None:
    pack, output, cases, plan = _fixture(tmp_path)
    prediction_lines = (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    shortened = "\n".join(prediction_lines[:-1]) + ("\n" if prediction_lines[:-1] else "")
    (output / "predictions.jsonl").write_text(shortened, encoding="utf-8")

    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    receipt["prediction_count"] = len(cases) - 1
    receipt["predictions_sha256"] = _sha256_bytes(shortened.encode("utf-8"))
    receipt["receipt_sha256"] = _digest_without(receipt, "receipt_sha256")
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_gold_holdout_inference_output(output, pack)

    assert report.valid is False
    assert report.complete_case_accounting is False
    assert any("omits cases" in row for row in report.violations)
    assert plan.case_count == len(cases)
