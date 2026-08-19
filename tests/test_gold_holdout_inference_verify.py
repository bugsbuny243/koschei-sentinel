import hashlib
import json

from koschei_sentinel.defense_reflex_gold_release import write_gold_defense_release
from koschei_sentinel.gold_holdout_evaluation import export_gold_holdout_inference_pack
from koschei_sentinel.gold_holdout_inference_runner import (
    GoldHoldoutGenerationPolicy,
    GoldHoldoutInferenceRunReceipt,
    _digest_without,
    _load_candidate_identity,
    _load_inference_pack,
    _prediction_from_generated_text,
    build_gold_holdout_inference_plan,
)
from koschei_sentinel.gold_holdout_inference_verify import (
    verify_gold_holdout_inference_output,
)
from tests.test_cyber_sft_export_verify import _build_export
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


def _fixture(tmp_path, monkeypatch):
    _policy, rows = _release_rows()
    release = tmp_path / "gold-release"
    write_gold_defense_release(rows, release)
    pack = tmp_path / "holdout-pack"
    export_gold_holdout_inference_pack(release, pack)
    cases, _inference_manifest, _manifest_raw = _load_inference_pack(pack)

    candidate_export = _build_export(
        tmp_path,
        monkeypatch,
        promotion_eligible=True,
    )
    generation_policy = GoldHoldoutGenerationPolicy()
    plan = build_gold_holdout_inference_plan(
        inference_pack_dir=pack,
        candidate_export_dir=candidate_export,
        model_ref="koschei-sentinel:test",
        generation_policy=generation_policy,
    )
    config, _manifest, _adapter, attestation, export_verification = _load_candidate_identity(
        candidate_export_dir=candidate_export
    )

    predictions = [
        _prediction_from_generated_text(
            case=case,
            generated_text=_prediction_text(),
            model_ref=plan.model_ref,
            adapter_digest=plan.adapter_digest,
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
        "schema_version": "sentinel.gold-holdout-inference-run-receipt.v2",
        "plan_sha256": plan.plan_sha256,
        "run_id": plan.run_id,
        "model_ref": plan.model_ref,
        "model_revision": plan.adapter_digest,
        "adapter_digest": plan.adapter_digest,
        "base_model": plan.base_model,
        "base_revision": plan.base_revision,
        "training_config_sha256": plan.training_config_sha256,
        "run_attestation_sha256": plan.run_attestation_sha256,
        "candidate_export_verification_sha256": (
            plan.candidate_export_verification_sha256
        ),
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
    artifacts = {
        "plan.json": plan.model_dump(mode="json"),
        "receipt.json": receipt.model_dump(mode="json"),
        "generation-policy.json": generation_policy.model_dump(mode="json"),
        "training-config.json": config.model_dump(mode="json"),
        "run-attestation.json": attestation.model_dump(mode="json"),
        "candidate-export-verification.json": export_verification.model_dump(mode="json"),
    }
    for name, payload in artifacts.items():
        (output / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (output / "predictions.jsonl").write_text(prediction_payload, encoding="utf-8")
    (output / "failures.jsonl").write_text(failure_payload, encoding="utf-8")
    return pack, output, cases, plan, candidate_export


def test_offline_inference_verifier_accepts_complete_bound_output(
    tmp_path,
    monkeypatch,
) -> None:
    pack, output, cases, _plan, candidate_export = _fixture(tmp_path, monkeypatch)

    report = verify_gold_holdout_inference_output(output, pack, candidate_export)

    assert report.schema_version == "sentinel.gold-holdout-inference-verification.v3"
    assert report.valid is True
    assert report.case_count == len(cases)
    assert report.prediction_count == len(cases)
    assert report.failure_count == 0
    assert report.plan_verified is True
    assert report.receipt_verified is True
    assert report.input_binding_verified is True
    assert report.generation_policy_verified is True
    assert report.training_config_verified is True
    assert report.run_attestation_verified is True
    assert report.candidate_export_verification_verified is True
    assert report.prediction_hashes_verified is True
    assert report.identity_verified is True
    assert report.complete_case_accounting is True
    assert report.violations == []


def test_offline_inference_verifier_rejects_generation_policy_drift(
    tmp_path,
    monkeypatch,
) -> None:
    pack, output, _cases, _plan, candidate_export = _fixture(tmp_path, monkeypatch)
    policy_path = output / "generation-policy.json"
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    payload["max_new_tokens"] = 2048
    policy_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_gold_holdout_inference_output(output, pack, candidate_export)

    assert report.valid is False
    assert report.generation_policy_verified is False
    assert any("generation policy SHA differs" in row for row in report.violations)


def test_offline_inference_verifier_rejects_training_config_drift(
    tmp_path,
    monkeypatch,
) -> None:
    pack, output, _cases, _plan, candidate_export = _fixture(tmp_path, monkeypatch)
    config_path = output / "training-config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["max_sequence_length"] = 8192
    config_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_gold_holdout_inference_output(output, pack, candidate_export)

    assert report.valid is False
    assert report.training_config_verified is False
    assert any("training config SHA differs" in row for row in report.violations)


def test_offline_inference_verifier_rejects_candidate_export_drift(
    tmp_path,
    monkeypatch,
) -> None:
    pack, output, _cases, _plan, candidate_export = _fixture(tmp_path, monkeypatch)
    config_path = candidate_export / "training-config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["max_sequence_length"] = 8192
    config_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_gold_holdout_inference_output(output, pack, candidate_export)

    assert report.valid is False
    assert any("candidate" in row.lower() for row in report.violations)


def test_offline_inference_verifier_rejects_missing_case_even_with_rewritten_receipt(
    tmp_path,
    monkeypatch,
) -> None:
    pack, output, cases, plan, candidate_export = _fixture(tmp_path, monkeypatch)
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

    report = verify_gold_holdout_inference_output(output, pack, candidate_export)

    assert report.valid is False
    assert report.complete_case_accounting is False
    assert any("omits cases" in row for row in report.violations)
    assert plan.case_count == len(cases)
