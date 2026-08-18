import hashlib
import json

import pytest

from koschei_sentinel.defense_reflex_gold_release import GoldHoldoutEvaluationCase
from koschei_sentinel.gold_holdout_evaluation import (
    GoldHoldoutPredictedStep,
    build_gold_holdout_prediction,
)
from koschei_sentinel.gold_holdout_evaluation_evidence import (
    build_gold_holdout_evaluation_evidence,
    verify_gold_holdout_evaluation_evidence,
)
from koschei_sentinel.gold_holdout_inference_runner import (
    GoldHoldoutInferenceRunReceipt,
    _digest_without,
)
from tests.test_gold_holdout_inference_verify import _fixture


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _passing_fixture(tmp_path):
    pack, output, inference_cases, plan = _fixture(tmp_path)
    release = tmp_path / "gold-release"
    gold_line = (release / "holdout" / "cases.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    gold_case = GoldHoldoutEvaluationCase.model_validate_json(gold_line)
    steps = [GoldHoldoutPredictedStep.model_validate(row) for row in gold_case.expected_sequence]
    prediction = build_gold_holdout_prediction(
        inference_case=inference_cases[0],
        model_ref=plan.model_ref,
        model_revision=plan.adapter_digest,
        adapter_digest=plan.adapter_digest,
        interpretation="Evidence supports the reviewed defensive sequence.",
        defense_sequence=steps,
    )
    prediction_payload = (
        json.dumps(
            prediction.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    (output / "predictions.jsonl").write_text(prediction_payload, encoding="utf-8")

    receipt = GoldHoldoutInferenceRunReceipt.model_validate_json(
        (output / "receipt.json").read_bytes()
    )
    receipt_payload = receipt.model_dump(mode="json")
    receipt_payload["prediction_count"] = 1
    receipt_payload["failure_count"] = 0
    receipt_payload["failed_case_ids"] = []
    receipt_payload["predictions_sha256"] = _sha256_bytes(prediction_payload.encode("utf-8"))
    receipt_payload["failures_sha256"] = _sha256_bytes(b"")
    receipt_payload["receipt_sha256"] = _digest_without(receipt_payload, "receipt_sha256")
    (output / "receipt.json").write_text(
        json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return release, pack, output, plan


def test_verified_inference_builds_passing_gold_evidence(tmp_path) -> None:
    release, pack, output, plan = _passing_fixture(tmp_path)

    evidence = build_gold_holdout_evaluation_evidence(
        release_dir=release,
        inference_pack_dir=pack,
        inference_output_dir=output,
    )

    assert evidence.passed is True
    assert evidence.failure_count == 0
    assert evidence.complete_case_accounting is True
    assert evidence.inference_verification_valid is True
    assert evidence.model_revision == plan.adapter_digest
    assert evidence.adapter_digest == plan.adapter_digest
    assert evidence.report.passed is True
    assert evidence.report.structural_exact_rate == 1.0
    verify_gold_holdout_evaluation_evidence(evidence)


def test_gold_evidence_rejects_pack_from_different_release_audit(tmp_path) -> None:
    release, pack, output, _plan = _passing_fixture(tmp_path)
    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_gold_audit_sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="different release audit"):
        build_gold_holdout_evaluation_evidence(
            release_dir=release,
            inference_pack_dir=pack,
            inference_output_dir=output,
        )


def test_gold_evidence_self_hash_detects_tamper(tmp_path) -> None:
    release, pack, output, _plan = _passing_fixture(tmp_path)
    evidence = build_gold_holdout_evaluation_evidence(
        release_dir=release,
        inference_pack_dir=pack,
        inference_output_dir=output,
    )
    tampered = evidence.model_copy(update={"inference_inputs_sha256": "0" * 64})

    with pytest.raises(ValueError, match="evidence self-hash does not verify"):
        verify_gold_holdout_evaluation_evidence(tampered)
