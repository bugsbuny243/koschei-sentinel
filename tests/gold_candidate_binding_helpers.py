import hashlib
import json
from pathlib import Path

from koschei_sentinel.gold_holdout_inference_runner import (
    GoldHoldoutGenerationPolicy,
    GoldHoldoutInferenceRunReceipt,
    _digest_without,
    _export_verification_sha256,
    _load_candidate_identity,
    build_gold_holdout_inference_plan,
)
from tests.test_cyber_sft_export_verify import _build_export


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_gold_bound_candidate_export(tmp_path, monkeypatch, release: Path) -> Path:
    fixture_root = tmp_path / "gold-bound-candidate"
    fixture_root.mkdir()
    return _build_export(
        fixture_root,
        monkeypatch,
        promotion_eligible=True,
        corpus_examples_raw=(release / "train" / "examples.jsonl").read_bytes(),
        corpus_manifest_raw=(release / "train" / "manifest.json").read_bytes(),
        validation_examples_sha256=_sha(release / "validation" / "examples.jsonl"),
        validation_manifest_sha256=_sha(release / "validation" / "manifest.json"),
    )


def rebind_inference_fixture_to_gold_candidate(
    tmp_path,
    monkeypatch,
    *,
    release: Path,
    pack: Path,
    output: Path,
    model_ref: str,
):
    candidate = build_gold_bound_candidate_export(tmp_path, monkeypatch, release)
    policy = GoldHoldoutGenerationPolicy.model_validate_json(
        (output / "generation-policy.json").read_bytes()
    )
    plan = build_gold_holdout_inference_plan(
        inference_pack_dir=pack,
        candidate_export_dir=candidate,
        model_ref=model_ref,
        generation_policy=policy,
    )
    config, _manifest, _adapter, attestation, export_verification = (
        _load_candidate_identity(candidate_export_dir=candidate)
    )

    artifacts = {
        "plan.json": plan.model_dump(mode="json"),
        "training-config.json": config.model_dump(mode="json"),
        "run-attestation.json": attestation.model_dump(mode="json"),
        "candidate-export-verification.json": export_verification.model_dump(
            mode="json"
        ),
    }
    for name, payload in artifacts.items():
        (output / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    receipt = GoldHoldoutInferenceRunReceipt.model_validate_json(
        (output / "receipt.json").read_bytes()
    )
    receipt_payload = receipt.model_dump(mode="json")
    receipt_payload.update(
        {
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
                _export_verification_sha256(export_verification)
            ),
        }
    )
    receipt_payload["receipt_sha256"] = _digest_without(
        receipt_payload,
        "receipt_sha256",
    )
    (output / "receipt.json").write_text(
        json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return plan, candidate
