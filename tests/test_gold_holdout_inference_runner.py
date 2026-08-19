import hashlib
import json
from types import SimpleNamespace

import pytest

import koschei_sentinel.gold_holdout_inference_runner as runner_module
from koschei_sentinel.cyber_sft_run_attestation import (
    _attestation_digest,
    _config_sha256,
)
from koschei_sentinel.cyber_sft_training import load_cyber_sft_config
from koschei_sentinel.defense_reflex_gold_release import write_gold_defense_release
from koschei_sentinel.gold_holdout_evaluation import export_gold_holdout_inference_pack
from koschei_sentinel.gold_holdout_inference_runner import (
    GoldHoldoutGenerationPolicy,
    _load_candidate_identity,
    _load_inference_pack,
    _prediction_from_generated_text,
    _prompt_messages,
)
from koschei_sentinel.training import canonical_json
from tests.test_defense_reflex_gold_release import _release_rows


def _pack(tmp_path):
    _policy, rows = _release_rows()
    release = tmp_path / "gold-release"
    write_gold_defense_release(rows, release)
    pack = tmp_path / "holdout-pack"
    export_gold_holdout_inference_pack(release, pack)
    return pack


def _candidate_fixture(tmp_path):
    export_root = tmp_path / "candidate-export"
    export_root.mkdir()
    config_path = export_root / "training-config.json"
    config_payload = {
        "schema_version": "sentinel.cyber-sft-config.v1",
        "run_id": "gold-inference-run",
        "stage": "DEFENSE_REFLEX",
        "execution_profile": "DENSE_SINGLE_GPU_QLORA",
        "base_model": "Qwen/Qwen3.5-9B-Base",
        "base_revision": "a" * 40,
        "corpus_dir": "build/gold/train",
        "validation_corpus_dir": "build/gold/validation",
        "output_dir": "build/run",
        "input_adapter_dir": None,
        "max_sequence_length": 2048,
        "epochs": 1.0,
        "learning_rate": 0.0001,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "validation_ratio": 0.0,
        "warmup_ratio": 0.03,
        "logging_steps": 1,
        "seed": 1701,
        "minimum_cuda_memory_gb": 0.0,
        "gradient_checkpointing": True,
        "enable_router_aux_loss": False,
        "quantization": {
            "bits": 4,
            "quant_type": "nf4",
            "double_quant": True,
            "compute_dtype": "float16",
        },
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_suffixes": ["q_proj"],
        },
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    config = load_cyber_sft_config(config_path)

    run = export_root / "run"
    adapter = run / "adapter"
    adapter.mkdir(parents=True)
    manifest = {
        "schema_version": "sentinel.cyber-sft-adapter-manifest.v1",
        "run_id": config.run_id,
        "stage": "DEFENSE_REFLEX",
        "execution_profile": "DENSE_SINGLE_GPU_QLORA",
        "base_model": config.base_model,
        "base_revision": config.base_revision,
        "corpus_examples_sha256": "b" * 64,
        "corpus_manifest_sha256": "c" * 64,
        "corpus_promotion_eligible": True,
        "input_adapter_dir": None,
        "adapter_digest": "d" * 64,
        "adapter_files": ["adapter/adapter_model.safetensors"],
        "trainable_target_module_count": 1,
        "trainable_target_modules_sha256": "e" * 64,
        "training_examples": 1,
        "validation_examples": 1,
        "gradient_checkpointing": True,
        "optimizer": "paged_adamw_8bit",
        "output_dir": config.output_dir,
    }
    (run / "adapter-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    attestation_payload = {
        "schema_version": "sentinel.cyber-sft-run-attestation.v1",
        "run_id": config.run_id,
        "selected_profile": "normal",
        "repository_commit": "f" * 40,
        "base_model": config.base_model,
        "base_revision": config.base_revision,
        "resolved_model_revision": config.base_revision,
        "config_sha256": _config_sha256(config),
        "plan_sha256": "1" * 64,
        "training_source_sha256": "2" * 64,
        "model_preflight_sha256": "3" * 64,
        "verification_sha256": "4" * 64,
        "model_runtime_sha256": "5" * 64,
        "resume_runtime_sha256": "6" * 64,
        "receipt_sha256": "7" * 64,
        "adapter_digest": manifest["adapter_digest"],
        "corpus_examples_sha256": manifest["corpus_examples_sha256"],
        "corpus_manifest_sha256": manifest["corpus_manifest_sha256"],
        "global_step": 1,
        "resumed": False,
        "resume_checkpoint": None,
        "smoke_only": False,
        "promotion_eligible": True,
    }
    attestation_payload["attestation_sha256"] = _attestation_digest(attestation_payload)
    (export_root / "run-attestation.json").write_text(
        json.dumps(attestation_payload),
        encoding="utf-8",
    )
    return export_root, config_path, adapter


def _valid_export_report():
    return SimpleNamespace(valid=True, violations=[])


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


def test_candidate_identity_resolves_verified_export_adapter(monkeypatch, tmp_path) -> None:
    export_root, _config_path, adapter = _candidate_fixture(tmp_path)
    monkeypatch.setattr(
        runner_module,
        "verify_cyber_sft_export",
        lambda *_args, **_kwargs: _valid_export_report(),
    )

    _config, manifest, resolved, attestation, _verification = _load_candidate_identity(
        candidate_export_dir=export_root,
    )

    assert resolved == adapter
    assert manifest.adapter_digest == "d" * 64
    assert attestation.adapter_digest == manifest.adapter_digest


def test_candidate_identity_rejects_missing_adapter_subdirectory(monkeypatch, tmp_path) -> None:
    export_root, _config_path, adapter = _candidate_fixture(tmp_path)
    adapter.rmdir()
    monkeypatch.setattr(
        runner_module,
        "verify_cyber_sft_export",
        lambda *_args, **_kwargs: _valid_export_report(),
    )

    with pytest.raises(ValueError, match="missing its adapter directory"):
        _load_candidate_identity(candidate_export_dir=export_root)


def test_candidate_identity_rejects_training_config_drift_even_with_stale_valid_report(
    monkeypatch,
    tmp_path,
) -> None:
    export_root, config_path, _adapter = _candidate_fixture(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["max_sequence_length"] = 4096
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        runner_module,
        "verify_cyber_sft_export",
        lambda *_args, **_kwargs: _valid_export_report(),
    )

    with pytest.raises(ValueError, match="training config SHA differs"):
        _load_candidate_identity(candidate_export_dir=export_root)


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
