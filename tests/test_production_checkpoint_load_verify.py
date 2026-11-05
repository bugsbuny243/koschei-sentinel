from __future__ import annotations

import base64
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.gold_holdout_evaluation import (
    GoldHoldoutCaseResult,
    GoldHoldoutEvaluationReport,
)
from koschei_sentinel.production_checkpoint_lineage import (
    _HOLDOUT_DOMAIN,
    _TRAINING_DOMAIN,
    IndependentHoldoutLineage,
    ProductionTrainingLineage,
    verify_production_checkpoint_lineage,
)
from koschei_sentinel.production_mcore_training_checkpoint import MCoreTrainingState
from koschei_sentinel.production_mcore_training_loop import (
    MCoreTrainingLoopResult,
    MCoreTrainingStepResult,
)
from koschei_sentinel.production_checkpoint_load_verify import (
    verify_rank_checkpoint_load,
)
from koschei_sentinel.production_checkpoint_manifest import (
    build_rank_checkpoint_manifest,
)
from koschei_sentinel.production_checkpoint_trust import (
    ApprovedRankCheckpointReceipt,
    checkpoint_approval_signature_message,
    verify_approved_rank_checkpoint_load,
)
from koschei_sentinel.production_megatron_model_spec import (
    load_production_megatron_model_spec,
)
from koschei_sentinel.production_model_source import ProductionModelSourceIntake
from koschei_sentinel.production_target_binding import ProductionTargetBinding
from koschei_sentinel.production_target_provenance import (
    owner_key_fingerprint,
    production_target_binding_digest,
)
from koschei_sentinel.production_target_provenance_authoring import (
    build_signed_production_target_provenance,
)

CANDIDATE = Path("configs/training/production-megatron-model.397b-35b.candidate.json")


class _UnsupportedCheckpointEntry:
    """An ordinary Python object that the tensor-only loader must reject."""



def test_rank_checkpoint_load_verifies_meta_shapes(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    spec = load_production_megatron_model_spec(CANDIDATE)
    manifest = build_rank_checkpoint_manifest(
        spec,
        pipeline_rank=0,
        tensor_rank=0,
        expert_rank=0,
    )
    state_dict = {
        item.name: torch.empty(item.shard_shape, dtype=torch.bfloat16, device="meta")
        for item in manifest.tensors
    }
    path = tmp_path / "rank.pt"
    torch.save(
        {
            "schema_version": "sentinel.rank-weight-checkpoint.v1",
            "manifest_sha256": manifest.manifest_sha256,
            "pipeline_rank": 0,
            "tensor_rank": 0,
            "expert_rank": 0,
            "dtype": "bfloat16",
            "state_dict": state_dict,
        },
        path,
    )
    result = verify_rank_checkpoint_load(manifest, checkpoint_path=path)
    assert result.status == "metadata_verified"
    assert result.map_location == "meta"
    assert result.real_storage_allocated is False
    assert result.forward_verified is False
    assert result.backward_verified is False


def test_rank_checkpoint_load_rejects_manifest_mismatch(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    spec = load_production_megatron_model_spec(CANDIDATE)
    manifest = build_rank_checkpoint_manifest(
        spec,
        pipeline_rank=3,
        tensor_rank=7,
        expert_rank=15,
    )
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "schema_version": "sentinel.rank-weight-checkpoint.v1",
            "manifest_sha256": "0" * 64,
            "pipeline_rank": 3,
            "tensor_rank": 7,
            "expert_rank": 15,
            "dtype": "bfloat16",
            "state_dict": {},
        },
        path,
    )
    with pytest.raises(ValueError, match="manifest digest mismatch"):
        verify_rank_checkpoint_load(manifest, checkpoint_path=path)

def test_rank_checkpoint_load_rejects_unallowlisted_python_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    # The explicit weights_only flag must not be weakened by this environment setting.
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    spec = load_production_megatron_model_spec(CANDIDATE)
    manifest = build_rank_checkpoint_manifest(
        spec,
        pipeline_rank=0,
        tensor_rank=0,
        expert_rank=0,
    )
    path = tmp_path / "unexpected-object.pt"
    torch.save(
        {
            "schema_version": "sentinel.rank-weight-checkpoint.v1",
            "manifest_sha256": manifest.manifest_sha256,
            "pipeline_rank": 0,
            "tensor_rank": 0,
            "expert_rank": 0,
            "dtype": "bfloat16",
            "state_dict": {"unexpected": _UnsupportedCheckpointEntry()},
        },
        path,
    )

    with pytest.raises(ValueError, match="unable to load checkpoint shard metadata"):
        verify_rank_checkpoint_load(manifest, checkpoint_path=path)


def _canonical_digest(value: dict[str, object]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def approved_case(tmp_path_factory):
    torch = pytest.importorskip("torch")
    root = tmp_path_factory.mktemp("approved-checkpoint")
    spec = load_production_megatron_model_spec(CANDIDATE)
    manifest = build_rank_checkpoint_manifest(
        spec, pipeline_rank=0, tensor_rank=0, expert_rank=0
    )
    path = root / "approved-rank.pt"
    state_dict = {
        item.name: torch.empty(item.shard_shape, dtype=torch.bfloat16, device="meta")
        for item in manifest.tensors
    }
    torch.save(
        {
            "schema_version": "sentinel.rank-weight-checkpoint.v1",
            "manifest_sha256": manifest.manifest_sha256,
            "pipeline_rank": 0,
            "tensor_rank": 0,
            "expert_rank": 0,
            "dtype": "bfloat16",
            "state_dict": state_dict,
        },
        path,
    )

    private_key = Ed25519PrivateKey.generate()
    binding = ProductionTargetBinding(
        model_ref="koschei/sentinel-397b-35b",
        model_revision="a" * 64,
        architecture_manifest_sha256="1" * 64,
        router_manifest_sha256="2" * 64,
        expert_topology_manifest_sha256="3" * 64,
        trainer_adapter_version="4.5.2",
        production_training_allowed=True,
        verification_status="verified",
        blockers=[],
    )
    intake_payload = {
        "schema_version": "sentinel.production-model-source-intake.v1",
        "model_ref": binding.model_ref,
        "model_revision": binding.model_revision,
        "source_kind": "internal-artifact",
        "source_ref": "pinned://test/base",
        "source_revision": "b" * 64,
        "source_payload_sha256": "c" * 64,
        "license_ref": "internal://license/reviewed",
        "architecture_evidence_sha256": "4" * 64,
        "router_evidence_sha256": "5" * 64,
        "expert_topology_evidence_sha256": "6" * 64,
        "tokenizer_template_evidence_sha256": "7" * 64,
        "framework_compatibility_evidence_sha256": "8" * 64,
        "independently_reviewed": True,
        "review_ref": "review://checkpoint-trust-test",
    }
    intake = ProductionModelSourceIntake.model_validate(
        {**intake_payload, "intake_sha256": _canonical_digest(intake_payload)}
    )
    provenance = build_signed_production_target_provenance(
        binding=binding, intake=intake, owner_private_key=private_key
    )
    approval_payload = {
        "schema_version": "sentinel.approved-rank-checkpoint.v1",
        "model_ref": binding.model_ref,
        "model_revision": binding.model_revision,
        "target_binding_sha256": production_target_binding_digest(binding),
        "target_provenance_sha256": provenance.provenance_sha256,
        "initialization_manifest_sha256": manifest.manifest_sha256,
        "pipeline_rank": manifest.pipeline_rank,
        "tensor_rank": manifest.tensor_rank,
        "expert_rank": manifest.expert_rank,
        "checkpoint_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checkpoint_file_size_bytes": path.stat().st_size,
        "owner_key_fingerprint": owner_key_fingerprint(private_key.public_key()),
    }
    receipt_sha256 = _canonical_digest(approval_payload)
    receipt = ApprovedRankCheckpointReceipt.model_validate(
        {
            **approval_payload,
            "receipt_sha256": receipt_sha256,
            "owner_signature_b64": base64.b64encode(
                private_key.sign(checkpoint_approval_signature_message(receipt_sha256))
            ).decode("ascii"),
        }
    )
    return manifest, path, receipt, binding, provenance, private_key.public_key()


def _verify_approved(case, *, path=None, receipt=None, public_key=None, binding=None):
    manifest, file_path, approved, target, provenance, owner_key = case
    return verify_approved_rank_checkpoint_load(
        manifest,
        checkpoint_path=str(path or file_path),
        receipt=receipt or approved,
        target_binding=binding or target,
        target_provenance=provenance,
        trusted_owner_public_key=public_key or owner_key,
    )


def test_approved_checkpoint_binds_signed_model_and_artifact(approved_case) -> None:
    result = _verify_approved(approved_case)
    assert result.status == "approved_metadata_verified"
    assert result.owner_signature_verified is True
    assert result.source_provenance_verified is True
    assert result.artifact_bytes_verified is True
    assert result.execution_authorized is False
    assert result.metadata.forward_verified is False


def test_approved_checkpoint_rejects_wrong_owner_key_before_deserialization(
    approved_case, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        torch, "load", lambda *_args, **_kwargs: pytest.fail("must not deserialize")
    )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _verify_approved(
            approved_case,
            public_key=Ed25519PrivateKey.generate().public_key(),
        )


def test_approved_checkpoint_rejects_changed_bytes_before_deserialization(
    approved_case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "mutated-rank.pt"
    shutil.copyfile(approved_case[1], path)
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        last_byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([last_byte[0] ^ 1]))
    monkeypatch.setattr(
        torch, "load", lambda *_args, **_kwargs: pytest.fail("must not deserialize")
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _verify_approved(approved_case, path=path)


def test_approved_checkpoint_rejects_forged_model_identity_before_loading(
    approved_case, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        torch, "load", lambda *_args, **_kwargs: pytest.fail("must not deserialize")
    )
    receipt = approved_case[2].model_copy(update={"model_ref": "another/model"})
    with pytest.raises(ValueError, match="self-hash mismatch"):
        _verify_approved(approved_case, receipt=receipt)


def test_approved_checkpoint_rejects_bad_signature_before_loading(
    approved_case, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        torch, "load", lambda *_args, **_kwargs: pytest.fail("must not deserialize")
    )
    receipt = approved_case[2].model_copy(
        update={"owner_signature_b64": base64.b64encode(b"bad").decode("ascii")}
    )
    with pytest.raises(ValueError, match="signature verification failed"):
        _verify_approved(approved_case, receipt=receipt)


def test_approved_checkpoint_requires_verified_training_target(
    approved_case, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        torch, "load", lambda *_args, **_kwargs: pytest.fail("must not deserialize")
    )
    binding = approved_case[3].model_copy(
        update={
            "production_training_allowed": False,
            "verification_status": "unverified",
            "blockers": ["not approved"],
        }
    )
    with pytest.raises(ValueError, match="not verified"):
        _verify_approved(approved_case, binding=binding)


def test_approved_checkpoint_rejects_symlink(
    approved_case, tmp_path: Path
) -> None:
    path = tmp_path / "symlink.pt"
    path.symlink_to(approved_case[1])
    with pytest.raises(ValueError, match="cannot be opened safely"):
        _verify_approved(approved_case, path=path)


def test_artifact_pin_arguments_must_be_complete(approved_case) -> None:
    manifest, path = approved_case[:2]
    with pytest.raises(ValueError, match="supplied together"):
        verify_rank_checkpoint_load(
            manifest, checkpoint_path=path, expected_file_sha256="a" * 64
        )


@pytest.fixture(scope="module")
def lineage_case(approved_case):
    _, _, approval, binding, _, _ = approved_case
    operator_key = Ed25519PrivateKey.generate()
    reviewer_key = Ed25519PrivateKey.generate()
    training_state = MCoreTrainingState(
        global_step=5,
        consumed_microbatches=10,
        learning_rate=0.001,
        global_seed=39735,
    )
    training_loop = MCoreTrainingLoopResult(
        start_step=0,
        final_step=5,
        resumed_from_checkpoint=False,
        step_results=[
            MCoreTrainingStepResult(
                global_step=5,
                consumed_microbatches=10,
                learning_rate=0.001,
                grad_norm=0.5,
            )
        ],
        final_checkpoint_dir="synthetic/checkpoint-5",
    )
    training_payload = {
        "schema_version": "sentinel.production-training-lineage.v1",
        "model_ref": binding.model_ref,
        "model_revision": binding.model_revision,
        "checkpoint_approval_sha256": approval.receipt_sha256,
        "checkpoint_file_sha256": approval.checkpoint_file_sha256,
        "training_run_id": "synthetic-run-005",
        "training_state_sha256": _canonical_digest(training_state.model_dump(mode="json")),
        "training_loop_sha256": _canonical_digest(training_loop.model_dump(mode="json")),
        "dataset_manifest_sha256": "b" * 64,
        "final_step": 5,
        "signer_key_fingerprint": owner_key_fingerprint(operator_key.public_key()),
    }
    training_sha = _canonical_digest(training_payload)
    training_lineage = ProductionTrainingLineage.model_validate(
        {
            **training_payload,
            "attestation_sha256": training_sha,
            "operator_signature_b64": base64.b64encode(
                operator_key.sign(_TRAINING_DOMAIN + training_sha.encode("ascii"))
            ).decode("ascii"),
        }
    )

    results = [
        GoldHoldoutCaseResult(
            case_id=f"synthetic-case-{i:03d}",
            scenario_id=f"synthetic-scenario-{i:03d}",
            structural_exact=True,
            violations=[],
        )
        for i in range(50)
    ]
    report_payload = {
        "schema_version": "sentinel.gold-holdout-evaluation-report.v2",
        "model_ref": binding.model_ref,
        "model_revision": binding.model_revision,
        "adapter_digest": "c" * 64,
        "case_count": 50,
        "prediction_count": 50,
        "missing_case_ids": [],
        "extra_case_ids": [],
        "structural_exact_cases": 50,
        "structural_exact_rate": 1.0,
        "compared_steps": 50,
        "predicted_steps": 50,
        "mode_accuracy": 1.0,
        "action_accuracy": 1.0,
        "target_accuracy": 1.0,
        "evidence_selection_accuracy": 1.0,
        "evidence_grounding_rate": 1.0,
        "target_grounding_rate": 1.0,
        "outcome_verification_rate": 1.0,
        "case_results": [case.model_dump(mode="json") for case in results],
        "passed": True,
        "violations": [],
    }
    holdout_report = GoldHoldoutEvaluationReport.model_validate(
        {**report_payload, "report_sha256": _canonical_digest(report_payload)}
    )
    holdout_payload = {
        "schema_version": "sentinel.independent-holdout-lineage.v1",
        "model_ref": binding.model_ref,
        "model_revision": binding.model_revision,
        "checkpoint_approval_sha256": approval.receipt_sha256,
        "checkpoint_file_sha256": approval.checkpoint_file_sha256,
        "training_attestation_sha256": training_lineage.attestation_sha256,
        "holdout_report_sha256": holdout_report.report_sha256,
        "evaluation_dataset_sha256": "d" * 64,
        "evaluation_policy_sha256": "e" * 64,
        "case_count": 50,
        "signer_key_fingerprint": owner_key_fingerprint(reviewer_key.public_key()),
    }
    holdout_sha = _canonical_digest(holdout_payload)
    holdout_lineage = IndependentHoldoutLineage.model_validate(
        {
            **holdout_payload,
            "attestation_sha256": holdout_sha,
            "reviewer_signature_b64": base64.b64encode(
                reviewer_key.sign(_HOLDOUT_DOMAIN + holdout_sha.encode("ascii"))
            ).decode("ascii"),
        }
    )
    return {
        "approved_case": approved_case,
        "training_state": training_state,
        "training_loop": training_loop,
        "training_lineage": training_lineage,
        "training_operator_key": operator_key.public_key(),
        "holdout_report": holdout_report,
        "holdout_lineage": holdout_lineage,
        "reviewer_key": reviewer_key.public_key(),
    }


def _verify_lineage(case, **overrides):
    manifest, path, approval, binding, provenance, owner_key = case["approved_case"]
    args = {
        "checkpoint_path": str(path),
        "approved_receipt": approval,
        "binding": binding,
        "provenance": provenance,
        "trusted_owner_key": owner_key,
        "training_state": case["training_state"],
        "training_loop": case["training_loop"],
        "training_lineage": case["training_lineage"],
        "trusted_training_operator_key": case["training_operator_key"],
        "holdout_report": case["holdout_report"],
        "holdout_lineage": case["holdout_lineage"],
        "trusted_reviewer_key": case["reviewer_key"],
    }
    args.update(overrides)
    return verify_production_checkpoint_lineage(manifest, **args)


def test_checkpoint_lineage_verifies_signed_claims_only(lineage_case) -> None:
    report = _verify_lineage(lineage_case)
    assert report.status == "signed_lineage_claims_verified"
    assert report.training_run_id == "synthetic-run-005"
    assert report.training_execution_independently_verified is False
    assert report.holdout_execution_independently_verified is False
    assert report.production_deployment_authorized is False


def test_checkpoint_lineage_rejects_owner_as_independent_reviewer(lineage_case) -> None:
    owner = lineage_case["approved_case"][-1]
    with pytest.raises(ValueError, match="independent trust root"):
        _verify_lineage(lineage_case, trusted_reviewer_key=owner)


def test_checkpoint_lineage_rejects_altered_training_run(lineage_case) -> None:
    original = lineage_case["training_lineage"]
    tampered = original.model_copy(update={"training_run_id": "fabricated-run"})
    with pytest.raises(ValueError, match="attestation self-hash"):
        _verify_lineage(lineage_case, training_lineage=tampered)


def test_checkpoint_lineage_rejects_holdout_other_checkpoint(lineage_case) -> None:
    original = lineage_case["holdout_lineage"]
    tampered = original.model_copy(update={"checkpoint_file_sha256": "f" * 64})
    with pytest.raises(ValueError, match="checkpoint file digest mismatch"):
        _verify_lineage(lineage_case, holdout_lineage=tampered)


def test_checkpoint_lineage_rejects_unbound_holdout_training(lineage_case) -> None:
    original = lineage_case["holdout_lineage"]
    tampered = original.model_copy(update={"training_attestation_sha256": "f" * 64})
    with pytest.raises(ValueError, match="self-hash mismatch"):
        _verify_lineage(lineage_case, holdout_lineage=tampered)


def test_checkpoint_lineage_rejects_tampered_holdout_report(lineage_case) -> None:
    original = lineage_case["holdout_report"]
    tampered = original.model_copy(update={"passed": False})
    with pytest.raises(ValueError, match="report self-hash mismatch"):
        _verify_lineage(lineage_case, holdout_report=tampered)


def test_checkpoint_lineage_rejects_fake_qwen_model_binding(lineage_case) -> None:
    original = lineage_case["training_lineage"]
    tampered = original.model_copy(update={"model_ref": "Qwen/Qwen3.5-397B-A17B"})
    with pytest.raises(ValueError, match="lineage model identity"):
        _verify_lineage(lineage_case, training_lineage=tampered)


def test_checkpoint_lineage_rejects_training_step_mismatch(lineage_case) -> None:
    training = lineage_case["training_state"].model_copy(update={"global_step": 4})
    with pytest.raises(ValueError, match="final step"):
        _verify_lineage(lineage_case, training_state=training)


def test_checkpoint_lineage_rejects_reviewer_signature_replay(lineage_case) -> None:
    original = lineage_case["holdout_lineage"]
    tampered = original.model_copy(
        update={"reviewer_signature_b64": lineage_case["training_lineage"].operator_signature_b64}
    )
    with pytest.raises(ValueError, match="signature verification failed"):
        _verify_lineage(lineage_case, holdout_lineage=tampered)
