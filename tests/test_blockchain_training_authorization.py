from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.blockchain_base_candidates import (
    BaseCandidateLane,
    BlockchainBaseCandidate,
    LicenseReviewStatus,
    RuntimeIntegrationStatus,
    build_base_candidate_registry,
)
from koschei_sentinel.blockchain_runtime_preflight import (
    BlockchainRuntimePreflightPolicy,
    BlockchainRuntimeProbeReceipt,
    audit_runtime_preflight,
    build_hardware_inventory,
    plan_runtime_preflight,
)
from koschei_sentinel.blockchain_security_corpus import (
    BlockchainSecurityDocument,
    BlockchainSourceClass,
    ChainFamily,
    ThreatDomain,
    content_digest,
)
from koschei_sentinel.blockchain_security_release import (
    BlockchainSecurityReleaseManifest,
    BlockchainSecurityReleaseSplit,
)
from koschei_sentinel.blockchain_training import BlockchainTrainingConfig
from koschei_sentinel.blockchain_training_authorization import (
    BlockchainRuntimePreflightSeal,
    BlockchainTrainingAuthorizationBlocked,
    BlockchainTrainingAuthorizationPolicy,
    _digest,
    approve_training_authorization_proposal,
    build_training_authorization_proposal,
    verify_training_authorization_bundle,
)
from koschei_sentinel.pretraining_corpus import RightsBasis
from koschei_sentinel.training import model_digest

MODEL_ID = "Qwen/Qwen2.5-Coder-7B"
REVISION = "0396a76181e127dfc13e5c5ec48a8cee09938b02"
CANDIDATE_ID = "qwen2.5-coder-7b-base"


def _write_model(path: Path, model) -> None:
    path.write_text(
        json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate() -> BlockchainBaseCandidate:
    return BlockchainBaseCandidate(
        candidate_id=CANDIDATE_ID,
        lane=BaseCandidateLane.FEASIBILITY,
        model_id=MODEL_ID,
        revision=REVISION,
        declared_total_parameters_billion=7.61,
        declared_active_parameters_billion=7.61,
        declared_context_length_tokens=131072,
        license_id="Apache-2.0",
        license_review_status=LicenseReviewStatus.ALLOWLISTED_OPEN_LICENSE,
        commercial_use_declared=True,
        trust_remote_code_required=False,
        runtime_integration_status=RuntimeIntegrationStatus.NATIVE_TRANSFORMERS_EXPECTED,
        blocked_reasons=["runtime_preflight_not_run", "hardware_plan_not_approved"],
    )


def _document(index: int, split: str) -> BlockchainSecurityDocument:
    character = "0123456789abcdef"[index]
    text = f"{split} authorized blockchain material {index}.\n"
    return BlockchainSecurityDocument(
        document_ref=f"doc_{character * 24}",
        source_class=BlockchainSourceClass.PROTOCOL_SOURCE,
        rights_basis=RightsBasis.KOSCHEI_OWNED,
        source_snapshot_digest=character * 64,
        content_digest=content_digest(text),
        family_refs=[f"family_{character * 24}"],
        chain_families=[ChainFamily.EVM],
        threat_domains=[ThreatDomain.PRIVILEGED_ACCESS],
        text=text,
    )


def _document_payload(documents: list[BlockchainSecurityDocument]) -> str:
    return "".join(
        json.dumps(
            item.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
        for item in documents
    )


def _materialize_release(root: Path) -> BlockchainSecurityReleaseManifest:
    release = root / "release"
    release.mkdir()
    split_documents = {
        "train": [_document(0, "train")],
        "validation": [_document(1, "validation")],
        "test": [_document(2, "test")],
    }
    reports: dict[str, BlockchainSecurityReleaseSplit] = {}
    all_documents: list[BlockchainSecurityDocument] = []
    for split_name, documents in split_documents.items():
        payload = _document_payload(documents)
        (release / f"{split_name}.jsonl").write_text(payload, encoding="utf-8")
        snapshot = documents[0].source_snapshot_digest
        reports[split_name] = BlockchainSecurityReleaseSplit(
            path=f"{split_name}.jsonl",
            documents=1,
            sources=1,
            components=1,
            families=1,
            digest=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            source_snapshot_digests=[snapshot],
            chain_source_counts={"EVM": 1},
            threat_source_counts={"PRIVILEGED_ACCESS": 1},
        )
        all_documents.extend(documents)

    source_payload = _document_payload(sorted(all_documents, key=lambda item: item.document_ref))
    source_digest = hashlib.sha256(source_payload.encode("utf-8")).hexdigest()
    manifest = BlockchainSecurityReleaseManifest(
        release_policy_id="authorization-test",
        split_seed="authorization-test-seed",
        catalog_file_digest="1" * 64,
        catalog_policy_file_digest="2" * 64,
        source_catalog_manifest_file_digest="3" * 64,
        source_catalog_digest="4" * 64,
        source_corpus_file_digest=source_digest,
        source_corpus_digest=source_digest,
        corpus_audit_file_digest="5" * 64,
        corpus_audit_digest="6" * 64,
        holdout_file_digest="7" * 64,
        holdout_digest="8" * 64,
        benchmark_suite_digest="9" * 64,
        blockchain_policy_file_digest="a" * 64,
        blockchain_policy_digest="b" * 64,
        release_policy_file_digest="c" * 64,
        release_policy_digest="d" * 64,
        documents=3,
        sources=3,
        components=3,
        family_leakage_detected=False,
        source_leakage_detected=False,
        splits=reports,
    )
    _write_model(release / "blockchain-security-release-manifest.json", manifest)
    return manifest


def _materialize_preflight(root: Path, registry) -> tuple[Path, Path]:
    run = root / "preflight-run"
    run.mkdir()
    hardware = build_hardware_inventory(
        inventory_id="authorization-test",
        cuda_available=True,
        gpu_count=1,
        gpu_model="Tesla T4",
        vram_per_gpu_mb=14912,
        host_ram_mb=128000,
        free_disk_mb=128000,
        compute_capability="7.5",
        bfloat16_supported=False,
    )
    policy = BlockchainRuntimePreflightPolicy(policy_id="authorization-test")
    plan = plan_runtime_preflight(registry, CANDIDATE_ID, hardware, policy)
    candidate = registry.candidates[0]
    receipt_payload = {
        "schema_version": "sentinel.blockchain-runtime-probe-receipt.v1",
        "candidate_id": CANDIDATE_ID,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "registry_digest": registry.registry_digest,
        "candidate_digest": model_digest(candidate),
        "policy_digest": model_digest(policy),
        "inventory_digest": hardware.inventory_digest,
        "plan_digest": plan.plan_digest,
        "environment_digest": "e" * 64,
        "tokenizer_loaded": True,
        "config_loaded": True,
        "quantized_model_loaded": True,
        "lora_attached": True,
        "forward_ok": True,
        "backward_ok": True,
        "exact_revision_observed": True,
        "trust_remote_code_used": False,
        "raw_model_outputs_stored": False,
        "probe_input_tokens": 20,
        "trainable_parameters": 20_185_088,
        "total_parameters": 4_373_157_376,
        "peak_gpu_memory_mb": 8409,
        "status": "PASSED",
        "error_code": None,
        "training_started": False,
        "training_authorized": False,
    }
    receipt = BlockchainRuntimeProbeReceipt.model_validate(
        {**receipt_payload, "receipt_digest": _digest(receipt_payload)}
    )
    audit = audit_runtime_preflight(registry, hardware, policy, plan, receipt)
    summary = {
        "candidate_id": CANDIDATE_ID,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "runtime_probe_status": "PASSED",
        "ready_for_training_authorization_review": True,
        "training_started": False,
        "training_authorized": False,
        "production_authority": False,
        "raw_model_outputs_stored": False,
    }

    _write_model(run / "hardware-inventory.json", hardware)
    _write_model(run / "preflight-plan.json", plan)
    _write_model(run / "runtime-probe-receipt.json", receipt)
    _write_model(run / "preflight-audit.json", audit)
    (run / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_digests = {
        name: _hash_file(run / name)
        for name in (
            "hardware-inventory.json",
            "preflight-plan.json",
            "runtime-probe-receipt.json",
            "preflight-audit.json",
            "summary.json",
        )
    }
    seal_payload = {
        "schema_version": "sentinel.blockchain-runtime-preflight-seal.v1",
        "candidate_id": CANDIDATE_ID,
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "github_base_commit": "a" * 40,
        "normalized_source_tree": "b" * 40,
        "source_patch_sha256": "c" * 64,
        "gpu_model": "Tesla T4",
        "available_single_gpu_vram_mb": 14912,
        "peak_gpu_memory_mb": 8409,
        "headroom_mb": 6503,
        "runtime_probe_status": "PASSED",
        "ready_for_training_authorization_review": True,
        "training_started": False,
        "training_authorized": False,
        "production_authority": False,
        "raw_model_outputs_stored": False,
        "persistent_run": run.name,
        "artifact_digests": artifact_digests,
    }
    seal = BlockchainRuntimePreflightSeal.model_validate(
        {**seal_payload, "seal_digest": _digest(seal_payload)}
    )
    seal_path = root / "preflight-seal.json"
    _write_model(seal_path, seal)
    return run, seal_path


def _materialize_bundle(root: Path):
    candidate = _candidate()
    registry = build_base_candidate_registry("authorization-test", [candidate])
    _write_model(root / "registry.json", registry)
    manifest = _materialize_release(root)
    run, seal_path = _materialize_preflight(root, registry)
    config = BlockchainTrainingConfig(
        run_id="authorization-test",
        base_model=MODEL_ID,
        base_revision=REVISION,
        blockchain_release="release",
        expected_source_corpus_digest=manifest.source_corpus_digest,
        expected_benchmark_suite_digest=manifest.benchmark_suite_digest,
        output_dir="training-output",
        max_sequence_length=512,
    )
    policy = BlockchainTrainingAuthorizationPolicy(
        policy_id="authorization-test",
        required_candidate_id=CANDIDATE_ID,
        required_model_id=MODEL_ID,
        required_model_revision=REVISION,
    )
    key = Ed25519PrivateKey.generate()
    return run, seal_path, config, policy, key


def _proposal(root: Path):
    run, seal_path, config, policy, key = _materialize_bundle(root)
    proposal = build_training_authorization_proposal(
        authorization_id="authorization-test",
        seal_path=seal_path.relative_to(root),
        preflight_run_dir=run.relative_to(root),
        registry_path="registry.json",
        release_dir="release",
        training_config=config,
        owner_public_key=key.public_key(),
        policy=policy,
        root=root,
    )
    return run, seal_path, config, policy, key, proposal


def test_owner_signed_authorization_round_trip(tmp_path: Path) -> None:
    run, seal_path, config, policy, key, proposal = _proposal(tmp_path)
    approval = approve_training_authorization_proposal(
        proposal,
        key,
        policy,
        approver_id="owner",
    )

    verified = verify_training_authorization_bundle(
        proposal=proposal,
        approval=approval,
        owner_public_key=key.public_key(),
        policy=policy,
        seal_path=seal_path.relative_to(tmp_path),
        preflight_run_dir=run.relative_to(tmp_path),
        registry_path="registry.json",
        release_dir="release",
        training_config=config,
        root=tmp_path,
    )

    assert verified.training_authorized is True
    assert verified.training_started is False
    assert verified.production_authority is False
    assert proposal.training_authorized is False
    assert proposal.preflight_receipt_digest
    assert proposal.held_out_test_split_digest


def test_tampered_preflight_artifact_blocks_authorization(tmp_path: Path) -> None:
    run, seal_path, config, policy, key = _materialize_bundle(tmp_path)
    (run / "summary.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(BlockchainTrainingAuthorizationBlocked, match="artifact digest mismatch"):
        build_training_authorization_proposal(
            authorization_id="authorization-test",
            seal_path=seal_path.relative_to(tmp_path),
            preflight_run_dir=run.relative_to(tmp_path),
            registry_path="registry.json",
            release_dir="release",
            training_config=config,
            owner_public_key=key.public_key(),
            policy=policy,
            root=tmp_path,
        )


def test_release_drift_blocks_authorization(tmp_path: Path) -> None:
    run, seal_path, config, policy, key = _materialize_bundle(tmp_path)
    train_path = tmp_path / "release" / "train.jsonl"
    train_path.write_text(train_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="release digest mismatch"):
        build_training_authorization_proposal(
            authorization_id="authorization-test",
            seal_path=seal_path.relative_to(tmp_path),
            preflight_run_dir=run.relative_to(tmp_path),
            registry_path="registry.json",
            release_dir="release",
            training_config=config,
            owner_public_key=key.public_key(),
            policy=policy,
            root=tmp_path,
        )


def test_wrong_owner_private_key_cannot_approve(tmp_path: Path) -> None:
    _, _, _, policy, _, proposal = _proposal(tmp_path)

    with pytest.raises(BlockchainTrainingAuthorizationBlocked, match="private key"):
        approve_training_authorization_proposal(
            proposal,
            Ed25519PrivateKey.generate(),
            policy,
            approver_id="wrong-owner",
        )


def test_policy_cannot_silently_switch_candidate(tmp_path: Path) -> None:
    run, seal_path, config, policy, key = _materialize_bundle(tmp_path)
    wrong = policy.model_copy(update={"required_candidate_id": "different-candidate"})

    with pytest.raises(BlockchainTrainingAuthorizationBlocked, match="candidate"):
        build_training_authorization_proposal(
            authorization_id="authorization-test",
            seal_path=seal_path.relative_to(tmp_path),
            preflight_run_dir=run.relative_to(tmp_path),
            registry_path="registry.json",
            release_dir="release",
            training_config=config,
            owner_public_key=key.public_key(),
            policy=wrong,
            root=tmp_path,
        )
