from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from koschei_sentinel.autotrain import AutotrainPlan
from koschei_sentinel.offline_job import build_offline_training_job
from koschei_sentinel.offline_receipt import (
    OfflineReceiptBlocked,
    OfflineTrainingReceipt,
    build_offline_training_receipt,
)
from koschei_sentinel.training import (
    AdapterManifest,
    TrainingConfig,
    TrainingPlan,
    TrainingSplit,
    model_digest,
)

_RUN_ID = "sentinel-receipt-v1"


def _config() -> TrainingConfig:
    return TrainingConfig(
        run_id=_RUN_ID,
        base_model="koschei-fixture/model",
        base_revision="1" * 40,
        dataset_release="build/releases/sentinel-receipt-v1",
        readiness_report="build/releases/sentinel-receipt-v1/readiness-report.json",
        require_readiness=True,
        output_dir="build/training/sentinel-receipt-v1",
    )


def _plan(config: TrainingConfig) -> TrainingPlan:
    split = TrainingSplit(
        path="build/releases/sentinel-receipt-v1/train.jsonl",
        examples=100,
        groups=100,
        digest="2" * 64,
    )
    return TrainingPlan(
        run_id=config.run_id,
        base_model=config.base_model,
        base_revision=config.base_revision,
        dataset_manifest_digest="3" * 64,
        readiness_report_digest="4" * 64,
        training_config_digest=model_digest(config),
        splits={"train": split, "validation": split, "test": split},
        effective_batch_size=16,
        estimated_optimizer_steps=7,
        output_dir=config.output_dir,
        warnings=[],
    )


def _job(config: TrainingConfig, plan: TrainingPlan):
    autotrain = AutotrainPlan(
        decision="ready_for_offline_training",
        training_run_id=plan.run_id,
        policy_digest="5" * 64,
        training_plan_digest=model_digest(plan),
        dataset_manifest_digest=plan.dataset_manifest_digest,
        readiness_report_digest=plan.readiness_report_digest,
        reasons=[],
        warnings=[],
    )
    return build_offline_training_job(
        autotrain,
        config,
        plan,
        config_path="configs/training/receipt.json",
        training_plan_path="build/jobs/receipt.training-plan.json",
    )


def _adapter(tmp_path: Path, config: TrainingConfig, plan: TrainingPlan) -> AdapterManifest:
    output = tmp_path / config.output_dir
    adapter_file = output / "adapter/adapter_model.safetensors"
    adapter_file.parent.mkdir(parents=True)
    adapter_file.write_bytes(b"koschei-adapter-fixture")
    relative = "adapter/adapter_model.safetensors"
    digest = hashlib.sha256()
    digest.update(relative.encode())
    digest.update(b"\0")
    digest.update(adapter_file.read_bytes())
    digest.update(b"\0")
    return AdapterManifest(
        run_id=config.run_id,
        base_model=config.base_model,
        base_revision=config.base_revision,
        dataset_manifest_digest=plan.dataset_manifest_digest,
        training_config_digest=plan.training_config_digest,
        adapter_digest=digest.hexdigest(),
        adapter_files=[relative],
        output_dir=config.output_dir,
    )


def test_receipt_verifies_real_adapter_files_and_remains_non_production(
    tmp_path: Path,
) -> None:
    config = _config()
    plan = _plan(config)
    receipt = build_offline_training_receipt(
        _job(config, plan),
        _adapter(tmp_path, config, plan),
        root=tmp_path,
    )

    assert receipt.state == "completed_offline"
    assert receipt.adapter_files_verified is True
    assert receipt.benchmark_required is True
    assert receipt.benchmark_passed is False
    assert receipt.automatic_registration_allowed is False
    assert receipt.automatic_promotion_allowed is False
    assert receipt.production_deployment_allowed is False
    assert len(receipt.receipt_digest) == 64


def test_adapter_file_tampering_blocks_receipt(tmp_path: Path) -> None:
    config = _config()
    plan = _plan(config)
    adapter = _adapter(tmp_path, config, plan)
    (tmp_path / config.output_dir / adapter.adapter_files[0]).write_bytes(b"tampered")

    with pytest.raises(OfflineReceiptBlocked, match="do not match"):
        build_offline_training_receipt(
            _job(config, plan),
            adapter,
            root=tmp_path,
        )


def test_lineage_mismatch_blocks_receipt(tmp_path: Path) -> None:
    config = _config()
    plan = _plan(config)
    adapter = _adapter(tmp_path, config, plan).model_copy(
        update={"dataset_manifest_digest": "0" * 64}
    )

    with pytest.raises(OfflineReceiptBlocked, match="dataset digest"):
        build_offline_training_receipt(
            _job(config, plan),
            adapter,
            root=tmp_path,
        )


def test_receipt_schema_cannot_claim_benchmark_or_deployment(tmp_path: Path) -> None:
    config = _config()
    plan = _plan(config)
    receipt = build_offline_training_receipt(
        _job(config, plan),
        _adapter(tmp_path, config, plan),
        root=tmp_path,
    )
    for field in (
        "benchmark_passed",
        "automatic_registration_allowed",
        "automatic_promotion_allowed",
        "production_deployment_allowed",
    ):
        payload = receipt.model_dump(mode="json")
        payload[field] = True
        with pytest.raises(ValidationError):
            OfflineTrainingReceipt.model_validate(payload)
