from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from koschei_sentinel.checkpoint_lineage import (
    finalize_checkpoint,
    prepare_execution_envelope,
    verify_checkpoint_manifest,
    write_checkpoint_manifest,
)
from koschei_sentinel.continued_pretraining import (
    ContinuedPretrainingPlan,
    write_continued_pretraining_plan,
)


def plan() -> ContinuedPretrainingPlan:
    return ContinuedPretrainingPlan(
        run_id="stage2-checkpoint-test",
        base_model="Qwen/Qwen2.5-1.5B",
        base_revision="b" * 40,
        corpus_file_digest="1" * 64,
        corpus_digest="2" * 64,
        corpus_audit_file_digest="3" * 64,
        corpus_policy_digest="4" * 64,
        holdout_digest="5" * 64,
        benchmark_suite_digest="6" * 64,
        training_config_digest="7" * 64,
        documents=10_000,
        unique_families=500,
        max_sequence_length=4096,
        epochs=1.0,
        learning_rate=0.00005,
        per_device_batch_size=1,
        gradient_accumulation_steps=32,
        effective_batch_size=32,
        estimated_optimizer_steps=313,
        seed=1701,
        output_dir="checkpoint-out",
    )


def write_plan(root: Path) -> Path:
    path = root / "stage2.plan.json"
    write_continued_pretraining_plan(plan(), path)
    return path


def materialize_checkpoint(root: Path) -> Path:
    output = root / "checkpoint-out"
    output.mkdir()
    (output / "config.json").write_text('{"model":"koschei"}\n', encoding="utf-8")
    (output / "model.safetensors").write_bytes(b"stage2-weights-v1")
    return output


def test_execution_envelope_binds_exact_plan_and_hyperparameters(tmp_path: Path) -> None:
    path = write_plan(tmp_path)
    envelope = prepare_execution_envelope(path, root=tmp_path)

    assert envelope.executor_id == "koschei-stage2-executor/v1"
    assert envelope.plan_path == "stage2.plan.json"
    assert envelope.plan_file_digest
    assert envelope.plan_digest
    assert envelope.base_revision == "b" * 40
    assert envelope.corpus_digest == "2" * 64
    assert envelope.epochs == 1.0
    assert envelope.learning_rate == 0.00005
    assert envelope.effective_batch_size == 32
    assert envelope.estimated_optimizer_steps == 313
    assert envelope.seed == 1701
    assert not (tmp_path / "checkpoint-out").exists()


def test_existing_output_blocks_execution_envelope(tmp_path: Path) -> None:
    path = write_plan(tmp_path)
    (tmp_path / "checkpoint-out").mkdir()
    with pytest.raises(FileExistsError, match="output already exists"):
        prepare_execution_envelope(path, root=tmp_path)


def test_checkpoint_manifest_round_trip_binds_every_file(tmp_path: Path) -> None:
    path = write_plan(tmp_path)
    envelope = prepare_execution_envelope(path, root=tmp_path)
    materialize_checkpoint(tmp_path)

    manifest = finalize_checkpoint(envelope, root=tmp_path)
    assert [item.path for item in manifest.files] == ["config.json", "model.safetensors"]
    assert all(item.sha256 for item in manifest.files)
    assert manifest.checkpoint_digest

    manifest_path = write_checkpoint_manifest(manifest, envelope, root=tmp_path)
    verified = verify_checkpoint_manifest(manifest_path, envelope, root=tmp_path)
    assert verified == manifest


def test_checkpoint_file_tampering_breaks_verification(tmp_path: Path) -> None:
    path = write_plan(tmp_path)
    envelope = prepare_execution_envelope(path, root=tmp_path)
    output = materialize_checkpoint(tmp_path)
    manifest = finalize_checkpoint(envelope, root=tmp_path)
    manifest_path = write_checkpoint_manifest(manifest, envelope, root=tmp_path)

    (output / "model.safetensors").write_bytes(b"tampered-weights")
    with pytest.raises(ValueError, match="stale or checkpoint files were modified"):
        verify_checkpoint_manifest(manifest_path, envelope, root=tmp_path)


def test_plan_drift_after_envelope_blocks_checkpoint_finalization(tmp_path: Path) -> None:
    path = write_plan(tmp_path)
    envelope = prepare_execution_envelope(path, root=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["seed"] = 1702
    path.write_text(json.dumps(payload), encoding="utf-8")
    materialize_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="plan file digest changed"):
        finalize_checkpoint(envelope, root=tmp_path)


def test_empty_checkpoint_and_symlink_fail_closed(tmp_path: Path) -> None:
    path = write_plan(tmp_path)
    envelope = prepare_execution_envelope(path, root=tmp_path)
    output = tmp_path / "checkpoint-out"
    output.mkdir()
    with pytest.raises(ValueError, match="contains no model artifacts"):
        finalize_checkpoint(envelope, root=tmp_path)

    target = tmp_path / "outside.bin"
    target.write_bytes(b"outside")
    link = output / "weights.bin"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")
    with pytest.raises(ValueError, match="symbolic links"):
        finalize_checkpoint(envelope, root=tmp_path)


def test_checkpoint_manifest_is_no_replace(tmp_path: Path) -> None:
    path = write_plan(tmp_path)
    envelope = prepare_execution_envelope(path, root=tmp_path)
    materialize_checkpoint(tmp_path)
    manifest = finalize_checkpoint(envelope, root=tmp_path)
    write_checkpoint_manifest(manifest, envelope, root=tmp_path)

    with pytest.raises(FileExistsError, match="already exists"):
        write_checkpoint_manifest(manifest, envelope, root=tmp_path)
