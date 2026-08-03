from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from koschei_sentinel.training import (
    TrainingConfig,
    load_release_examples,
    plan_training,
    supervised_messages,
)

_ROW = {
    "schema_version": "sentinel.dataset.v1",
    "example_id": "example_" + "a" * 24,
    "group_ref": "group_" + "b" * 24,
    "source_digest": "c" * 64,
    "case": {
        "schema_version": "sentinel.case.v1",
        "case_id": "case_" + "d" * 24,
        "target_ref": "target_" + "e" * 24,
        "network": "solana-mainnet",
        "signed_verdict": {
            "grade": "D",
            "signature": "signature_" + "f" * 24,
            "triggered_rules": ["KS-HOLDER-001"],
            "summary": "A deterministic holder concentration condition was found.",
        },
        "evidence": [
            {
                "evidence_id": "evidence_" + "1" * 24,
                "kind": "holder_intelligence",
                "statement": "The largest holder controls a material supply share.",
                "confidence": "VERIFIED",
                "rule_ids": ["KS-HOLDER-001"],
                "attributes": {"share_bps": 4200},
            }
        ],
        "limitations": ["No live market data was supplied."],
    },
}


def _config() -> TrainingConfig:
    return TrainingConfig(
        run_id="fixture-v0.6",
        base_model="sentinel-fixture/model",
        base_revision="0" * 40,
        dataset_release="fixtures/training/release",
        output_dir="build/training/fixture-v0.6",
    )


def _write_release(root: Path) -> None:
    release = root / "fixtures/training/release"
    release.mkdir(parents=True)
    payloads = {
        "train": json.dumps(_ROW, sort_keys=True, separators=(",", ":")) + "\n",
        "validation": "",
        "test": "",
    }
    reports = {}
    for split_name, payload in payloads.items():
        (release / f"{split_name}.jsonl").write_text(payload, encoding="utf-8")
        populated = split_name == "train"
        reports[split_name] = {
            "examples": 1 if populated else 0,
            "groups": 1 if populated else 0,
            "digest": hashlib.sha256(payload.encode()).hexdigest(),
        }
    manifest = {
        "schema_version": "sentinel.quality-manifest.v1",
        "dry_run": False,
        "seed": "fixture",
        "train_bps": 8000,
        "validation_bps": 1000,
        "test_bps": 1000,
        "total_examples": 1,
        "total_groups": 1,
        "splits": reports,
        "grade_counts": {"D": 1},
        "confidence_counts": {"VERIFIED": 1},
        "evidence_kind_counts": {"holder_intelligence": 1},
        "warnings": ["validation split is empty", "test split is empty"],
    }
    (release / "quality-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_training_plan_validates_release_and_estimates_steps(tmp_path: Path) -> None:
    _write_release(tmp_path)

    plan = plan_training(_config(), root=tmp_path)

    assert plan.splits["train"].examples == 1
    assert plan.estimated_optimizer_steps == 1
    assert plan.warnings == ["validation split is empty", "test split is empty"]


def test_training_plan_rejects_release_digest_drift(tmp_path: Path) -> None:
    _write_release(tmp_path)
    train_path = tmp_path / "fixtures/training/release/train.jsonl"
    train_path.write_text(train_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        plan_training(_config(), root=tmp_path)


def test_training_config_rejects_path_escape_and_mutable_revision() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(
            run_id="bad-path",
            base_model="owner/model",
            base_revision="0" * 40,
            dataset_release="../private",
            output_dir="build/training/bad",
        )
    with pytest.raises(ValidationError):
        TrainingConfig(
            run_id="bad-revision",
            base_model="owner/model",
            base_revision="main",
            dataset_release="build/releases/safe",
            output_dir="build/training/bad",
        )


def test_supervision_is_policy_grounded_and_timestamp_free(tmp_path: Path) -> None:
    _write_release(tmp_path)
    rows, _ = load_release_examples(
        tmp_path.resolve(), "fixtures/training/release/train.jsonl"
    )

    messages = supervised_messages(rows[0])
    answer = json.loads(messages[-1]["content"])

    assert [item["role"] for item in messages] == ["system", "user", "assistant"]
    assert answer["assessment"] == "EXPLANATION_ONLY"
    assert answer["claims"][0]["evidence_ids"] == ["evidence_" + "1" * 24]
    assert "generated_at" not in answer
