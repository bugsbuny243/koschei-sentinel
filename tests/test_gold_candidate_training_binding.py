import hashlib
import json

from koschei_sentinel.cyber_sft_export_verify import verify_cyber_sft_export
from koschei_sentinel.cyber_sft_run_attestation import _attestation_digest
from koschei_sentinel.cyber_sft_training_source import build_training_source_binding
from koschei_sentinel.defense_reflex_gold_release import write_gold_defense_release
from koschei_sentinel.gold_candidate_training_binding import (
    verify_gold_candidate_training_binding,
)
from tests.gold_candidate_binding_helpers import build_gold_bound_candidate_export
from tests.test_cyber_sft_export_verify import _build_export, _write_json
from tests.test_defense_reflex_gold_release import _release_rows


def _release(tmp_path):
    _policy, rows = _release_rows()
    release = tmp_path / "gold-release"
    write_gold_defense_release(rows, release)
    return release


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gold_candidate_binding_accepts_exact_train_and_validation_splits(
    tmp_path,
    monkeypatch,
) -> None:
    release = _release(tmp_path)
    candidate = build_gold_bound_candidate_export(tmp_path, monkeypatch, release)

    report = verify_gold_candidate_training_binding(release, candidate)

    assert report.valid is True
    assert report.stage_verified is True
    assert report.promotion_eligible_verified is True
    assert report.explicit_validation_verified is True
    assert report.exported_training_bytes_verified is True
    assert report.training_hashes_verified is True
    assert report.validation_hashes_verified is True
    assert report.gold_train_examples_sha256 == _sha(release / "train" / "examples.jsonl")
    assert report.gold_validation_examples_sha256 == _sha(
        release / "validation" / "examples.jsonl"
    )
    assert report.violations == []


def test_gold_candidate_binding_rejects_generic_promotion_candidate(
    tmp_path,
    monkeypatch,
) -> None:
    release = _release(tmp_path)
    generic_root = tmp_path / "generic-candidate"
    generic_root.mkdir()
    candidate = _build_export(
        generic_root,
        monkeypatch,
        promotion_eligible=True,
    )

    assert verify_cyber_sft_export(candidate).valid is True
    report = verify_gold_candidate_training_binding(release, candidate)

    assert report.valid is False
    assert report.explicit_validation_verified is False
    assert report.training_hashes_verified is False
    assert report.validation_hashes_verified is False


def test_gold_candidate_binding_rejects_internally_rehashed_validation_drift(
    tmp_path,
    monkeypatch,
) -> None:
    release = _release(tmp_path)
    candidate = build_gold_bound_candidate_export(tmp_path, monkeypatch, release)

    plan_path = candidate / "training-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["validation_corpus_examples_sha256"] = "9" * 64
    plan_raw = _write_json(plan_path, plan)

    source = build_training_source_binding(
        config_path=candidate / "training-config.json",
        plan_path=plan_path,
        repository_commit="f" * 40,
    )
    source_raw = _write_json(
        candidate / "training-source.json",
        source.model_dump(mode="json"),
    )

    attestation_path = candidate / "run-attestation.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["plan_sha256"] = hashlib.sha256(plan_raw).hexdigest()
    attestation["training_source_sha256"] = hashlib.sha256(source_raw).hexdigest()
    attestation["attestation_sha256"] = _attestation_digest(attestation)
    _write_json(attestation_path, attestation)

    assert verify_cyber_sft_export(candidate).valid is True
    report = verify_gold_candidate_training_binding(release, candidate)

    assert report.valid is False
    assert report.training_hashes_verified is True
    assert report.validation_hashes_verified is False
    assert any("Gold VALIDATION split" in row for row in report.violations)
