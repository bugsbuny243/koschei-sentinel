import json

from koschei_sentinel.defense_reflex_gold_release import write_gold_defense_release
from koschei_sentinel.defense_reflex_gold_release_audit import (
    audit_gold_defense_release,
)
from tests.test_defense_reflex_gold_release import _release_rows


def _release(tmp_path):
    _policy, rows = _release_rows()
    output = tmp_path / "gold-release"
    write_gold_defense_release(rows, output)
    return output


def test_gold_release_audit_accepts_intact_split_safe_release(tmp_path) -> None:
    output = _release(tmp_path)

    first = audit_gold_defense_release(output)
    second = audit_gold_defense_release(output)

    assert first == second
    assert first.valid is True
    assert first.hashes_verified is True
    assert first.human_review_only_verified is True
    assert first.holdout_isolation_verified is True
    assert first.release_digest_verified is True
    assert first.train_examples == 1
    assert first.validation_examples == 1
    assert first.holdout_cases == 1
    assert first.train_validation_overlap == []
    assert first.train_holdout_overlap == []
    assert first.validation_holdout_overlap == []
    assert first.violations == []


def test_gold_release_audit_rejects_training_example_tamper(tmp_path) -> None:
    output = _release(tmp_path)
    examples_path = output / "train" / "examples.jsonl"
    payload = examples_path.read_text(encoding="utf-8")
    examples_path.write_text(payload.replace("GOLD", "CORRECTION", 1), encoding="utf-8")

    report = audit_gold_defense_release(output)

    assert report.valid is False
    assert report.hashes_verified is False
    assert any("TRAIN" in row for row in report.violations)


def test_gold_release_audit_rejects_holdout_training_file(tmp_path) -> None:
    output = _release(tmp_path)
    (output / "holdout" / "examples.jsonl").write_text("{}\n", encoding="utf-8")

    report = audit_gold_defense_release(output)

    assert report.valid is False
    assert report.holdout_isolation_verified is False
    assert "HOLDOUT directory contains forbidden examples.jsonl" in report.violations


def test_gold_release_audit_rejects_holdout_case_self_hash_tamper(tmp_path) -> None:
    output = _release(tmp_path)
    cases_path = output / "holdout" / "cases.jsonl"
    row = json.loads(cases_path.read_text(encoding="utf-8").splitlines()[0])
    row["expected_interpretation"] += " tampered"
    cases_path.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    report = audit_gold_defense_release(output)

    assert report.valid is False
    assert any("self-hash does not verify" in row for row in report.violations)
