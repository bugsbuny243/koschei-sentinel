from __future__ import annotations

import json
from pathlib import Path

import pytest

from koschei_sentinel.web4_benchmark_batch_queue import (
    Web4BenchmarkBatchQueueManifest,
    build_web4_benchmark_batch_queue,
)
from koschei_sentinel.web4_benchmark_batch_queue_cli import main as batch_main
from koschei_sentinel.web4_benchmark_intake import (
    Web4BenchmarkSplit,
    build_web4_benchmark_intake,
)

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _ROOT / "fixtures/web4/benchmark-intake"
_PROPOSAL = _FIXTURE / "proposal.json"
_ANSWER_KEY = _FIXTURE / "answer-key.json"
_SOURCES = _ROOT / "configs/corpus/web4-v1.sources.proposed.jsonl"
_BENCHMARK = _ROOT / "evals/web4-security-benchmark.v1.json"
_INTAKE_POLICY = _ROOT / "evals/web4-benchmark-intake-policy.v1.json"
_SECRET_ANSWER = "REJECT_UNAUTHORIZED_WIDENING"


def _write_case(root: Path, case_id: str, suffix: str) -> tuple[Path, Path]:
    case_dir = root / suffix
    case_dir.mkdir(parents=True, exist_ok=False)
    proposal = json.loads(_PROPOSAL.read_text(encoding="utf-8"))
    answer = json.loads(_ANSWER_KEY.read_text(encoding="utf-8"))
    proposal["case_id"] = case_id
    answer["case_id"] = case_id
    proposal_path = case_dir / "proposal.json"
    answer_path = case_dir / "answer-key.json"
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
    answer_path.write_text(json.dumps(answer), encoding="utf-8")
    return proposal_path, answer_path


def _find_one_case_per_split(root: Path) -> dict[Web4BenchmarkSplit, tuple[Path, Path]]:
    selected: dict[Web4BenchmarkSplit, tuple[Path, Path]] = {}
    for index in range(1, 500):
        case_id = f"batch-split-{index:04d}"
        proposal, answer = _write_case(root, case_id, f"candidate-{index:04d}")
        packet = build_web4_benchmark_intake(
            proposal_path=proposal,
            answer_key_path=answer,
            source_registry_path=_SOURCES,
            benchmark_policy_path=_BENCHMARK,
            intake_policy_path=_INTAKE_POLICY,
            snapshot_root=_FIXTURE,
        )
        selected.setdefault(packet.split, (proposal, answer))
        if len(selected) == 3:
            return selected
    raise AssertionError("could not find deterministic DEVELOPMENT/VALIDATION/HOLDOUT fixtures")


def _write_manifest(
    path: Path,
    root: Path,
    rows: list[tuple[Path, Path]],
    *,
    batch_id: str = "web4-batch-fixture-v1",
) -> None:
    payload = {
        "schema_version": "sentinel.web4-benchmark-batch-manifest.v1",
        "batch_id": batch_id,
        "items": [
            {
                "proposal": str(proposal.relative_to(root)),
                "answer_key": str(answer.relative_to(root)),
            }
            for proposal, answer in rows
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_batch_queue_builds_all_splits_without_answer_key_leakage(tmp_path: Path) -> None:
    materials = tmp_path / "materials"
    materials.mkdir()
    selected = _find_one_case_per_split(materials)
    manifest_path = materials / "batch.json"
    _write_manifest(manifest_path, materials, list(selected.values()))
    output = tmp_path / "queue"

    manifest = build_web4_benchmark_batch_queue(
        manifest_path=manifest_path,
        source_registry_path=_SOURCES,
        benchmark_policy_path=_BENCHMARK,
        intake_policy_path=_INTAKE_POLICY,
        snapshot_root=_FIXTURE,
        output_dir=output,
    )

    assert manifest.packet_count == 3
    assert manifest.development_packets == 1
    assert manifest.validation_packets == 1
    assert manifest.holdout_packets == 1
    assert manifest.answer_key_values_embedded is False
    assert manifest.split_before_human_review is True
    assert manifest.manual_split_override_allowed is False
    assert manifest.human_review_required is True
    assert manifest.all_packets_human_reviewed is False
    assert manifest.training_authorization is False
    assert manifest.evaluation_authorization is False
    assert manifest.promotion_eligible is False
    assert manifest.production_activation_allowed is False

    expected_files = {
        "development.jsonl",
        "validation.jsonl",
        "holdout.jsonl",
        "manifest.json",
    }
    assert {item.name for item in output.iterdir()} == expected_files
    for filename, expected_split in (
        ("development.jsonl", "DEVELOPMENT"),
        ("validation.jsonl", "VALIDATION"),
        ("holdout.jsonl", "HOLDOUT"),
    ):
        rows = _read_jsonl(output / filename)
        assert len(rows) == 1
        assert rows[0]["split"] == expected_split
        assert rows[0]["contains_answer_key"] is False
        assert rows[0]["training_authorization"] is False
        assert rows[0]["evaluation_authorization"] is False

    serialized_queue = "".join(
        (output / filename).read_text(encoding="utf-8")
        for filename in sorted(expected_files)
    )
    assert _SECRET_ANSWER not in serialized_queue
    reparsed = Web4BenchmarkBatchQueueManifest.model_validate_json(
        (output / "manifest.json").read_bytes()
    )
    assert reparsed.queue_sha256 == manifest.queue_sha256


def test_batch_queue_cli_prints_only_safe_manifest(tmp_path: Path, capsys) -> None:
    materials = tmp_path / "materials"
    materials.mkdir()
    selected = _find_one_case_per_split(materials)
    manifest_path = materials / "batch.json"
    _write_manifest(manifest_path, materials, list(selected.values()))
    output = tmp_path / "queue"

    result = batch_main(
        [
            "--manifest",
            str(manifest_path),
            "--snapshot-root",
            str(_FIXTURE),
            "--sources",
            str(_SOURCES),
            "--benchmark",
            str(_BENCHMARK),
            "--intake-policy",
            str(_INTAKE_POLICY),
            "--output-dir",
            str(output),
        ]
    )
    assert result == 0
    printed = capsys.readouterr().out
    assert _SECRET_ANSWER not in printed
    assert '"human_review_required": true' in printed
    assert '"training_authorization": false' in printed


def test_batch_queue_rejects_duplicate_case_ids_without_publishing(tmp_path: Path) -> None:
    materials = tmp_path / "materials"
    materials.mkdir()
    first = _write_case(materials, "duplicate-batch-case", "first")
    second = _write_case(materials, "duplicate-batch-case", "second")
    manifest_path = materials / "batch.json"
    _write_manifest(manifest_path, materials, [first, second])
    output = tmp_path / "queue"

    with pytest.raises(ValueError, match="duplicate case IDs"):
        build_web4_benchmark_batch_queue(
            manifest_path=manifest_path,
            source_registry_path=_SOURCES,
            benchmark_policy_path=_BENCHMARK,
            intake_policy_path=_INTAKE_POLICY,
            snapshot_root=_FIXTURE,
            output_dir=output,
        )
    assert not output.exists()


def test_batch_queue_rejects_parent_traversal(tmp_path: Path) -> None:
    materials = tmp_path / "materials"
    materials.mkdir()
    outside_proposal, outside_answer = _write_case(
        tmp_path,
        "traversal-batch-case",
        "outside",
    )
    manifest = {
        "schema_version": "sentinel.web4-benchmark-batch-manifest.v1",
        "batch_id": "web4-traversal-batch-v1",
        "items": [
            {
                "proposal": f"../{outside_proposal.parent.name}/{outside_proposal.name}",
                "answer_key": f"../{outside_answer.parent.name}/{outside_answer.name}",
            }
        ],
    }
    manifest_path = materials / "batch.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "queue"

    with pytest.raises(ValueError, match="relative to the batch manifest"):
        build_web4_benchmark_batch_queue(
            manifest_path=manifest_path,
            source_registry_path=_SOURCES,
            benchmark_policy_path=_BENCHMARK,
            intake_policy_path=_INTAKE_POLICY,
            snapshot_root=_FIXTURE,
            output_dir=output,
        )
    assert not output.exists()


def test_batch_queue_preexisting_output_fails_before_input_reads(tmp_path: Path) -> None:
    output = tmp_path / "queue"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        build_web4_benchmark_batch_queue(
            manifest_path=tmp_path / "missing-manifest.json",
            source_registry_path=tmp_path / "missing-sources.jsonl",
            benchmark_policy_path=tmp_path / "missing-benchmark.json",
            intake_policy_path=tmp_path / "missing-intake.json",
            snapshot_root=tmp_path / "missing-snapshots",
            output_dir=output,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_batch_queue_manifest_self_hash_tamper_is_rejected(tmp_path: Path) -> None:
    materials = tmp_path / "materials"
    materials.mkdir()
    selected = _find_one_case_per_split(materials)
    manifest_path = materials / "batch.json"
    _write_manifest(manifest_path, materials, list(selected.values()))
    output = tmp_path / "queue"
    report = build_web4_benchmark_batch_queue(
        manifest_path=manifest_path,
        source_registry_path=_SOURCES,
        benchmark_policy_path=_BENCHMARK,
        intake_policy_path=_INTAKE_POLICY,
        snapshot_root=_FIXTURE,
        output_dir=output,
    )
    tampered = report.model_dump(mode="json")
    tampered["queue_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="self-hash"):
        Web4BenchmarkBatchQueueManifest.model_validate(tampered)
