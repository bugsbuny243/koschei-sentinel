from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import atomic_write, canonical_json
from koschei_sentinel.web4_benchmark_intake import (
    Web4BenchmarkIntakePacket,
    Web4BenchmarkSplit,
    build_web4_benchmark_intake,
)

_DIGEST = r"^[a-f0-9]{64}$"


def _digest(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_file(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Web4BenchmarkBatchItem(StrictModel):
    proposal: str = Field(min_length=1, max_length=4096)
    answer_key: str = Field(min_length=1, max_length=4096)


class Web4BenchmarkBatchManifest(StrictModel):
    schema_version: Literal["sentinel.web4-benchmark-batch-manifest.v1"] = (
        "sentinel.web4-benchmark-batch-manifest.v1"
    )
    batch_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    items: list[Web4BenchmarkBatchItem] = Field(min_length=1, max_length=100000)

    @model_validator(mode="after")
    def item_paths_are_unique(self) -> Web4BenchmarkBatchManifest:
        proposals = [item.proposal for item in self.items]
        answers = [item.answer_key for item in self.items]
        if len(proposals) != len(set(proposals)):
            raise ValueError("Web4 batch manifest contains duplicate proposal paths")
        if len(answers) != len(set(answers)):
            raise ValueError("Web4 batch manifest contains duplicate answer-key paths")
        return self


class Web4BenchmarkBatchQueueManifest(StrictModel):
    schema_version: Literal["sentinel.web4-benchmark-batch-queue.v1"] = (
        "sentinel.web4-benchmark-batch-queue.v1"
    )
    batch_id: str
    input_manifest_sha256: str = Field(pattern=_DIGEST)
    source_registry_sha256: str = Field(pattern=_DIGEST)
    benchmark_policy_sha256: str = Field(pattern=_DIGEST)
    intake_policy_sha256: str = Field(pattern=_DIGEST)
    packet_count: int = Field(gt=0)
    development_packets: int = Field(ge=0)
    validation_packets: int = Field(ge=0)
    holdout_packets: int = Field(ge=0)
    family_counts: dict[str, int]
    holdout_family_counts: dict[str, int]
    packet_ids_sha256: str = Field(pattern=_DIGEST)
    development_sha256: str = Field(pattern=_DIGEST)
    validation_sha256: str = Field(pattern=_DIGEST)
    holdout_sha256: str = Field(pattern=_DIGEST)
    answer_key_values_embedded: Literal[False] = False
    split_before_human_review: Literal[True] = True
    manual_split_override_allowed: Literal[False] = False
    human_review_required: Literal[True] = True
    all_packets_human_reviewed: Literal[False] = False
    training_authorization: Literal[False] = False
    evaluation_authorization: Literal[False] = False
    promotion_eligible: Literal[False] = False
    production_activation_allowed: Literal[False] = False
    queue_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def queue_contract_verifies(self) -> Web4BenchmarkBatchQueueManifest:
        if (
            self.development_packets + self.validation_packets + self.holdout_packets
            != self.packet_count
        ):
            raise ValueError("Web4 batch split counts do not sum to packet count")
        if sum(self.family_counts.values()) != self.packet_count:
            raise ValueError("Web4 batch family counts do not sum to packet count")
        if sum(self.holdout_family_counts.values()) != self.holdout_packets:
            raise ValueError("Web4 batch HOLDOUT family counts do not sum to HOLDOUT count")
        if any(value < 0 for value in self.family_counts.values()):
            raise ValueError("Web4 batch family counts must not be negative")
        if any(value < 0 for value in self.holdout_family_counts.values()):
            raise ValueError("Web4 batch HOLDOUT family counts must not be negative")
        payload = self.model_dump(mode="json")
        observed = str(payload.pop("queue_sha256"))
        if _digest(payload) != observed:
            raise ValueError("Web4 batch queue self-hash does not verify")
        return self


def _regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return candidate


def _relative_file(root: Path, value: str, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be relative to the batch manifest")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} must not traverse symlinks")
    root_resolved = root.resolve()
    candidate = (root / relative).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"{label} escapes the batch manifest directory")
    return _regular_file(candidate, label)


def _load_batch_manifest(path: str | Path) -> tuple[Web4BenchmarkBatchManifest, str, Path]:
    manifest_path = _regular_file(path, "Web4 batch manifest")
    raw = manifest_path.read_bytes()
    try:
        manifest = Web4BenchmarkBatchManifest.model_validate_json(raw)
    except ValueError as exc:
        raise ValueError("invalid Web4 batch manifest") from exc
    return manifest, hashlib.sha256(raw).hexdigest(), manifest_path.resolve().parent


def _load_policy_contract(path: str | Path) -> dict[str, object]:
    policy_path = _regular_file(path, "Web4 batch intake policy")
    try:
        payload = json.loads(policy_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid Web4 batch intake policy JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Web4 batch intake policy must contain one JSON object")
    if payload.get("schema_version") != "sentinel.web4-benchmark-intake-policy.v1":
        raise ValueError("unsupported Web4 batch intake policy schema")
    if payload.get("split_before_human_review") is not True:
        raise ValueError("Web4 batch intake must split before human review")
    if payload.get("manual_split_override_allowed") is not False:
        raise ValueError("Web4 batch intake manual split override must remain disabled")
    if payload.get("training_authorization") is not False:
        raise ValueError("Web4 batch intake training authorization must remain disabled")
    if payload.get("evaluation_authorization_before_human_review") is not False:
        raise ValueError("Web4 batch intake evaluation must remain closed before review")
    return payload


def _serialize(rows: list[Web4BenchmarkIntakePacket]) -> str:
    return "".join(
        json.dumps(row.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in sorted(rows, key=lambda item: item.case_id)
    )


def build_web4_benchmark_batch_queue(
    *,
    manifest_path: str | Path,
    source_registry_path: str | Path,
    benchmark_policy_path: str | Path,
    intake_policy_path: str | Path,
    snapshot_root: str | Path,
    output_dir: str | Path,
) -> Web4BenchmarkBatchQueueManifest:
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Web4 batch queue output already exists: {destination}")
    parent = destination.parent
    if parent.is_symlink():
        raise ValueError("Web4 batch queue output parent must not be a symlink")
    parent.mkdir(parents=True, exist_ok=True)

    manifest, manifest_sha, manifest_root = _load_batch_manifest(manifest_path)
    intake_policy = _load_policy_contract(intake_policy_path)
    source_registry = _regular_file(source_registry_path, "Web4 source registry")
    benchmark_policy = _regular_file(benchmark_policy_path, "Web4 benchmark policy")
    intake_policy_file = _regular_file(intake_policy_path, "Web4 intake policy")
    snapshot = Path(snapshot_root)
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ValueError("Web4 batch snapshot root must be a regular non-symlink directory")

    packets: list[Web4BenchmarkIntakePacket] = []
    for index, item in enumerate(manifest.items):
        proposal = _relative_file(
            manifest_root,
            item.proposal,
            f"Web4 batch item[{index}] proposal",
        )
        answer_key = _relative_file(
            manifest_root,
            item.answer_key,
            f"Web4 batch item[{index}] answer key",
        )
        packets.append(
            build_web4_benchmark_intake(
                proposal_path=proposal,
                answer_key_path=answer_key,
                source_registry_path=source_registry,
                benchmark_policy_path=benchmark_policy,
                intake_policy_path=intake_policy_file,
                snapshot_root=snapshot,
            )
        )

    case_ids = [packet.case_id for packet in packets]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Web4 batch intake produced duplicate case IDs")
    packet_shas = [packet.packet_sha256 for packet in packets]
    if len(packet_shas) != len(set(packet_shas)):
        raise ValueError("Web4 batch intake produced duplicate packet identities")
    packets.sort(key=lambda row: row.case_id)

    by_split = {
        split: [packet for packet in packets if packet.split is split]
        for split in Web4BenchmarkSplit
    }
    serialized = {split: _serialize(rows) for split, rows in by_split.items()}
    family_counts: dict[str, int] = {}
    holdout_family_counts: dict[str, int] = {}
    for packet in packets:
        family_counts[packet.family] = family_counts.get(packet.family, 0) + 1
        if packet.split is Web4BenchmarkSplit.HOLDOUT:
            holdout_family_counts[packet.family] = holdout_family_counts.get(packet.family, 0) + 1
    family_counts = dict(sorted(family_counts.items()))
    holdout_family_counts = dict(sorted(holdout_family_counts.items()))

    payload: dict[str, object] = {
        "schema_version": "sentinel.web4-benchmark-batch-queue.v1",
        "batch_id": manifest.batch_id,
        "input_manifest_sha256": manifest_sha,
        "source_registry_sha256": _sha256_file(source_registry, "Web4 source registry"),
        "benchmark_policy_sha256": _sha256_file(benchmark_policy, "Web4 benchmark policy"),
        "intake_policy_sha256": _sha256_file(intake_policy_file, "Web4 intake policy"),
        "packet_count": len(packets),
        "development_packets": len(by_split[Web4BenchmarkSplit.DEVELOPMENT]),
        "validation_packets": len(by_split[Web4BenchmarkSplit.VALIDATION]),
        "holdout_packets": len(by_split[Web4BenchmarkSplit.HOLDOUT]),
        "family_counts": family_counts,
        "holdout_family_counts": holdout_family_counts,
        "packet_ids_sha256": _digest(sorted(packet_shas)),
        "development_sha256": hashlib.sha256(
            serialized[Web4BenchmarkSplit.DEVELOPMENT].encode("utf-8")
        ).hexdigest(),
        "validation_sha256": hashlib.sha256(
            serialized[Web4BenchmarkSplit.VALIDATION].encode("utf-8")
        ).hexdigest(),
        "holdout_sha256": hashlib.sha256(
            serialized[Web4BenchmarkSplit.HOLDOUT].encode("utf-8")
        ).hexdigest(),
        "answer_key_values_embedded": False,
        "split_before_human_review": intake_policy["split_before_human_review"],
        "manual_split_override_allowed": intake_policy["manual_split_override_allowed"],
        "human_review_required": True,
        "all_packets_human_reviewed": False,
        "training_authorization": False,
        "evaluation_authorization": False,
        "promotion_eligible": False,
        "production_activation_allowed": False,
    }
    payload["queue_sha256"] = _digest(payload)
    queue_manifest = Web4BenchmarkBatchQueueManifest.model_validate(payload)

    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
    try:
        atomic_write(staging / "development.jsonl", serialized[Web4BenchmarkSplit.DEVELOPMENT])
        atomic_write(staging / "validation.jsonl", serialized[Web4BenchmarkSplit.VALIDATION])
        atomic_write(staging / "holdout.jsonl", serialized[Web4BenchmarkSplit.HOLDOUT])
        atomic_write(
            staging / "manifest.json",
            json.dumps(queue_manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Web4 batch queue output appeared during build: {destination}")
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return queue_manifest
