from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import atomic_write, canonical_json

_DIGEST = r"^[a-f0-9]{64}$"
_FORBIDDEN_MODEL_INPUT_KEYS = {
    "answer_key",
    "correct_answer",
    "expected_answer",
    "gold_answer",
    "ground_truth",
    "reference_answer",
}


class Web4BenchmarkSplit(StrEnum):
    DEVELOPMENT = "DEVELOPMENT"
    VALIDATION = "VALIDATION"
    HOLDOUT = "HOLDOUT"


class Web4BenchmarkSourceInput(StrictModel):
    source_ref: str = Field(min_length=3, max_length=256)
    snapshot_path: str = Field(min_length=1, max_length=4096)


class Web4BenchmarkCaseProposal(StrictModel):
    schema_version: Literal["sentinel.web4-benchmark-case-proposal.v1"] = (
        "sentinel.web4-benchmark-case-proposal.v1"
    )
    case_id: str = Field(min_length=3, max_length=256)
    family: str = Field(min_length=3, max_length=256)
    created_at: str = Field(min_length=10, max_length=64)
    model_input: dict[str, object]
    sources: list[Web4BenchmarkSourceInput] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def proposal_is_safe_for_intake(self) -> Web4BenchmarkCaseProposal:
        try:
            created_at = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Web4 benchmark created_at must be ISO-8601") from exc
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("Web4 benchmark created_at must include a timezone")
        source_refs = [source.source_ref for source in self.sources]
        if len(source_refs) != len(set(source_refs)):
            raise ValueError("Web4 benchmark proposal contains duplicate source refs")
        if not self.model_input:
            raise ValueError("Web4 benchmark proposal requires model-visible input")
        _reject_answer_key_fields(self.model_input)
        return self


class Web4BenchmarkAnswerKey(StrictModel):
    schema_version: Literal["sentinel.web4-benchmark-answer-key.v1"] = (
        "sentinel.web4-benchmark-answer-key.v1"
    )
    case_id: str = Field(min_length=3, max_length=256)
    source_refs: list[str] = Field(min_length=1, max_length=16)
    expected: dict[str, object]

    @model_validator(mode="after")
    def answer_key_is_bound(self) -> Web4BenchmarkAnswerKey:
        if len(self.source_refs) != len(set(self.source_refs)):
            raise ValueError("Web4 benchmark answer key contains duplicate source refs")
        if not self.expected:
            raise ValueError("Web4 benchmark answer key requires expected output")
        return self


class Web4BenchmarkIntakePolicy(StrictModel):
    schema_version: Literal["sentinel.web4-benchmark-intake-policy.v1"] = (
        "sentinel.web4-benchmark-intake-policy.v1"
    )
    split_seed: str = Field(min_length=8, max_length=256)
    development_bps: int = Field(ge=0, le=10000)
    validation_bps: int = Field(ge=0, le=10000)
    holdout_bps: int = Field(ge=1, le=10000)
    split_before_human_review: Literal[True] = True
    manual_split_override_allowed: Literal[False] = False
    training_authorization: Literal[False] = False
    evaluation_authorization_before_human_review: Literal[False] = False
    answer_key_storage: Literal["SEPARATE_FILE_SHA256_ONLY"] = "SEPARATE_FILE_SHA256_ONLY"
    require_source_snapshot_hashes: Literal[True] = True
    source_registry_eval_exclusion_required: Literal[True] = True
    source_registry_training_closed_required: Literal[True] = True

    @model_validator(mode="after")
    def split_contract_is_complete(self) -> Web4BenchmarkIntakePolicy:
        if self.development_bps + self.validation_bps + self.holdout_bps != 10000:
            raise ValueError("Web4 benchmark intake split basis points must sum to 10000")
        return self


class Web4BenchmarkIntakePacket(StrictModel):
    schema_version: Literal["sentinel.web4-benchmark-intake-packet.v1"] = (
        "sentinel.web4-benchmark-intake-packet.v1"
    )
    case_id: str
    family: str
    created_at: str
    split: Web4BenchmarkSplit
    split_seed: str
    split_material_sha256: str = Field(pattern=_DIGEST)
    benchmark_policy_sha256: str = Field(pattern=_DIGEST)
    intake_policy_sha256: str = Field(pattern=_DIGEST)
    source_registry_sha256: str = Field(pattern=_DIGEST)
    source_refs: list[str]
    source_revision_status: dict[str, str]
    source_snapshot_sha256s: dict[str, str]
    model_input: dict[str, object]
    model_input_sha256: str = Field(pattern=_DIGEST)
    answer_key_sha256: str = Field(pattern=_DIGEST)
    review_status: Literal["UNREVIEWED"] = "UNREVIEWED"
    human_reviewed: Literal[False] = False
    contains_answer_key: Literal[False] = False
    training_authorization: Literal[False] = False
    evaluation_authorization: Literal[False] = False
    promotion_eligible: Literal[False] = False
    packet_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def packet_contract_verifies(self) -> Web4BenchmarkIntakePacket:
        if set(self.source_refs) != set(self.source_revision_status):
            raise ValueError("Web4 benchmark packet source revision bindings differ")
        if set(self.source_refs) != set(self.source_snapshot_sha256s):
            raise ValueError("Web4 benchmark packet source snapshot bindings differ")
        _reject_answer_key_fields(self.model_input)
        unsigned = self.model_dump(mode="json")
        observed = str(unsigned.pop("packet_sha256"))
        if _sha256_canonical(unsigned) != observed:
            raise ValueError("Web4 benchmark intake packet self-hash does not verify")
        return self


def _sha256_canonical(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _read_regular(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return path.read_bytes()


def _sha256_file(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> tuple[dict[str, object], str]:
    raw = _read_regular(path, label)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def _load_source_registry(path: Path) -> tuple[dict[str, dict[str, object]], str]:
    raw = _read_regular(path, "Web4 source registry")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Web4 source registry is not valid UTF-8") from exc

    by_id: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid Web4 source row at line {line_number}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Web4 source row {line_number} must be a JSON object")
        source_id = payload.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"Web4 source row {line_number} lacks source_id")
        if source_id in by_id:
            raise ValueError(f"duplicate Web4 source_id: {source_id}")
        by_id[source_id] = payload
    if not by_id:
        raise ValueError("Web4 source registry is empty")
    return by_id, hashlib.sha256(raw).hexdigest()


def _reject_answer_key_fields(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.strip().lower().replace("-", "_").replace(" ", "_")
            if normalized in _FORBIDDEN_MODEL_INPUT_KEYS:
                raise ValueError(
                    f"Web4 model-visible input contains answer-key-like field: {key}"
                )
            _reject_answer_key_fields(child)
    elif isinstance(value, list):
        for child in value:
            _reject_answer_key_fields(child)


def _validate_benchmark_policy(payload: dict[str, object]) -> set[str]:
    if payload.get("schema_version") != "sentinel.web4-eval-policy.v1":
        raise ValueError("Web4 benchmark policy schema mismatch")
    if payload.get("status") != "research_only":
        raise ValueError("Web4 benchmark policy must remain research only")
    if payload.get("promotion_eligible") is not False:
        raise ValueError("Web4 benchmark policy must remain promotion-ineligible")
    if payload.get("training_overlap_allowed") is not False:
        raise ValueError("Web4 benchmark training overlap must remain forbidden")
    if payload.get("requires_human_reviewed_holdout") is not True:
        raise ValueError("Web4 benchmark policy must require human-reviewed HOLDOUT")

    required_metadata = payload.get("required_case_metadata")
    required_fields = {
        "case_id",
        "source_refs",
        "source_revision_status",
        "created_at",
        "review_status",
        "split",
        "answer_key_sha256",
    }
    if not isinstance(required_metadata, list) or not required_fields.issubset(
        set(required_metadata)
    ):
        raise ValueError("Web4 benchmark policy case metadata contract is incomplete")

    families = payload.get("required_families")
    if not isinstance(families, list) or not families:
        raise ValueError("Web4 benchmark policy has no required families")
    if not all(isinstance(family, str) and family for family in families):
        raise ValueError("Web4 benchmark policy contains invalid family names")
    if len(families) != len(set(families)):
        raise ValueError("Web4 benchmark policy contains duplicate families")
    return set(families)


def _resolve_snapshot(root: Path, relative_path: str, label: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must stay under snapshot root")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} must not traverse symlinks")
    root_resolved = root.resolve()
    candidate = (root / relative).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"{label} escapes snapshot root")
    return candidate


def _assign_split(
    *,
    policy: Web4BenchmarkIntakePolicy,
    case_id: str,
    family: str,
    source_refs: list[str],
    source_snapshot_sha256s: dict[str, str],
) -> tuple[Web4BenchmarkSplit, str]:
    material = {
        "split_seed": policy.split_seed,
        "case_id": case_id,
        "family": family,
        "source_refs": source_refs,
        "source_snapshot_sha256s": source_snapshot_sha256s,
    }
    material_sha = _sha256_canonical(material)
    bucket = int(material_sha[:16], 16) % 10000
    if bucket < policy.development_bps:
        return Web4BenchmarkSplit.DEVELOPMENT, material_sha
    if bucket < policy.development_bps + policy.validation_bps:
        return Web4BenchmarkSplit.VALIDATION, material_sha
    return Web4BenchmarkSplit.HOLDOUT, material_sha


def build_web4_benchmark_intake(
    *,
    proposal_path: str | Path,
    answer_key_path: str | Path,
    source_registry_path: str | Path,
    benchmark_policy_path: str | Path,
    intake_policy_path: str | Path,
    snapshot_root: str | Path | None = None,
) -> Web4BenchmarkIntakePacket:
    proposal_payload, _ = _load_json(Path(proposal_path), "Web4 benchmark proposal")
    answer_payload, answer_key_sha = _load_json(
        Path(answer_key_path), "Web4 benchmark answer key"
    )
    benchmark_payload, benchmark_sha = _load_json(
        Path(benchmark_policy_path), "Web4 benchmark policy"
    )
    intake_payload, intake_sha = _load_json(
        Path(intake_policy_path), "Web4 benchmark intake policy"
    )
    source_by_id, source_registry_sha = _load_source_registry(Path(source_registry_path))

    proposal = Web4BenchmarkCaseProposal.model_validate(proposal_payload)
    answer_key = Web4BenchmarkAnswerKey.model_validate(answer_payload)
    intake_policy = Web4BenchmarkIntakePolicy.model_validate(intake_payload)
    required_families = _validate_benchmark_policy(benchmark_payload)

    if proposal.family not in required_families:
        raise ValueError(f"Web4 benchmark family is not required by policy: {proposal.family}")
    if answer_key.case_id != proposal.case_id:
        raise ValueError("Web4 answer key case_id differs from proposal")

    source_refs = sorted(source.source_ref for source in proposal.sources)
    if set(answer_key.source_refs) != set(source_refs):
        raise ValueError("Web4 answer key source refs differ from proposal")

    root = (
        Path(snapshot_root)
        if snapshot_root is not None
        else Path(proposal_path).resolve().parent
    )
    source_revision_status: dict[str, str] = {}
    source_snapshot_sha256s: dict[str, str] = {}

    for source_input in proposal.sources:
        source = source_by_id.get(source_input.source_ref)
        if source is None:
            raise ValueError(f"unknown Web4 benchmark source_ref: {source_input.source_ref}")
        if source.get("schema_version") != "sentinel.web4-source.v1":
            raise ValueError(f"Web4 source schema mismatch: {source_input.source_ref}")
        if (
            intake_policy.source_registry_eval_exclusion_required
            and source.get("eval_exclusion") is not True
        ):
            raise ValueError(
                f"Web4 source eval exclusion is not enabled: {source_input.source_ref}"
            )
        if (
            intake_policy.source_registry_training_closed_required
            and source.get("training_authorization") is not False
        ):
            raise ValueError(
                f"Web4 source training authorization is not closed: {source_input.source_ref}"
            )
        revision_status = source.get("revision_status")
        if not isinstance(revision_status, str) or not revision_status:
            raise ValueError(f"Web4 source lacks revision status: {source_input.source_ref}")

        snapshot = _resolve_snapshot(
            root,
            source_input.snapshot_path,
            f"Web4 source snapshot {source_input.source_ref}",
        )
        source_revision_status[source_input.source_ref] = revision_status
        source_snapshot_sha256s[source_input.source_ref] = _sha256_file(
            snapshot,
            f"Web4 source snapshot {source_input.source_ref}",
        )

    source_revision_status = dict(sorted(source_revision_status.items()))
    source_snapshot_sha256s = dict(sorted(source_snapshot_sha256s.items()))
    split, split_material_sha = _assign_split(
        policy=intake_policy,
        case_id=proposal.case_id,
        family=proposal.family,
        source_refs=source_refs,
        source_snapshot_sha256s=source_snapshot_sha256s,
    )

    unsigned: dict[str, object] = {
        "schema_version": "sentinel.web4-benchmark-intake-packet.v1",
        "case_id": proposal.case_id,
        "family": proposal.family,
        "created_at": proposal.created_at,
        "split": split.value,
        "split_seed": intake_policy.split_seed,
        "split_material_sha256": split_material_sha,
        "benchmark_policy_sha256": benchmark_sha,
        "intake_policy_sha256": intake_sha,
        "source_registry_sha256": source_registry_sha,
        "source_refs": source_refs,
        "source_revision_status": source_revision_status,
        "source_snapshot_sha256s": source_snapshot_sha256s,
        "model_input": proposal.model_input,
        "model_input_sha256": _sha256_canonical(proposal.model_input),
        "answer_key_sha256": answer_key_sha,
        "review_status": "UNREVIEWED",
        "human_reviewed": False,
        "contains_answer_key": False,
        "training_authorization": False,
        "evaluation_authorization": False,
        "promotion_eligible": False,
    }
    return Web4BenchmarkIntakePacket(
        **unsigned,
        packet_sha256=_sha256_canonical(unsigned),
    )


def write_web4_benchmark_intake_packet(
    packet: Web4BenchmarkIntakePacket,
    path: str | Path,
) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Web4 benchmark intake output already exists: {destination}")
    payload = json.dumps(packet.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    atomic_write(destination, payload)
