from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.strict_json import require_json_value, strict_json_loads
from koschei_sentinel.training import atomic_write

_DIGEST = r"^[a-f0-9]{64}$"


class Web4ResearchSnapshotReceipt(StrictModel):
    schema_version: Literal["sentinel.web4-research-snapshot-receipt.v1"] = (
        "sentinel.web4-research-snapshot-receipt.v1"
    )
    source_id: str = Field(min_length=3, max_length=256)
    source_type: str = Field(min_length=1, max_length=256)
    authority_tier: str = Field(min_length=1, max_length=128)
    canonical_locator: str = Field(min_length=1, max_length=4096)
    publication_date: str | None
    revision_status: str = Field(min_length=1, max_length=256)
    registry_license_status: Literal["REVIEW_REQUIRED"] = "REVIEW_REQUIRED"
    registry_eval_exclusion: Literal[True] = True
    source_registry_sha256: str = Field(pattern=_DIGEST)
    source_row_sha256: str = Field(pattern=_DIGEST)
    captured_at: str = Field(min_length=10, max_length=64)
    snapshot_name: str = Field(min_length=1, max_length=1024)
    snapshot_sha256: str = Field(pattern=_DIGEST)
    snapshot_size_bytes: int = Field(ge=1)
    collection_method: Literal["OPERATOR_SUPPLIED_LOCAL_FILE"] = "OPERATOR_SUPPLIED_LOCAL_FILE"
    snapshot_state: Literal["UNREVIEWED_RESEARCH_CAPTURE"] = "UNREVIEWED_RESEARCH_CAPTURE"
    source_match_verified: Literal[False] = False
    provenance_review_status: Literal["REVIEW_REQUIRED"] = "REVIEW_REQUIRED"
    network_fetch_performed: Literal[False] = False
    corpus_materialized: Literal[False] = False
    training_authorization: Literal[False] = False
    evaluation_authorization: Literal[False] = False
    promotion_eligible: Literal[False] = False
    production_activation_allowed: Literal[False] = False
    receipt_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def receipt_contract_verifies(self) -> Web4ResearchSnapshotReceipt:
        _parse_timestamp(self.captured_at)
        unsigned = self.model_dump(mode="json")
        observed = str(unsigned.pop("receipt_sha256"))
        if _sha256_canonical(unsigned) != observed:
            raise ValueError("Web4 research snapshot receipt self-hash does not verify")
        return self


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Web4 research snapshot captured_at must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Web4 research snapshot captured_at must include a timezone")
    return parsed


def _sha256_canonical(payload: object) -> str:
    require_json_value(payload)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_regular(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return path.read_bytes()


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
            row = strict_json_loads(line)
        except ValueError as exc:
            raise ValueError(f"invalid Web4 source row at line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Web4 source row {line_number} must be a JSON object")
        source_id = row.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"Web4 source row {line_number} lacks source_id")
        if source_id in by_id:
            raise ValueError(f"duplicate Web4 source_id: {source_id}")
        by_id[source_id] = row
    if not by_id:
        raise ValueError("Web4 source registry is empty")
    return by_id, hashlib.sha256(raw).hexdigest()


def _validate_source_for_research_capture(source: dict[str, object], source_id: str) -> None:
    if source.get("schema_version") != "sentinel.web4-source.v1":
        raise ValueError(f"Web4 source schema mismatch: {source_id}")
    if source.get("training_authorization") is not False:
        raise ValueError(f"Web4 source training authorization is not closed: {source_id}")
    if source.get("eval_exclusion") is not True:
        raise ValueError(f"Web4 source eval exclusion is not enabled: {source_id}")
    if source.get("license_status") != "REVIEW_REQUIRED":
        raise ValueError(f"Web4 source license review state is not REVIEW_REQUIRED: {source_id}")
    for field in ("source_type", "authority_tier", "canonical_locator", "revision_status"):
        value = source.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Web4 source lacks required {field}: {source_id}")


def _hash_stable_snapshot(path: Path) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Web4 research snapshot must be a regular non-symlink file")
    before = path.stat()
    if before.st_size <= 0:
        raise ValueError("Web4 research snapshot must not be empty")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)

    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ino != after.st_ino
        or before.st_dev != after.st_dev
    ):
        raise ValueError("Web4 research snapshot changed while hashing")
    return digest.hexdigest(), after.st_size


def build_web4_research_snapshot_receipt(
    *,
    source_id: str,
    snapshot_path: str | Path,
    captured_at: str,
    source_registry_path: str | Path,
) -> Web4ResearchSnapshotReceipt:
    _parse_timestamp(captured_at)
    sources, registry_sha = _load_source_registry(Path(source_registry_path))
    source = sources.get(source_id)
    if source is None:
        raise ValueError(f"unknown Web4 research snapshot source_id: {source_id}")
    _validate_source_for_research_capture(source, source_id)

    snapshot = Path(snapshot_path)
    snapshot_sha, snapshot_size = _hash_stable_snapshot(snapshot)
    source_row_sha = _sha256_canonical(source)

    unsigned: dict[str, object] = {
        "schema_version": "sentinel.web4-research-snapshot-receipt.v1",
        "source_id": source_id,
        "source_type": str(source["source_type"]),
        "authority_tier": str(source["authority_tier"]),
        "canonical_locator": str(source["canonical_locator"]),
        "publication_date": source.get("publication_date"),
        "revision_status": str(source["revision_status"]),
        "registry_license_status": "REVIEW_REQUIRED",
        "registry_eval_exclusion": True,
        "source_registry_sha256": registry_sha,
        "source_row_sha256": source_row_sha,
        "captured_at": captured_at,
        "snapshot_name": snapshot.name,
        "snapshot_sha256": snapshot_sha,
        "snapshot_size_bytes": snapshot_size,
        "collection_method": "OPERATOR_SUPPLIED_LOCAL_FILE",
        "snapshot_state": "UNREVIEWED_RESEARCH_CAPTURE",
        "source_match_verified": False,
        "provenance_review_status": "REVIEW_REQUIRED",
        "network_fetch_performed": False,
        "corpus_materialized": False,
        "training_authorization": False,
        "evaluation_authorization": False,
        "promotion_eligible": False,
        "production_activation_allowed": False,
    }
    return Web4ResearchSnapshotReceipt(
        **unsigned,
        receipt_sha256=_sha256_canonical(unsigned),
    )


def verify_web4_research_snapshot_receipt(
    *,
    receipt_path: str | Path,
    snapshot_path: str | Path,
    source_registry_path: str | Path,
) -> Web4ResearchSnapshotReceipt:
    raw = _read_regular(Path(receipt_path), "Web4 research snapshot receipt")
    try:
        payload = strict_json_loads(raw)
    except ValueError as exc:
        raise ValueError("invalid Web4 research snapshot receipt JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Web4 research snapshot receipt must contain one JSON object")
    receipt = Web4ResearchSnapshotReceipt.model_validate(payload)

    rebuilt = build_web4_research_snapshot_receipt(
        source_id=receipt.source_id,
        snapshot_path=snapshot_path,
        captured_at=receipt.captured_at,
        source_registry_path=source_registry_path,
    )
    if rebuilt.model_dump(mode="json") != receipt.model_dump(mode="json"):
        raise ValueError("Web4 research snapshot receipt differs from current source/snapshot binding")
    return receipt


def write_web4_research_snapshot_receipt(
    receipt: Web4ResearchSnapshotReceipt,
    path: str | Path,
) -> None:
    payload = json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    atomic_write(Path(path), payload)
