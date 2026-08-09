from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.models import StrictModel
from koschei_sentinel.shadow_replay import ShadowReplayPlan

_DIGEST = r"^[a-f0-9]{64}$"
_CANDIDATE_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_MAX_RESULTS_BYTES = 256 * 1024 * 1024
_MAX_RESULTS_CASES = 100_000
_RESERVED_FALSE_FIELDS = (
    "automatic_deployment_allowed",
    "production_deployment_allowed",
    "web3_runtime_integration_allowed",
    "verdict_mutation_allowed",
    "live_customer_traffic_allowed",
)


class ShadowReceiptBlocked(ValueError):
    """Raised when shadow replay outputs cannot be bound to the sealed plan."""


class ShadowReplayReceipt(StrictModel):
    schema_version: Literal["sentinel.shadow-replay-receipt.v1"] = (
        "sentinel.shadow-replay-receipt.v1"
    )
    candidate_id: str = Field(pattern=_CANDIDATE_ID)
    state: Literal["completed_shadow_replay"] = "completed_shadow_replay"
    stage: Literal["shadow_research_candidate"] = "shadow_research_candidate"
    authority: Literal["explanation_only"] = "explanation_only"
    plan_digest: str = Field(pattern=_DIGEST)
    proposal_digest: str = Field(pattern=_DIGEST)
    approval_digest: str = Field(pattern=_DIGEST)
    replay_sha256: str = Field(pattern=_DIGEST)
    replay_cases: int = Field(ge=1, le=_MAX_RESULTS_CASES)
    results_path: str
    results_sha256: str = Field(pattern=_DIGEST)
    results_cases: int = Field(ge=1, le=_MAX_RESULTS_CASES)
    output_dir: str
    complete_case_coverage: Literal[True] = True
    ordered_case_identity_match: Literal[True] = True
    manual_review_required: Literal[True] = True
    benchmark_recheck_required: Literal[True] = True
    automatic_promotion_allowed: Literal[False] = False
    automatic_deployment_allowed: Literal[False] = False
    production_deployment_allowed: Literal[False] = False
    web3_runtime_integration_allowed: Literal[False] = False
    verdict_mutation_allowed: Literal[False] = False
    receipt_digest: str = Field(pattern=_DIGEST)


def build_shadow_replay_receipt(
    plan: ShadowReplayPlan,
    *,
    results_path: str | Path,
    root: str | Path = ".",
) -> ShadowReplayReceipt:
    _require_plan_digest(plan)
    root_path = Path(root).resolve()

    replay_file = _resolve_under_root(root_path, plan.replay_path, "replay")
    replay_raw = replay_file.read_bytes()
    replay_identifiers = _jsonl_identifiers(
        replay_raw,
        label="replay",
        candidate_id=None,
        plan_digest=None,
    )
    replay_sha256 = hashlib.sha256(replay_raw).hexdigest()
    if replay_sha256 != plan.replay_sha256:
        raise ShadowReceiptBlocked("replay bytes no longer match the sealed shadow plan")
    if len(replay_identifiers) != plan.replay_cases:
        raise ShadowReceiptBlocked("replay case count no longer matches the sealed shadow plan")

    output_dir = _resolve_under_root(
        root_path,
        plan.output_dir,
        "output directory",
        require_file=False,
        require_exists=False,
    )
    results_file = _resolve_under_root(root_path, results_path, "results")
    if output_dir not in results_file.parents:
        raise ShadowReceiptBlocked("results file must be inside the sealed output directory")

    results_raw = results_file.read_bytes()
    if len(results_raw) > _MAX_RESULTS_BYTES:
        raise ShadowReceiptBlocked("results dataset exceeds the offline size limit")
    result_identifiers = _jsonl_identifiers(
        results_raw,
        label="results",
        candidate_id=plan.candidate_id,
        plan_digest=plan.plan_digest,
    )
    if result_identifiers != replay_identifiers:
        raise ShadowReceiptBlocked("results do not cover replay identifiers in sealed order")

    payload = {
        "schema_version": "sentinel.shadow-replay-receipt.v1",
        "candidate_id": plan.candidate_id,
        "state": "completed_shadow_replay",
        "stage": "shadow_research_candidate",
        "authority": "explanation_only",
        "plan_digest": plan.plan_digest,
        "proposal_digest": plan.proposal_digest,
        "approval_digest": plan.approval_digest,
        "replay_sha256": plan.replay_sha256,
        "replay_cases": plan.replay_cases,
        "results_path": results_file.relative_to(root_path).as_posix(),
        "results_sha256": hashlib.sha256(results_raw).hexdigest(),
        "results_cases": len(result_identifiers),
        "output_dir": plan.output_dir,
        "complete_case_coverage": True,
        "ordered_case_identity_match": True,
        "manual_review_required": True,
        "benchmark_recheck_required": True,
        "automatic_promotion_allowed": False,
        "automatic_deployment_allowed": False,
        "production_deployment_allowed": False,
        "web3_runtime_integration_allowed": False,
        "verdict_mutation_allowed": False,
    }
    return ShadowReplayReceipt.model_validate(
        {**payload, "receipt_digest": _digest(payload)}
    )


def load_shadow_replay_receipt(path: str | Path) -> ShadowReplayReceipt:
    try:
        receipt = ShadowReplayReceipt.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid shadow replay receipt") from exc
    payload = receipt.model_dump(mode="json")
    claimed = payload.pop("receipt_digest")
    if claimed != _digest(payload):
        raise ShadowReceiptBlocked("shadow replay receipt digest does not match contents")
    return receipt


def write_shadow_replay_receipt(receipt: ShadowReplayReceipt, path: str | Path) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"shadow replay receipt already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _jsonl_identifiers(
    raw: bytes,
    *,
    label: str,
    candidate_id: str | None,
    plan_digest: str | None,
) -> tuple[str, ...]:
    if not raw:
        raise ShadowReceiptBlocked(f"{label} dataset is empty")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ShadowReceiptBlocked(f"{label} dataset must be UTF-8 JSONL") from exc

    identifiers: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ShadowReceiptBlocked(f"blank {label} row at line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ShadowReceiptBlocked(
                f"invalid {label} JSON at line {line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise ShadowReceiptBlocked(f"{label} row {line_number} must be an object")
        identifier = _row_identifier(row, label, line_number)
        if identifier in seen:
            raise ShadowReceiptBlocked(f"duplicate {label} identifier: {identifier}")
        seen.add(identifier)
        identifiers.append(identifier)
        if len(identifiers) > _MAX_RESULTS_CASES:
            raise ShadowReceiptBlocked(f"{label} dataset exceeds the case limit")

        if candidate_id is not None:
            if row.get("candidate_id") != candidate_id:
                raise ShadowReceiptBlocked(
                    f"results row {line_number} candidate_id does not match the plan"
                )
            if row.get("plan_digest") != plan_digest:
                raise ShadowReceiptBlocked(
                    f"results row {line_number} plan_digest does not match the plan"
                )
            if "authority" in row and row["authority"] != "explanation_only":
                raise ShadowReceiptBlocked(
                    f"results row {line_number} exceeds explanation-only authority"
                )
            for field in _RESERVED_FALSE_FIELDS:
                if row.get(field) is True:
                    raise ShadowReceiptBlocked(
                        f"results row {line_number} attempts to enable {field}"
                    )

    if not identifiers:
        raise ShadowReceiptBlocked(f"{label} dataset contains no cases")
    return tuple(identifiers)


def _row_identifier(row: dict[str, object], label: str, line_number: int) -> str:
    case_id = row.get("case_id")
    test_id = row.get("test_id")
    if case_id is not None and test_id is not None and case_id != test_id:
        raise ShadowReceiptBlocked(
            f"{label} row {line_number} has conflicting case_id and test_id"
        )
    identifier = case_id if case_id is not None else test_id
    if not isinstance(identifier, str) or not identifier or len(identifier) > 128:
        raise ShadowReceiptBlocked(
            f"{label} row {line_number} requires case_id or test_id"
        )
    return identifier


def _resolve_under_root(
    root: Path,
    value: str | Path,
    label: str,
    *,
    require_file: bool = True,
    require_exists: bool = True,
) -> Path:
    supplied = Path(value)
    candidate = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    if candidate != root and root not in candidate.parents:
        raise ShadowReceiptBlocked(f"{label} escapes the repository root")
    if require_exists and not candidate.exists():
        raise ShadowReceiptBlocked(f"{label} is missing: {candidate}")
    if require_file and require_exists and not candidate.is_file():
        raise ShadowReceiptBlocked(f"{label} is not a file: {candidate}")
    return candidate


def _require_plan_digest(plan: ShadowReplayPlan) -> None:
    payload = plan.model_dump(mode="json")
    claimed = payload.pop("plan_digest")
    if claimed != _digest(payload):
        raise ShadowReceiptBlocked("shadow replay plan digest does not match contents")


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ShadowReceiptBlocked",
    "ShadowReplayReceipt",
    "build_shadow_replay_receipt",
    "load_shadow_replay_receipt",
    "write_shadow_replay_receipt",
]
