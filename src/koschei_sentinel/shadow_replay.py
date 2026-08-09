from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.models import StrictModel
from koschei_sentinel.promotion import (
    PromotionApproval,
    PromotionPolicy,
    PromotionProposal,
    load_owner_public_key,
    load_promotion_approval,
    load_promotion_policy,
    load_promotion_proposal,
    verify_promotion_approval,
)

_DIGEST = r"^[a-f0-9]{64}$"
_CANDIDATE_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_MAX_REPLAY_BYTES = 256 * 1024 * 1024
_MAX_REPLAY_CASES = 100_000


class ShadowReplayBlocked(ValueError):
    """Raised when a shadow-research replay plan is not safely reproducible."""


class ShadowReplayPlan(StrictModel):
    schema_version: Literal["sentinel.shadow-replay-plan.v1"] = (
        "sentinel.shadow-replay-plan.v1"
    )
    candidate_id: str = Field(pattern=_CANDIDATE_ID)
    state: Literal["sealed_shadow_replay"] = "sealed_shadow_replay"
    stage: Literal["shadow_research_candidate"] = "shadow_research_candidate"
    authority: Literal["explanation_only"] = "explanation_only"
    proposal_digest: str = Field(pattern=_DIGEST)
    approval_digest: str = Field(pattern=_DIGEST)
    promotion_policy_digest: str = Field(pattern=_DIGEST)
    finalization_digest: str = Field(pattern=_DIGEST)
    adapter_digest: str = Field(pattern=_DIGEST)
    benchmark_suite_digest: str = Field(pattern=_DIGEST)
    benchmark_report_digest: str = Field(pattern=_DIGEST)
    registry_digest: str = Field(pattern=_DIGEST)
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    approver_id: str
    replay_path: str
    replay_sha256: str = Field(pattern=_DIGEST)
    replay_cases: int = Field(ge=1, le=_MAX_REPLAY_CASES)
    output_dir: str
    manual_dispatch_required: Literal[True] = True
    network_access_allowed: Literal[False] = False
    live_chain_reads_allowed: Literal[False] = False
    live_customer_traffic_allowed: Literal[False] = False
    verdict_mutation_allowed: Literal[False] = False
    automatic_deployment_allowed: Literal[False] = False
    production_deployment_allowed: Literal[False] = False
    web3_runtime_integration_allowed: Literal[False] = False
    plan_digest: str = Field(pattern=_DIGEST)


def build_shadow_replay_plan(
    proposal: PromotionProposal,
    approval: PromotionApproval,
    policy: PromotionPolicy,
    *,
    owner_public_key_path: str | Path,
    replay_path: str | Path,
    output_dir: str | Path,
    root: str | Path = ".",
) -> ShadowReplayPlan:
    owner_public_key = load_owner_public_key(owner_public_key_path)
    verify_promotion_approval(proposal, approval, owner_public_key, policy)

    if proposal.requested_stage != "shadow_research_candidate":
        raise ShadowReplayBlocked("proposal does not authorize shadow research")
    if approval.approved_stage != "shadow_research_candidate":
        raise ShadowReplayBlocked("approval does not authorize shadow research")
    if proposal.authority != "explanation_only" or approval.authority != "explanation_only":
        raise ShadowReplayBlocked("shadow authority must remain explanation-only")

    root_path = Path(root).resolve()
    replay_relative, replay_file = _resolve_under_root(root_path, replay_path, "replay")
    output_relative, _ = _resolve_under_root(
        root_path,
        output_dir,
        "output directory",
        require_exists=False,
    )
    replay_sha256, replay_cases = _replay_identity(replay_file)

    payload = {
        "schema_version": "sentinel.shadow-replay-plan.v1",
        "candidate_id": proposal.candidate_id,
        "state": "sealed_shadow_replay",
        "stage": "shadow_research_candidate",
        "authority": "explanation_only",
        "proposal_digest": proposal.proposal_digest,
        "approval_digest": approval.approval_digest,
        "promotion_policy_digest": proposal.promotion_policy_digest,
        "finalization_digest": proposal.finalization_digest,
        "adapter_digest": proposal.adapter_digest,
        "benchmark_suite_digest": proposal.benchmark_suite_digest,
        "benchmark_report_digest": proposal.benchmark_report_digest,
        "registry_digest": proposal.registry_digest,
        "owner_key_fingerprint": proposal.owner_key_fingerprint,
        "approver_id": approval.approver_id,
        "replay_path": replay_relative,
        "replay_sha256": replay_sha256,
        "replay_cases": replay_cases,
        "output_dir": output_relative,
        "manual_dispatch_required": True,
        "network_access_allowed": False,
        "live_chain_reads_allowed": False,
        "live_customer_traffic_allowed": False,
        "verdict_mutation_allowed": False,
        "automatic_deployment_allowed": False,
        "production_deployment_allowed": False,
        "web3_runtime_integration_allowed": False,
    }
    return ShadowReplayPlan.model_validate(
        {**payload, "plan_digest": _digest(payload)}
    )


def load_shadow_replay_plan(path: str | Path) -> ShadowReplayPlan:
    try:
        plan = ShadowReplayPlan.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError("invalid shadow replay plan") from exc
    payload = plan.model_dump(mode="json")
    claimed = payload.pop("plan_digest")
    if claimed != _digest(payload):
        raise ShadowReplayBlocked("shadow replay plan digest does not match its contents")
    return plan


def write_shadow_replay_plan(plan: ShadowReplayPlan, path: str | Path) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"shadow replay plan already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
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


def _replay_identity(path: Path) -> tuple[str, int]:
    if not path.is_file():
        raise ShadowReplayBlocked(f"replay dataset is missing: {path}")
    size = path.stat().st_size
    if size == 0:
        raise ShadowReplayBlocked("replay dataset is empty")
    if size > _MAX_REPLAY_BYTES:
        raise ShadowReplayBlocked("replay dataset exceeds the offline size limit")

    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ShadowReplayBlocked("replay dataset must be UTF-8 JSONL") from exc

    identifiers: set[str] = set()
    cases = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ShadowReplayBlocked(f"blank replay row at line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ShadowReplayBlocked(
                f"invalid replay JSON at line {line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise ShadowReplayBlocked(f"replay row {line_number} must be an object")
        identifier = row.get("case_id", row.get("test_id"))
        if not isinstance(identifier, str) or not identifier or len(identifier) > 128:
            raise ShadowReplayBlocked(
                f"replay row {line_number} requires case_id or test_id"
            )
        if identifier in identifiers:
            raise ShadowReplayBlocked(f"duplicate replay identifier: {identifier}")
        identifiers.add(identifier)
        cases += 1
        if cases > _MAX_REPLAY_CASES:
            raise ShadowReplayBlocked("replay dataset exceeds the case limit")

    if cases == 0:
        raise ShadowReplayBlocked("replay dataset contains no cases")
    return hashlib.sha256(raw).hexdigest(), cases


def _resolve_under_root(
    root: Path,
    value: str | Path,
    label: str,
    *,
    require_exists: bool = True,
) -> tuple[str, Path]:
    supplied = Path(value)
    candidate = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    if candidate != root and root not in candidate.parents:
        raise ShadowReplayBlocked(f"{label} escapes the repository root")
    if require_exists and not candidate.exists():
        raise ShadowReplayBlocked(f"{label} is missing: {candidate}")
    relative = candidate.relative_to(root).as_posix()
    if not relative or relative == ".":
        raise ShadowReplayBlocked(f"{label} must not be the repository root")
    return relative, candidate


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ShadowReplayBlocked",
    "ShadowReplayPlan",
    "build_shadow_replay_plan",
    "load_promotion_approval",
    "load_promotion_policy",
    "load_promotion_proposal",
    "load_shadow_replay_plan",
    "write_shadow_replay_plan",
]
