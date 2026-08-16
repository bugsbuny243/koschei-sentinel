from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import Field, model_validator

from koschei_sentinel.blockchain_base_candidates import load_base_candidate_registry
from koschei_sentinel.blockchain_runtime_preflight import (
    BlockchainRuntimePreflightAudit,
    load_hardware_inventory,
    load_runtime_preflight_plan,
    load_runtime_probe_receipt,
)
from koschei_sentinel.blockchain_security_release import verify_blockchain_security_release
from koschei_sentinel.blockchain_training import BlockchainTrainingConfig
from koschei_sentinel.models import StrictModel
from koschei_sentinel.promotion import public_key_fingerprint
from koschei_sentinel.training import model_digest, resolve_under_root

_DIGEST = r"^[a-f0-9]{64}$"
_COMMIT = r"^[a-f0-9]{40}$"
_TREE = r"^[a-f0-9]{40}$"
_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_MODEL_ID = r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$"
_APPROVER_ID = r"^[A-Za-z0-9][A-Za-z0-9._@-]{0,127}$"
_SIGNATURE_CONTEXT = b"koschei-sentinel-blockchain-training-authorization-v1\0"
_REQUIRED_PREFLIGHT_ARTIFACTS = frozenset(
    {
        "hardware-inventory.json",
        "preflight-plan.json",
        "runtime-probe-receipt.json",
        "preflight-audit.json",
        "summary.json",
    }
)


class BlockchainTrainingAuthorizationBlocked(ValueError):
    """Raised when blockchain training authorization fails closed."""


class BlockchainRuntimePreflightSeal(StrictModel):
    schema_version: Literal["sentinel.blockchain-runtime-preflight-seal.v1"] = (
        "sentinel.blockchain-runtime-preflight-seal.v1"
    )
    candidate_id: str = Field(pattern=_ID)
    model_id: str = Field(pattern=_MODEL_ID)
    model_revision: str = Field(pattern=_COMMIT)
    github_base_commit: str = Field(pattern=_COMMIT)
    normalized_source_tree: str = Field(pattern=_TREE)
    source_patch_sha256: str = Field(pattern=_DIGEST)
    gpu_model: str = Field(min_length=1, max_length=256)
    available_single_gpu_vram_mb: int = Field(ge=1)
    peak_gpu_memory_mb: int = Field(ge=1)
    headroom_mb: int = Field(ge=0)
    runtime_probe_status: Literal["PASSED"] = "PASSED"
    ready_for_training_authorization_review: Literal[True] = True
    training_started: Literal[False] = False
    training_authorized: Literal[False] = False
    production_authority: Literal[False] = False
    raw_model_outputs_stored: Literal[False] = False
    persistent_run: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
    artifact_digests: dict[str, str] = Field(min_length=5, max_length=32)
    seal_digest: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def seal_is_canonical(self) -> BlockchainRuntimePreflightSeal:
        payload = self.model_dump(mode="json")
        expected = payload.pop("seal_digest")
        if _digest(payload) != expected:
            raise ValueError("blockchain runtime preflight seal digest mismatch")
        if self.peak_gpu_memory_mb > self.available_single_gpu_vram_mb:
            raise ValueError("preflight seal peak GPU memory exceeds available VRAM")
        expected_headroom = self.available_single_gpu_vram_mb - self.peak_gpu_memory_mb
        if self.headroom_mb != expected_headroom:
            raise ValueError("preflight seal GPU headroom mismatch")
        if not _REQUIRED_PREFLIGHT_ARTIFACTS.issubset(self.artifact_digests):
            raise ValueError("preflight seal is missing required runtime artifacts")
        for name, digest in self.artifact_digests.items():
            _validate_plain_artifact_name(name)
            if not _is_digest(digest):
                raise ValueError("preflight seal contains invalid artifact digest")
        return self


class BlockchainTrainingAuthorizationPolicy(StrictModel):
    schema_version: Literal["sentinel.blockchain-training-authorization-policy.v1"] = (
        "sentinel.blockchain-training-authorization-policy.v1"
    )
    policy_id: str = Field(pattern=_ID)
    required_candidate_id: str = Field(pattern=_ID)
    required_model_id: str = Field(pattern=_MODEL_ID)
    required_model_revision: str = Field(pattern=_COMMIT)
    require_exact_registry_binding: Literal[True] = True
    require_passing_preflight: Literal[True] = True
    require_verified_release: Literal[True] = True
    require_owner_signature: Literal[True] = True
    automatic_training_allowed: Literal[False] = False
    production_authority_allowed: Literal[False] = False
    web3_runtime_authority_allowed: Literal[False] = False


class BlockchainTrainingAuthorizationProposal(StrictModel):
    schema_version: Literal["sentinel.blockchain-training-authorization-proposal.v1"] = (
        "sentinel.blockchain-training-authorization-proposal.v1"
    )
    authorization_id: str = Field(pattern=_ID)
    state: Literal["awaiting_owner_signature"] = "awaiting_owner_signature"
    authority: Literal["offline_blockchain_research_only"] = "offline_blockchain_research_only"
    candidate_id: str = Field(pattern=_ID)
    model_id: str = Field(pattern=_MODEL_ID)
    model_revision: str = Field(pattern=_COMMIT)
    github_base_commit: str = Field(pattern=_COMMIT)
    normalized_source_tree: str = Field(pattern=_TREE)
    source_patch_sha256: str = Field(pattern=_DIGEST)
    preflight_seal_digest: str = Field(pattern=_DIGEST)
    registry_digest: str = Field(pattern=_DIGEST)
    candidate_digest: str = Field(pattern=_DIGEST)
    preflight_policy_digest: str = Field(pattern=_DIGEST)
    hardware_inventory_digest: str = Field(pattern=_DIGEST)
    preflight_plan_digest: str = Field(pattern=_DIGEST)
    preflight_receipt_digest: str = Field(pattern=_DIGEST)
    release_manifest_file_digest: str = Field(pattern=_DIGEST)
    release_manifest_digest: str = Field(pattern=_DIGEST)
    source_corpus_digest: str = Field(pattern=_DIGEST)
    benchmark_suite_digest: str = Field(pattern=_DIGEST)
    holdout_digest: str = Field(pattern=_DIGEST)
    train_split_digest: str = Field(pattern=_DIGEST)
    validation_split_digest: str = Field(pattern=_DIGEST)
    held_out_test_split_digest: str = Field(pattern=_DIGEST)
    training_config_digest: str = Field(pattern=_DIGEST)
    authorization_policy_digest: str = Field(pattern=_DIGEST)
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    owner_signature_required: Literal[True] = True
    automatic_training_allowed: Literal[False] = False
    training_started: Literal[False] = False
    training_authorized: Literal[False] = False
    production_authority: Literal[False] = False
    web3_runtime_authority: Literal[False] = False
    proposal_digest: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def proposal_digest_is_valid(self) -> BlockchainTrainingAuthorizationProposal:
        _require_model_digest(self, "proposal_digest", "training authorization proposal")
        return self


class BlockchainTrainingAuthorizationApproval(StrictModel):
    schema_version: Literal["sentinel.blockchain-training-authorization-approval.v1"] = (
        "sentinel.blockchain-training-authorization-approval.v1"
    )
    authorization_id: str = Field(pattern=_ID)
    state: Literal["owner_approved_offline_training"] = "owner_approved_offline_training"
    authority: Literal["offline_blockchain_research_only"] = "offline_blockchain_research_only"
    candidate_id: str = Field(pattern=_ID)
    approver_id: str = Field(pattern=_APPROVER_ID)
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    authorization_policy_digest: str = Field(pattern=_DIGEST)
    proposal_digest: str = Field(pattern=_DIGEST)
    signature_algorithm: Literal["ed25519"] = "ed25519"
    signature_base64: str = Field(min_length=80, max_length=128)
    signature_verified: Literal[True] = True
    automatic_training_allowed: Literal[False] = False
    training_started: Literal[False] = False
    training_authorized: Literal[True] = True
    production_authority: Literal[False] = False
    web3_runtime_authority: Literal[False] = False
    approval_digest: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def approval_digest_is_valid(self) -> BlockchainTrainingAuthorizationApproval:
        _require_model_digest(self, "approval_digest", "training authorization approval")
        return self


def load_preflight_seal(path: str | Path) -> BlockchainRuntimePreflightSeal:
    try:
        return BlockchainRuntimePreflightSeal.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise BlockchainTrainingAuthorizationBlocked("invalid runtime preflight seal") from exc


def load_training_authorization_policy(
    path: str | Path,
) -> BlockchainTrainingAuthorizationPolicy:
    try:
        return BlockchainTrainingAuthorizationPolicy.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise BlockchainTrainingAuthorizationBlocked(
            "invalid blockchain training authorization policy"
        ) from exc


def load_training_authorization_proposal(
    path: str | Path,
) -> BlockchainTrainingAuthorizationProposal:
    try:
        return BlockchainTrainingAuthorizationProposal.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise BlockchainTrainingAuthorizationBlocked(
            "invalid blockchain training authorization proposal"
        ) from exc


def load_training_authorization_approval(
    path: str | Path,
) -> BlockchainTrainingAuthorizationApproval:
    try:
        return BlockchainTrainingAuthorizationApproval.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise BlockchainTrainingAuthorizationBlocked(
            "invalid blockchain training authorization approval"
        ) from exc


def training_authorization_policy_digest(policy: BlockchainTrainingAuthorizationPolicy) -> str:
    return _digest(policy.model_dump(mode="json"))


def build_training_authorization_proposal(
    *,
    authorization_id: str,
    seal_path: str | Path,
    preflight_run_dir: str | Path,
    registry_path: str | Path,
    release_dir: str | Path,
    training_config: BlockchainTrainingConfig,
    owner_public_key: Ed25519PublicKey,
    policy: BlockchainTrainingAuthorizationPolicy,
    root: str | Path = ".",
) -> BlockchainTrainingAuthorizationProposal:
    evidence = _verify_authorization_inputs(
        seal_path=seal_path,
        preflight_run_dir=preflight_run_dir,
        registry_path=registry_path,
        release_dir=release_dir,
        training_config=training_config,
        policy=policy,
        root=root,
    )
    payload = {
        "schema_version": "sentinel.blockchain-training-authorization-proposal.v1",
        "authorization_id": authorization_id,
        "state": "awaiting_owner_signature",
        "authority": "offline_blockchain_research_only",
        **evidence,
        "authorization_policy_digest": training_authorization_policy_digest(policy),
        "owner_key_fingerprint": public_key_fingerprint(owner_public_key),
        "owner_signature_required": True,
        "automatic_training_allowed": False,
        "training_started": False,
        "training_authorized": False,
        "production_authority": False,
        "web3_runtime_authority": False,
    }
    return BlockchainTrainingAuthorizationProposal.model_validate(
        {**payload, "proposal_digest": _digest(payload)}
    )


def approve_training_authorization_proposal(
    proposal: BlockchainTrainingAuthorizationProposal,
    owner_private_key: Ed25519PrivateKey,
    policy: BlockchainTrainingAuthorizationPolicy,
    *,
    approver_id: str,
) -> BlockchainTrainingAuthorizationApproval:
    _require_model_digest(proposal, "proposal_digest", "training authorization proposal")
    _require_policy_binding(proposal, policy)
    fingerprint = public_key_fingerprint(owner_private_key.public_key())
    if fingerprint != proposal.owner_key_fingerprint:
        raise BlockchainTrainingAuthorizationBlocked(
            "private key does not match training authorization proposal"
        )
    signature = owner_private_key.sign(_signature_message(proposal.proposal_digest))
    payload = {
        "schema_version": "sentinel.blockchain-training-authorization-approval.v1",
        "authorization_id": proposal.authorization_id,
        "state": "owner_approved_offline_training",
        "authority": "offline_blockchain_research_only",
        "candidate_id": proposal.candidate_id,
        "approver_id": approver_id,
        "owner_key_fingerprint": fingerprint,
        "authorization_policy_digest": proposal.authorization_policy_digest,
        "proposal_digest": proposal.proposal_digest,
        "signature_algorithm": "ed25519",
        "signature_base64": base64.b64encode(signature).decode("ascii"),
        "signature_verified": True,
        "automatic_training_allowed": False,
        "training_started": False,
        "training_authorized": True,
        "production_authority": False,
        "web3_runtime_authority": False,
    }
    return BlockchainTrainingAuthorizationApproval.model_validate(
        {**payload, "approval_digest": _digest(payload)}
    )


def verify_training_authorization_bundle(
    *,
    proposal: BlockchainTrainingAuthorizationProposal,
    approval: BlockchainTrainingAuthorizationApproval,
    owner_public_key: Ed25519PublicKey,
    policy: BlockchainTrainingAuthorizationPolicy,
    seal_path: str | Path,
    preflight_run_dir: str | Path,
    registry_path: str | Path,
    release_dir: str | Path,
    training_config: BlockchainTrainingConfig,
    root: str | Path = ".",
) -> BlockchainTrainingAuthorizationApproval:
    _require_model_digest(proposal, "proposal_digest", "training authorization proposal")
    _require_model_digest(approval, "approval_digest", "training authorization approval")
    _require_policy_binding(proposal, policy)
    expected = build_training_authorization_proposal(
        authorization_id=proposal.authorization_id,
        seal_path=seal_path,
        preflight_run_dir=preflight_run_dir,
        registry_path=registry_path,
        release_dir=release_dir,
        training_config=training_config,
        owner_public_key=owner_public_key,
        policy=policy,
        root=root,
    )
    if expected != proposal:
        raise BlockchainTrainingAuthorizationBlocked(
            "authorization proposal no longer matches current immutable inputs"
        )
    fingerprint = public_key_fingerprint(owner_public_key)
    if approval.authorization_id != proposal.authorization_id:
        raise BlockchainTrainingAuthorizationBlocked("approval authorization_id mismatch")
    if approval.candidate_id != proposal.candidate_id:
        raise BlockchainTrainingAuthorizationBlocked("approval candidate mismatch")
    if approval.owner_key_fingerprint != fingerprint:
        raise BlockchainTrainingAuthorizationBlocked("approval owner key mismatch")
    if approval.proposal_digest != proposal.proposal_digest:
        raise BlockchainTrainingAuthorizationBlocked("approval proposal digest mismatch")
    if approval.authorization_policy_digest != proposal.authorization_policy_digest:
        raise BlockchainTrainingAuthorizationBlocked("approval policy digest mismatch")
    try:
        signature = base64.b64decode(approval.signature_base64, validate=True)
        owner_public_key.verify(signature, _signature_message(proposal.proposal_digest))
    except (InvalidSignature, ValueError) as exc:
        raise BlockchainTrainingAuthorizationBlocked(
            "training authorization owner signature verification failed"
        ) from exc
    return approval


def write_training_authorization_artifact(model: StrictModel, path: str | Path) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"training authorization artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
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


def _verify_authorization_inputs(
    *,
    seal_path: str | Path,
    preflight_run_dir: str | Path,
    registry_path: str | Path,
    release_dir: str | Path,
    training_config: BlockchainTrainingConfig,
    policy: BlockchainTrainingAuthorizationPolicy,
    root: str | Path,
) -> dict[str, object]:
    root_path = Path(root).resolve()
    seal = load_preflight_seal(resolve_under_root(root_path, seal_path))
    run_dir = resolve_under_root(root_path, preflight_run_dir)
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise BlockchainTrainingAuthorizationBlocked("preflight run directory is missing")
    if run_dir.name != seal.persistent_run:
        raise BlockchainTrainingAuthorizationBlocked(
            "preflight run directory does not match sealed persistent_run"
        )
    for name, expected_digest in seal.artifact_digests.items():
        artifact = run_dir / name
        if not artifact.is_file() or artifact.is_symlink():
            raise BlockchainTrainingAuthorizationBlocked(
                f"sealed preflight artifact is missing: {name}"
            )
        if _hash_file(artifact) != expected_digest:
            raise BlockchainTrainingAuthorizationBlocked(
                f"sealed preflight artifact digest mismatch: {name}"
            )

    hardware = load_hardware_inventory(run_dir / "hardware-inventory.json")
    plan = load_runtime_preflight_plan(run_dir / "preflight-plan.json")
    receipt = load_runtime_probe_receipt(run_dir / "runtime-probe-receipt.json")
    try:
        audit = BlockchainRuntimePreflightAudit.model_validate_json(
            (run_dir / "preflight-audit.json").read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise BlockchainTrainingAuthorizationBlocked("invalid preflight audit") from exc
    summary = _load_json_object(run_dir / "summary.json", "preflight summary")

    if receipt.status != "PASSED" or not audit.ready_for_training_authorization_review:
        raise BlockchainTrainingAuthorizationBlocked("runtime preflight did not pass")
    if not plan.static_hardware_fit or not plan.runtime_probe_authorized or plan.blockers:
        raise BlockchainTrainingAuthorizationBlocked("runtime preflight plan is not authorized")
    if audit.plan_digest != plan.plan_digest or audit.receipt_digest != receipt.receipt_digest:
        raise BlockchainTrainingAuthorizationBlocked("runtime preflight audit lineage mismatch")
    if receipt.inventory_digest != hardware.inventory_digest:
        raise BlockchainTrainingAuthorizationBlocked("runtime preflight hardware lineage mismatch")
    if summary.get("runtime_probe_status") != "PASSED" or summary.get(
        "ready_for_training_authorization_review"
    ) is not True:
        raise BlockchainTrainingAuthorizationBlocked("preflight summary is not authorization-ready")

    _require_policy_candidate(seal, policy)
    if plan.candidate_id != seal.candidate_id:
        raise BlockchainTrainingAuthorizationBlocked("preflight plan candidate does not match seal")
    if plan.model_id != seal.model_id or plan.revision != seal.model_revision:
        raise BlockchainTrainingAuthorizationBlocked("preflight plan model pin does not match seal")
    if receipt.candidate_id != seal.candidate_id:
        raise BlockchainTrainingAuthorizationBlocked("preflight receipt candidate does not match seal")
    if receipt.model_id != seal.model_id or receipt.revision != seal.model_revision:
        raise BlockchainTrainingAuthorizationBlocked("preflight receipt model pin does not match seal")

    registry = load_base_candidate_registry(resolve_under_root(root_path, registry_path))
    candidates = [item for item in registry.candidates if item.candidate_id == seal.candidate_id]
    if len(candidates) != 1:
        raise BlockchainTrainingAuthorizationBlocked("sealed candidate is absent or duplicated")
    candidate = candidates[0]
    if candidate.model_id != seal.model_id or candidate.revision != seal.model_revision:
        raise BlockchainTrainingAuthorizationBlocked("registry candidate pin does not match seal")
    candidate_digest = model_digest(candidate)
    if plan.registry_digest != registry.registry_digest or receipt.registry_digest != registry.registry_digest:
        raise BlockchainTrainingAuthorizationBlocked("preflight registry digest is stale")
    if plan.candidate_digest != candidate_digest or receipt.candidate_digest != candidate_digest:
        raise BlockchainTrainingAuthorizationBlocked("preflight candidate digest is stale")
    if plan.policy_digest != receipt.policy_digest:
        raise BlockchainTrainingAuthorizationBlocked("preflight policy lineage mismatch")

    release = resolve_under_root(root_path, release_dir)
    manifest = verify_blockchain_security_release(release)
    manifest_path = release / "blockchain-security-release-manifest.json"
    if training_config.base_model != seal.model_id:
        raise BlockchainTrainingAuthorizationBlocked("training config base model does not match seal")
    if training_config.base_revision != seal.model_revision:
        raise BlockchainTrainingAuthorizationBlocked("training config base revision does not match seal")
    configured_release = resolve_under_root(root_path, training_config.blockchain_release)
    if configured_release != release:
        raise BlockchainTrainingAuthorizationBlocked("training config release path does not match authorization")
    if training_config.expected_source_corpus_digest != manifest.source_corpus_digest:
        raise BlockchainTrainingAuthorizationBlocked("training config source corpus pin is stale")
    if training_config.expected_benchmark_suite_digest != manifest.benchmark_suite_digest:
        raise BlockchainTrainingAuthorizationBlocked("training config benchmark suite pin is stale")

    return {
        "candidate_id": seal.candidate_id,
        "model_id": seal.model_id,
        "model_revision": seal.model_revision,
        "github_base_commit": seal.github_base_commit,
        "normalized_source_tree": seal.normalized_source_tree,
        "source_patch_sha256": seal.source_patch_sha256,
        "preflight_seal_digest": seal.seal_digest,
        "registry_digest": registry.registry_digest,
        "candidate_digest": candidate_digest,
        "preflight_policy_digest": plan.policy_digest,
        "hardware_inventory_digest": hardware.inventory_digest,
        "preflight_plan_digest": plan.plan_digest,
        "preflight_receipt_digest": receipt.receipt_digest,
        "release_manifest_file_digest": _hash_file(manifest_path),
        "release_manifest_digest": model_digest(manifest),
        "source_corpus_digest": manifest.source_corpus_digest,
        "benchmark_suite_digest": manifest.benchmark_suite_digest,
        "holdout_digest": manifest.holdout_digest,
        "train_split_digest": manifest.splits["train"].digest,
        "validation_split_digest": manifest.splits["validation"].digest,
        "held_out_test_split_digest": manifest.splits["test"].digest,
        "training_config_digest": model_digest(training_config),
    }


def _require_policy_candidate(
    seal: BlockchainRuntimePreflightSeal,
    policy: BlockchainTrainingAuthorizationPolicy,
) -> None:
    if seal.candidate_id != policy.required_candidate_id:
        raise BlockchainTrainingAuthorizationBlocked("preflight candidate is not policy-authorized")
    if seal.model_id != policy.required_model_id:
        raise BlockchainTrainingAuthorizationBlocked("preflight model is not policy-authorized")
    if seal.model_revision != policy.required_model_revision:
        raise BlockchainTrainingAuthorizationBlocked("preflight revision is not policy-authorized")


def _require_policy_binding(
    proposal: BlockchainTrainingAuthorizationProposal,
    policy: BlockchainTrainingAuthorizationPolicy,
) -> None:
    if proposal.authorization_policy_digest != training_authorization_policy_digest(policy):
        raise BlockchainTrainingAuthorizationBlocked(
            "training authorization proposal does not match supplied policy"
        )
    if proposal.candidate_id != policy.required_candidate_id:
        raise BlockchainTrainingAuthorizationBlocked("proposal candidate is not policy-authorized")
    if proposal.model_id != policy.required_model_id:
        raise BlockchainTrainingAuthorizationBlocked("proposal model is not policy-authorized")
    if proposal.model_revision != policy.required_model_revision:
        raise BlockchainTrainingAuthorizationBlocked("proposal revision is not policy-authorized")


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BlockchainTrainingAuthorizationBlocked(f"invalid {label}") from exc
    if not isinstance(value, dict):
        raise BlockchainTrainingAuthorizationBlocked(f"invalid {label}")
    return value


def _validate_plain_artifact_name(value: str) -> None:
    parsed = PurePosixPath(value)
    if not value or parsed.is_absolute() or len(parsed.parts) != 1 or value.startswith("~"):
        raise ValueError("preflight artifact name must be a plain file name")
    if "\\" in value or ".." in parsed.parts:
        raise ValueError("preflight artifact name is unsafe")


def _require_model_digest(model: StrictModel, field: str, label: str) -> None:
    payload = model.model_dump(mode="json")
    claimed = payload.pop(field)
    if claimed != _digest(payload):
        raise BlockchainTrainingAuthorizationBlocked(f"{label} digest mismatch")


def _signature_message(proposal_digest: str) -> bytes:
    return _SIGNATURE_CONTEXT + proposal_digest.encode("ascii")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_digest(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
