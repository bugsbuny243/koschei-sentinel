from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.cyber_megatron_training import (
    LAUNCH_STATE_FILENAME,
    QWEN35_397B_MODEL,
    QWEN35_397B_REVISION,
    RUN_IDENTITY_FILENAME,
    CyberMegatronDatasetManifest,
    CyberMegatronLaunchState,
    CyberMegatronPlan,
    CyberMegatronSFTConfig,
    _config_sha256,
    _load_run_identity,
    _plan_sha256,
    load_cyber_megatron_config,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import atomic_write, canonical_json

CANDIDATE_FILENAME = "koschei-397b-candidate.json"
_DIGEST = r"^[a-f0-9]{64}$"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_canonical(payload: object) -> str:
    return _sha256_bytes(canonical_json(payload).encode("utf-8"))


def _assert_relative_safe(value: str, field_name: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or value.startswith("~")
        or "\\" in value
    ):
        raise ValueError(f"{field_name} must be a portable relative POSIX path")


def _resolve_no_symlinks(root: Path, value: str | Path, label: str) -> Path:
    requested = Path(value)
    lexical = requested if requested.is_absolute() else root / requested
    absolute = Path(os.path.abspath(lexical))
    if absolute != root and root not in absolute.parents:
        raise ValueError(f"{label} escapes the repository root")
    relative_parts = () if absolute == root else absolute.relative_to(root).parts
    current = root
    for part in relative_parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink: {current}")
    resolved = absolute.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} resolves outside the repository root")
    return resolved


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _sha256_stable_regular_file(path: Path) -> tuple[int, str]:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"397B checkpoint file cannot be opened safely: {path.name}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"397B checkpoint contains a non-regular entry: {path.name}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = path.lstat()
    except OSError as exc:
        raise ValueError(f"397B checkpoint file disappeared while hashing: {path.name}") from exc
    if (
        _file_identity(before) != _file_identity(after)
        or _file_identity(after) != _file_identity(path_after)
    ):
        raise ValueError(f"397B checkpoint file changed while hashing: {path.name}")
    return after.st_size, digest.hexdigest()


class CyberMegatronCheckpointFile(StrictModel):
    path: str = Field(min_length=1, max_length=4096)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def path_is_portable(self) -> CyberMegatronCheckpointFile:
        _assert_relative_safe(self.path, "checkpoint file path")
        return self


class CyberMegatronCandidateManifest(StrictModel):
    schema_version: Literal["sentinel.cyber-megatron-candidate.v1"] = (
        "sentinel.cyber-megatron-candidate.v1"
    )
    backend: Literal["megatron-swift-mcore"] = "megatron-swift-mcore"
    model: Literal["Qwen/Qwen3.5-397B-A17B"]
    model_revision: Literal["8472618112abcbd45acbcdc58436aff4233c23f7"]
    run_id: str
    run_identity_sha256: str = Field(pattern=_DIGEST)
    training_config_sha256: str = Field(pattern=_DIGEST)
    dataset_manifest_sha256: str = Field(pattern=_DIGEST)
    source_examples_sha256: str = Field(pattern=_DIGEST)
    source_manifest_sha256: str = Field(pattern=_DIGEST)
    validation_source_examples_sha256: str = Field(pattern=_DIGEST)
    validation_source_manifest_sha256: str = Field(pattern=_DIGEST)
    train_jsonl_sha256: str = Field(pattern=_DIGEST)
    validation_jsonl_sha256: str = Field(pattern=_DIGEST)
    gold_release_audit_sha256: str = Field(pattern=_DIGEST)
    plan_file_sha256: str = Field(pattern=_DIGEST)
    plan_contract_sha256: str = Field(pattern=_DIGEST)
    launch_state_sha256: str = Field(pattern=_DIGEST)
    checkpoint_relative_path: str = Field(min_length=1, max_length=4096)
    checkpoint_file_count: int = Field(gt=0)
    checkpoint_total_bytes: int = Field(gt=0)
    checkpoint_files: list[CyberMegatronCheckpointFile] = Field(min_length=1)
    checkpoint_tree_sha256: str = Field(pattern=_DIGEST)
    candidate_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def internal_contract_is_coherent(self) -> CyberMegatronCandidateManifest:
        _assert_relative_safe(self.checkpoint_relative_path, "checkpoint_relative_path")
        paths = [entry.path for entry in self.checkpoint_files]
        if paths != sorted(paths):
            raise ValueError("checkpoint file inventory must be sorted")
        if len(paths) != len(set(paths)):
            raise ValueError("checkpoint file inventory contains duplicate paths")
        if self.checkpoint_file_count != len(self.checkpoint_files):
            raise ValueError("checkpoint_file_count differs from inventory")
        expected_bytes = sum(entry.size_bytes for entry in self.checkpoint_files)
        if self.checkpoint_total_bytes != expected_bytes:
            raise ValueError("checkpoint_total_bytes differs from inventory")
        expected_tree = _sha256_canonical(
            [entry.model_dump(mode="json") for entry in self.checkpoint_files]
        )
        if self.checkpoint_tree_sha256 != expected_tree:
            raise ValueError("checkpoint_tree_sha256 does not verify")
        unsigned = self.model_dump(mode="json")
        observed = str(unsigned.pop("candidate_sha256"))
        if _sha256_canonical(unsigned) != observed:
            raise ValueError("candidate_sha256 does not verify")
        return self


class CyberMegatronCandidateVerification(StrictModel):
    valid: bool
    manifest: CyberMegatronCandidateManifest | None
    manifest_sha256: str | None = Field(default=None, pattern=_DIGEST)
    violations: list[str]


def _load_dataset_manifest(
    config: CyberMegatronSFTConfig,
    *,
    root: Path,
) -> tuple[CyberMegatronDatasetManifest, str]:
    path = _resolve_no_symlinks(
        root,
        Path(config.dataset_dir) / "manifest.json",
        "397B dataset manifest",
    )
    if not path.is_file():
        raise ValueError("397B candidate requires a regular materialized dataset manifest")
    raw = path.read_bytes()
    try:
        manifest = CyberMegatronDatasetManifest.model_validate_json(raw)
    except ValueError as exc:
        raise ValueError("397B dataset manifest cannot be parsed") from exc
    return manifest, _sha256_bytes(raw)


def _load_plan(path: Path) -> tuple[CyberMegatronPlan, bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("397B candidate requires a regular saved launch plan")
    raw = path.read_bytes()
    try:
        plan = CyberMegatronPlan.model_validate_json(raw)
    except ValueError as exc:
        raise ValueError("397B launch plan cannot be parsed") from exc
    return plan, raw


def _load_completed_launch_state(
    output_root: Path,
    *,
    config: CyberMegatronSFTConfig,
    run_identity_sha256: str,
) -> tuple[CyberMegatronLaunchState, bytes]:
    path = output_root / LAUNCH_STATE_FILENAME
    if path.is_symlink() or not path.is_file():
        raise ValueError("397B candidate requires a regular launch-state receipt")
    raw = path.read_bytes()
    try:
        state = CyberMegatronLaunchState.model_validate_json(raw)
    except ValueError as exc:
        raise ValueError("397B launch-state receipt cannot be parsed") from exc
    if state.state != "completed":
        raise ValueError("397B checkpoint cannot become a candidate before training completes")
    if state.run_identity_sha256 != run_identity_sha256:
        raise ValueError("397B launch-state receipt belongs to a different run identity")
    if state.nodes != config.topology.nodes:
        raise ValueError("397B launch-state node count differs from training config")
    return state, raw


def _checkpoint_paths(checkpoint_root: Path) -> list[tuple[str, Path]]:
    observed: list[tuple[str, Path]] = []
    for path in checkpoint_root.rglob("*"):
        relative = path.relative_to(checkpoint_root).as_posix()
        if path.is_symlink():
            raise ValueError(f"397B checkpoint contains a symlink: {relative}")
        if path.is_dir():
            continue
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise ValueError(f"397B checkpoint entry disappeared: {relative}") from exc
        if not stat.S_ISREG(mode):
            raise ValueError(f"397B checkpoint contains a non-regular entry: {relative}")
        observed.append((relative, path))
    observed.sort(key=lambda item: item[0])
    paths = [relative for relative, _path in observed]
    if len(paths) != len(set(paths)):
        raise ValueError("397B checkpoint contains duplicate relative paths")
    return observed


def _checkpoint_inventory(
    checkpoint_root: Path,
) -> tuple[list[CyberMegatronCheckpointFile], int, str]:
    if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
        raise ValueError("397B checkpoint must be a real directory, not a symlink")
    before = _checkpoint_paths(checkpoint_root)
    if not before:
        raise ValueError("397B checkpoint directory contains no files")
    entries: list[CyberMegatronCheckpointFile] = []
    for relative, path in before:
        size, digest = _sha256_stable_regular_file(path)
        entries.append(
            CyberMegatronCheckpointFile(
                path=relative,
                size_bytes=size,
                sha256=digest,
            )
        )
    after = _checkpoint_paths(checkpoint_root)
    if [item[0] for item in before] != [item[0] for item in after]:
        raise ValueError("397B checkpoint file set changed while snapshotting")
    total = sum(entry.size_bytes for entry in entries)
    if total <= 0:
        raise ValueError("397B checkpoint inventory contains no bytes")
    tree_sha = _sha256_canonical(
        [entry.model_dump(mode="json") for entry in entries]
    )
    return entries, total, tree_sha


def _require_equal(label: str, observed: object, expected: object, prefix: str) -> None:
    if observed != expected:
        raise ValueError(f"{prefix}: {label}")


def _expected_candidate(
    *,
    root: Path,
    config_path: Path,
    plan_path: Path,
    checkpoint_dir: Path,
) -> CyberMegatronCandidateManifest:
    config = load_cyber_megatron_config(config_path)
    if config.model != QWEN35_397B_MODEL or config.model_revision != QWEN35_397B_REVISION:
        raise ValueError("candidate builder accepts only the pinned Qwen3.5-397B-A17B target")

    output_root = _resolve_no_symlinks(
        root,
        config.output_dir,
        "397B training output directory",
    )
    if not output_root.is_dir():
        raise ValueError("397B candidate requires the bound training output directory")

    identity_path = _resolve_no_symlinks(
        root,
        Path(config.output_dir) / RUN_IDENTITY_FILENAME,
        "397B run identity",
    )
    if not identity_path.is_file():
        raise ValueError("397B candidate requires the bound run identity")
    identity = _load_run_identity(identity_path)
    _require_equal(
        "model",
        identity.model,
        config.model,
        "397B run identity differs from training config",
    )
    _require_equal(
        "model_revision",
        identity.model_revision,
        config.model_revision,
        "397B run identity differs from training config",
    )
    _require_equal(
        "run_id",
        identity.run_id,
        config.run_id,
        "397B run identity differs from training config",
    )
    _require_equal(
        "config_sha256",
        identity.config_sha256,
        _config_sha256(config),
        "397B run identity differs from training config",
    )
    _require_equal(
        "output_dir",
        identity.output_dir,
        config.output_dir,
        "397B run identity differs from training config",
    )

    dataset, dataset_sha = _load_dataset_manifest(config, root=root)
    _require_equal(
        "dataset_manifest_sha256",
        dataset_sha,
        identity.dataset_manifest_sha256,
        "397B dataset manifest differs from run identity",
    )
    _require_equal(
        "run_id",
        dataset.run_id,
        config.run_id,
        "397B dataset manifest differs from training config",
    )
    _require_equal(
        "model",
        dataset.model,
        config.model,
        "397B dataset manifest differs from training config",
    )
    _require_equal(
        "model_revision",
        dataset.model_revision,
        config.model_revision,
        "397B dataset manifest differs from training config",
    )
    if dataset.gold_release_audit_sha256 is None:
        raise ValueError("397B promotion candidate requires a Gold release audit binding")
    if dataset.validation_source_examples_sha256 is None:
        raise ValueError("397B promotion candidate requires explicit Gold VALIDATION examples")
    if dataset.validation_source_manifest_sha256 is None:
        raise ValueError("397B promotion candidate requires explicit Gold VALIDATION manifest")
    if dataset.source_promotion_eligible is not True:
        raise ValueError("397B TRAIN source is not promotion eligible")
    if dataset.validation_source_promotion_eligible is not True:
        raise ValueError("397B VALIDATION source is not promotion eligible")

    identity_checks = (
        ("source_examples_sha256", identity.source_examples_sha256, dataset.source_examples_sha256),
        ("source_manifest_sha256", identity.source_manifest_sha256, dataset.source_manifest_sha256),
        (
            "validation_source_examples_sha256",
            identity.validation_source_examples_sha256,
            dataset.validation_source_examples_sha256,
        ),
        (
            "validation_source_manifest_sha256",
            identity.validation_source_manifest_sha256,
            dataset.validation_source_manifest_sha256,
        ),
        ("train_jsonl_sha256", identity.train_jsonl_sha256, dataset.train_jsonl_sha256),
        (
            "validation_jsonl_sha256",
            identity.validation_jsonl_sha256,
            dataset.validation_jsonl_sha256,
        ),
    )
    for label, observed, expected in identity_checks:
        _require_equal(
            label,
            observed,
            expected,
            "397B run identity differs from dataset manifest",
        )

    plan, plan_raw = _load_plan(plan_path)
    plan_checks = (
        ("run_id", plan.run_id, config.run_id),
        ("model", plan.model, config.model),
        ("model_revision", plan.model_revision, config.model_revision),
        ("config_sha256", plan.config_sha256, identity.config_sha256),
        ("dataset_manifest_sha256", plan.dataset_manifest_sha256, dataset_sha),
        ("source_examples_sha256", plan.source_examples_sha256, dataset.source_examples_sha256),
        ("source_manifest_sha256", plan.source_manifest_sha256, dataset.source_manifest_sha256),
        (
            "validation_source_examples_sha256",
            plan.validation_source_examples_sha256,
            dataset.validation_source_examples_sha256,
        ),
        (
            "validation_source_manifest_sha256",
            plan.validation_source_manifest_sha256,
            dataset.validation_source_manifest_sha256,
        ),
        ("train_jsonl_sha256", plan.train_jsonl_sha256, dataset.train_jsonl_sha256),
        (
            "validation_jsonl_sha256",
            plan.validation_jsonl_sha256,
            dataset.validation_jsonl_sha256,
        ),
        (
            "gold_release_audit_sha256",
            plan.gold_release_audit_sha256,
            dataset.gold_release_audit_sha256,
        ),
        ("run_identity_sha256", plan.run_identity_sha256, identity.identity_sha256),
    )
    for label, observed, expected in plan_checks:
        _require_equal(
            label,
            observed,
            expected,
            "397B saved launch plan binding mismatch",
        )
    if not plan.static_ready or not plan.dataset_verified or plan.blockers:
        raise ValueError("397B saved launch plan was not statically ready")
    if plan.backend != "megatron-swift":
        raise ValueError("397B saved launch plan has the wrong backend")

    _state, state_raw = _load_completed_launch_state(
        output_root,
        config=config,
        run_identity_sha256=identity.identity_sha256,
    )

    checkpoint_root = checkpoint_dir
    try:
        checkpoint_relative = checkpoint_root.relative_to(output_root).as_posix()
    except ValueError as exc:
        raise ValueError("397B checkpoint must stay inside the bound run output") from exc
    if checkpoint_root == output_root:
        raise ValueError("397B checkpoint must be a dedicated directory inside run output")
    _assert_relative_safe(checkpoint_relative, "checkpoint_relative_path")
    files, total_bytes, tree_sha = _checkpoint_inventory(checkpoint_root)

    unsigned: dict[str, object] = {
        "schema_version": "sentinel.cyber-megatron-candidate.v1",
        "backend": "megatron-swift-mcore",
        "model": config.model,
        "model_revision": config.model_revision,
        "run_id": config.run_id,
        "run_identity_sha256": identity.identity_sha256,
        "training_config_sha256": identity.config_sha256,
        "dataset_manifest_sha256": dataset_sha,
        "source_examples_sha256": dataset.source_examples_sha256,
        "source_manifest_sha256": dataset.source_manifest_sha256,
        "validation_source_examples_sha256": dataset.validation_source_examples_sha256,
        "validation_source_manifest_sha256": dataset.validation_source_manifest_sha256,
        "train_jsonl_sha256": dataset.train_jsonl_sha256,
        "validation_jsonl_sha256": dataset.validation_jsonl_sha256,
        "gold_release_audit_sha256": dataset.gold_release_audit_sha256,
        "plan_file_sha256": _sha256_bytes(plan_raw),
        "plan_contract_sha256": _plan_sha256(plan),
        "launch_state_sha256": _sha256_bytes(state_raw),
        "checkpoint_relative_path": checkpoint_relative,
        "checkpoint_file_count": len(files),
        "checkpoint_total_bytes": total_bytes,
        "checkpoint_files": [entry.model_dump(mode="json") for entry in files],
        "checkpoint_tree_sha256": tree_sha,
    }
    return CyberMegatronCandidateManifest(
        **unsigned,
        candidate_sha256=_sha256_canonical(unsigned),
    )


def _manifest_text(manifest: CyberMegatronCandidateManifest) -> str:
    return json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def build_cyber_megatron_candidate(
    *,
    config_path: str | Path,
    plan_path: str | Path,
    checkpoint_dir: str | Path,
    output_path: str | Path,
    root: str | Path = ".",
) -> CyberMegatronCandidateManifest:
    root_path = Path(root).resolve()
    config_file = _resolve_no_symlinks(root_path, config_path, "397B training config")
    plan_file = _resolve_no_symlinks(root_path, plan_path, "397B launch plan")
    checkpoint = _resolve_no_symlinks(root_path, checkpoint_dir, "397B checkpoint")
    destination = _resolve_no_symlinks(root_path, output_path, "397B candidate output")
    if destination.exists():
        raise FileExistsError(f"397B candidate manifest already exists: {output_path}")
    manifest = _expected_candidate(
        root=root_path,
        config_path=config_file,
        plan_path=plan_file,
        checkpoint_dir=checkpoint,
    )
    atomic_write(destination, _manifest_text(manifest))
    return manifest


def verify_cyber_megatron_candidate(
    *,
    manifest_path: str | Path,
    config_path: str | Path,
    plan_path: str | Path,
    checkpoint_dir: str | Path,
    root: str | Path = ".",
) -> CyberMegatronCandidateVerification:
    root_path = Path(root).resolve()
    try:
        candidate_path = _resolve_no_symlinks(
            root_path,
            manifest_path,
            "397B candidate manifest",
        )
    except ValueError as exc:
        return CyberMegatronCandidateVerification(
            valid=False,
            manifest=None,
            manifest_sha256=None,
            violations=[str(exc)],
        )
    violations: list[str] = []
    if not candidate_path.is_file():
        return CyberMegatronCandidateVerification(
            valid=False,
            manifest=None,
            manifest_sha256=None,
            violations=["397B candidate manifest is missing"],
        )
    raw = candidate_path.read_bytes()
    try:
        observed = CyberMegatronCandidateManifest.model_validate_json(raw)
    except ValueError as exc:
        return CyberMegatronCandidateVerification(
            valid=False,
            manifest=None,
            manifest_sha256=_sha256_bytes(raw),
            violations=[f"397B candidate manifest cannot be verified: {exc}"],
        )
    if raw != _manifest_text(observed).encode("utf-8"):
        violations.append("397B candidate manifest is not canonical byte-for-byte")
    try:
        expected = _expected_candidate(
            root=root_path,
            config_path=_resolve_no_symlinks(
                root_path,
                config_path,
                "397B training config",
            ),
            plan_path=_resolve_no_symlinks(root_path, plan_path, "397B launch plan"),
            checkpoint_dir=_resolve_no_symlinks(
                root_path,
                checkpoint_dir,
                "397B checkpoint",
            ),
        )
    except (OSError, TypeError, ValueError) as exc:
        violations.append(f"397B candidate source revalidation failed: {exc}")
    else:
        if observed != expected:
            violations.append(
                "397B candidate manifest differs from freshly revalidated source artifacts"
            )
    return CyberMegatronCandidateVerification(
        valid=not violations,
        manifest=observed,
        manifest_sha256=_sha256_bytes(raw),
        violations=violations,
    )
