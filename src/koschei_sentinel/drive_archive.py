from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import atomic_write, canonical_json

_DIGEST_RE = r"^[a-f0-9]{64}$"
_ARTIFACT_ID_RE = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_REMOTE_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_MAX_FILES = 10_000
_DRIVE_FOLDER_ENV = "KOSCHEI_DRIVE_ARCHIVE_FOLDER_ID"
_RCLONE_REMOTE_ENV = "KOSCHEI_RCLONE_REMOTE"
_RCLONE_CONFIG_ENV = "KOSCHEI_RCLONE_CONFIG"


class DriveArchiveFile(StrictModel):
    path: str = Field(min_length=1, max_length=1024)
    sha256: str = Field(pattern=_DIGEST_RE)
    bytes: int = Field(ge=0)


class DriveArchiveManifest(StrictModel):
    schema_version: Literal["sentinel.drive-archive-manifest.v1"] = (
        "sentinel.drive-archive-manifest.v1"
    )
    artifact_id: str = Field(pattern=_ARTIFACT_ID_RE)
    artifact_kind: Literal[
        "CHECKPOINT",
        "DATASET_RELEASE",
        "MODEL_ADAPTER",
        "EVALUATION",
        "GENERIC_ARTIFACT",
    ]
    archive_folder_env: Literal["KOSCHEI_DRIVE_ARCHIVE_FOLDER_ID"] = _DRIVE_FOLDER_ENV
    source_manifest_sha256: str | None = Field(default=None, pattern=_DIGEST_RE)
    files: list[DriveArchiveFile] = Field(min_length=1, max_length=_MAX_FILES)
    file_count: int = Field(ge=1, le=_MAX_FILES)
    total_bytes: int = Field(ge=0)
    artifact_digest: str = Field(pattern=_DIGEST_RE)

    @model_validator(mode="after")
    def counts_match_files(self) -> DriveArchiveManifest:
        if self.file_count != len(self.files):
            raise ValueError("file_count does not match files")
        if self.total_bytes != sum(item.bytes for item in self.files):
            raise ValueError("total_bytes does not match files")
        if [item.path for item in self.files] != sorted(item.path for item in self.files):
            raise ValueError("archive files must be sorted by path")
        if len({item.path for item in self.files}) != len(self.files):
            raise ValueError("archive file paths must be unique")
        return self


class DriveArchiveTransportReceipt(StrictModel):
    schema_version: Literal["sentinel.drive-archive-transport-receipt.v1"] = (
        "sentinel.drive-archive-transport-receipt.v1"
    )
    artifact_id: str = Field(pattern=_ARTIFACT_ID_RE)
    artifact_digest: str = Field(pattern=_DIGEST_RE)
    manifest_sha256: str = Field(pattern=_DIGEST_RE)
    transport: Literal["RCLONE_GOOGLE_DRIVE"] = "RCLONE_GOOGLE_DRIVE"
    remote_name: str = Field(min_length=1, max_length=64)
    archive_folder_binding_sha256: str = Field(pattern=_DIGEST_RE)
    remote_base_path: str = Field(min_length=1, max_length=512)
    upload_verified: Literal[True] = True
    verification_mode: Literal["REMOTE_SHA256_DOWNLOAD"] = "REMOTE_SHA256_DOWNLOAD"
    verification_command_digest: str = Field(pattern=_DIGEST_RE)


class DriveArchiveVerification(StrictModel):
    schema_version: Literal["sentinel.drive-archive-verification.v1"] = (
        "sentinel.drive-archive-verification.v1"
    )
    artifact_id: str
    artifact_digest: str = Field(pattern=_DIGEST_RE)
    verified: bool
    observed_file_count: int = Field(ge=0)
    observed_total_bytes: int = Field(ge=0)
    violations: list[str] = Field(default_factory=list, max_length=256)


class RcloneDriveBinding(StrictModel):
    remote_name: str
    archive_folder_id: str = Field(min_length=8, max_length=256)
    config_path: str | None = Field(default=None, min_length=1, max_length=1024)

    @model_validator(mode="after")
    def binding_is_safe(self) -> RcloneDriveBinding:
        if not _REMOTE_RE.fullmatch(self.remote_name):
            raise ValueError("rclone remote name contains unsupported characters")
        if any(char.isspace() for char in self.archive_folder_id):
            raise ValueError("Drive archive folder id may not contain whitespace")
        if self.config_path is not None and "\x00" in self.config_path:
            raise ValueError("rclone config path contains NUL")
        return self


def build_drive_archive_manifest(
    artifact_root: str | Path,
    *,
    artifact_id: str,
    artifact_kind: str,
    source_manifest: str | Path | None = None,
) -> DriveArchiveManifest:
    root = Path(artifact_root).resolve()
    if not root.is_dir():
        raise ValueError("artifact root must be an existing directory")

    files = _artifact_files(root)
    if not files:
        raise ValueError("artifact root contains no files")
    source_manifest_sha256 = None
    if source_manifest is not None:
        source_path = _resolve_under(root, source_manifest)
        if not source_path.is_file():
            raise ValueError("source manifest is missing from artifact root")
        source_manifest_sha256 = _hash_file(source_path)

    payload = {
        "artifact_id": artifact_id,
        "artifact_kind": artifact_kind,
        "archive_folder_env": _DRIVE_FOLDER_ENV,
        "source_manifest_sha256": source_manifest_sha256,
        "files": [item.model_dump(mode="json") for item in files],
    }
    return DriveArchiveManifest(
        artifact_id=artifact_id,
        artifact_kind=artifact_kind,
        source_manifest_sha256=source_manifest_sha256,
        files=files,
        file_count=len(files),
        total_bytes=sum(item.bytes for item in files),
        artifact_digest=_digest(payload),
    )


def write_drive_archive_manifest(
    manifest: DriveArchiveManifest,
    destination: str | Path,
) -> Path:
    path = Path(destination)
    atomic_write(
        path,
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )
    return path


def load_drive_archive_manifest(path: str | Path) -> DriveArchiveManifest:
    try:
        return DriveArchiveManifest.model_validate_json(Path(path).read_bytes())
    except ValueError as exc:
        raise ValueError("invalid Drive archive manifest") from exc


def verify_local_archive(
    manifest: DriveArchiveManifest,
    artifact_root: str | Path,
) -> DriveArchiveVerification:
    violations: list[str] = []
    try:
        observed = build_drive_archive_manifest(
            artifact_root,
            artifact_id=manifest.artifact_id,
            artifact_kind=manifest.artifact_kind,
            source_manifest=_find_source_manifest(manifest, artifact_root),
        )
    except (OSError, ValueError) as exc:
        return DriveArchiveVerification(
            artifact_id=manifest.artifact_id,
            artifact_digest=manifest.artifact_digest,
            verified=False,
            observed_file_count=0,
            observed_total_bytes=0,
            violations=[str(exc)],
        )

    if observed.files != manifest.files:
        violations.append("archive file hashes, sizes, or paths differ from manifest")
    if observed.source_manifest_sha256 != manifest.source_manifest_sha256:
        violations.append("source manifest digest differs from archive manifest")
    if observed.artifact_digest != manifest.artifact_digest:
        violations.append("artifact digest differs from archive manifest")
    return DriveArchiveVerification(
        artifact_id=manifest.artifact_id,
        artifact_digest=manifest.artifact_digest,
        verified=not violations,
        observed_file_count=observed.file_count,
        observed_total_bytes=observed.total_bytes,
        violations=violations,
    )


def binding_from_env(env: Mapping[str, str] | None = None) -> RcloneDriveBinding:
    environ = os.environ if env is None else env
    remote = environ.get(_RCLONE_REMOTE_ENV, "").strip()
    folder = environ.get(_DRIVE_FOLDER_ENV, "").strip()
    config = environ.get(_RCLONE_CONFIG_ENV, "").strip() or None
    missing = [
        name
        for name, value in (
            (_RCLONE_REMOTE_ENV, remote),
            (_DRIVE_FOLDER_ENV, folder),
        )
        if not value
    ]
    if missing:
        raise ValueError("missing Drive transport bindings: " + ", ".join(missing))
    return RcloneDriveBinding(
        remote_name=remote,
        archive_folder_id=folder,
        config_path=config,
    )


def archive_remote_base_path(manifest: DriveArchiveManifest) -> str:
    return f"sentinel-archive/{manifest.artifact_id}/{manifest.artifact_digest}"


def upload_and_verify_with_rclone(
    manifest_path: str | Path,
    artifact_root: str | Path,
    *,
    env: Mapping[str, str] | None = None,
    executable: str = "rclone",
) -> DriveArchiveTransportReceipt:
    manifest = load_drive_archive_manifest(manifest_path)
    local = verify_local_archive(manifest, artifact_root)
    if not local.verified:
        raise ValueError("local artifact does not match Drive archive manifest")

    binding = binding_from_env(env)
    binary = _resolve_rclone(executable)
    remote_base = archive_remote_base_path(manifest)
    remote_data = f"{binding.remote_name}:{remote_base}/data"
    remote_manifest = f"{binding.remote_name}:{remote_base}/archive-manifest.json"
    tail = _rclone_tail_args(binding)

    _run_rclone(
        [
            binary,
            "copy",
            str(Path(artifact_root).resolve()),
            remote_data,
            "--checksum",
            "--immutable",
            *tail,
        ]
    )
    _run_rclone(
        [
            binary,
            "copyto",
            str(Path(manifest_path).resolve()),
            remote_manifest,
            "--checksum",
            "--immutable",
            *tail,
        ]
    )

    checksum_path = _write_sha256_sumfile(manifest, Path(manifest_path))
    try:
        verification_command = [
            binary,
            "checksum",
            "sha256",
            str(checksum_path),
            f"{binding.remote_name}:{remote_base}",
            "--download",
            *tail,
        ]
        _run_rclone(verification_command)
    finally:
        checksum_path.unlink(missing_ok=True)

    return DriveArchiveTransportReceipt(
        artifact_id=manifest.artifact_id,
        artifact_digest=manifest.artifact_digest,
        manifest_sha256=_hash_file(Path(manifest_path)),
        remote_name=binding.remote_name,
        archive_folder_binding_sha256=_text_digest(binding.archive_folder_id),
        remote_base_path=remote_base,
        verification_command_digest=_command_digest(verification_command),
    )


def restore_and_verify_with_rclone(
    manifest_path: str | Path,
    destination: str | Path,
    *,
    env: Mapping[str, str] | None = None,
    executable: str = "rclone",
) -> DriveArchiveVerification:
    manifest = load_drive_archive_manifest(manifest_path)
    binding = binding_from_env(env)
    binary = _resolve_rclone(executable)
    remote = f"{binding.remote_name}:{archive_remote_base_path(manifest)}/data"
    destination_path = Path(destination).resolve()
    if destination_path.exists() and not destination_path.is_dir():
        raise ValueError("restore destination must be a directory")
    if destination_path.exists() and any(destination_path.iterdir()):
        raise FileExistsError("restore destination must be empty")
    destination_path.mkdir(parents=True, exist_ok=True)

    _run_rclone(
        [
            binary,
            "copy",
            remote,
            str(destination_path),
            "--checksum",
            "--immutable",
            *_rclone_tail_args(binding),
        ]
    )
    return verify_local_archive(manifest, destination_path)


def _artifact_files(root: Path) -> list[DriveArchiveFile]:
    output: list[DriveArchiveFile] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("archive may not contain symbolic links")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("archive contains an unsupported filesystem entry")
        relative = path.relative_to(root).as_posix()
        _validate_relative(relative)
        output.append(
            DriveArchiveFile(
                path=relative,
                sha256=_hash_file(path),
                bytes=path.stat().st_size,
            )
        )
        if len(output) > _MAX_FILES:
            raise ValueError("archive contains too many files")
    return output


def _find_source_manifest(
    manifest: DriveArchiveManifest,
    artifact_root: str | Path,
) -> str | None:
    if manifest.source_manifest_sha256 is None:
        return None
    root = Path(artifact_root).resolve()
    matches = [
        item.path
        for item in manifest.files
        if (root / item.path).is_file()
        and _hash_file(root / item.path) == manifest.source_manifest_sha256
    ]
    if len(matches) != 1:
        raise ValueError("source manifest cannot be uniquely resolved from archive")
    return matches[0]


def _resolve_under(root: Path, value: str | Path) -> Path:
    relative = PurePosixPath(str(value)).as_posix()
    _validate_relative(relative)
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("archive path escapes artifact root")
    return candidate


def _validate_relative(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or value.startswith("~")
        or "\\" in value
    ):
        raise ValueError("archive path must stay within artifact root")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _command_digest(command: list[str]) -> str:
    scrubbed: list[str] = []
    redact_next = False
    for item in command:
        if redact_next:
            scrubbed.append("<redacted>")
            redact_next = False
            continue
        scrubbed.append(item)
        if item in {"--drive-root-folder-id", "--config"}:
            redact_next = True
    return _digest(scrubbed)


def _resolve_rclone(executable: str) -> str:
    if executable != "rclone":
        raise ValueError("Drive transport executable must be rclone")
    found = shutil.which(executable)
    if found is None:
        raise RuntimeError("rclone executable is not available")
    return found


def _rclone_tail_args(binding: RcloneDriveBinding) -> list[str]:
    args = ["--drive-root-folder-id", binding.archive_folder_id]
    if binding.config_path is not None:
        args.extend(["--config", binding.config_path])
    return args


def _run_rclone(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Drive archive transport failed with exit code {completed.returncode}")


def _write_sha256_sumfile(
    manifest: DriveArchiveManifest,
    manifest_path: Path,
) -> Path:
    descriptor, name = tempfile.mkstemp(prefix="sentinel-drive-", suffix=".sha256")
    path = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for item in manifest.files:
                handle.write(f"{item.sha256}  data/{item.path}\n")
            handle.write(f"{_hash_file(manifest_path)}  archive-manifest.json\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path
