from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from koschei_sentinel.cyber_sft_export_verify import verify_cyber_sft_export


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_tree_sha256(root: Path) -> str:
    if root.is_symlink():
        raise ValueError("candidate export directory must not be a symlink")
    if not root.is_dir():
        raise ValueError("candidate export directory is missing")

    files: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"candidate export artifact must not be a symlink: {relative}")
        if path.is_file():
            files.append(path)

    if not files:
        raise ValueError("candidate export directory contains no files")

    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        file_sha = _file_sha256(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _require_valid_candidate(path: Path, *, label: str) -> None:
    report = verify_cyber_sft_export(path)
    if not report.valid:
        detail = "; ".join(report.violations[:5])
        raise ValueError(
            f"Cyber SFT candidate {label} failed fresh verification"
            + (f": {detail}" if detail else "")
        )


def snapshot_verified_cyber_sft_export(
    candidate_export: str | Path,
    destination: str | Path,
) -> Path:
    source = Path(candidate_export)
    snapshot = Path(destination)
    if snapshot.exists():
        raise FileExistsError(f"Cyber SFT candidate snapshot already exists: {snapshot}")
    try:
        source_sha_before = _candidate_tree_sha256(source)
        _require_valid_candidate(source, label="source")
        source_sha_after = _candidate_tree_sha256(source)
        if source_sha_before != source_sha_after:
            raise ValueError(
                "Cyber SFT candidate source changed while it was being verified"
            )

        shutil.copytree(source, snapshot, symlinks=True)
        _require_valid_candidate(snapshot, label="snapshot")
        snapshot_sha = _candidate_tree_sha256(snapshot)
        if snapshot_sha != source_sha_after:
            raise ValueError(
                "Cyber SFT candidate snapshot bytes differ from verified source"
            )
        return snapshot
    except (OSError, TypeError, ValueError):
        shutil.rmtree(snapshot, ignore_errors=True)
        raise
