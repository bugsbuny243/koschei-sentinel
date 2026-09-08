from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from koschei_sentinel import drive_archive


def _artifact(tmp_path: Path) -> Path:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "weights.bin").write_bytes(b"sentinel-weights")
    nested = root / "meta"
    nested.mkdir()
    (nested / "config.json").write_text('{"model":"sentinel"}\n', encoding="utf-8")
    (root / "source-manifest.json").write_text('{"version":1}\n', encoding="utf-8")
    return root


def _manifest(tmp_path: Path) -> tuple[Path, Path, drive_archive.DriveArchiveManifest]:
    root = _artifact(tmp_path)
    manifest = drive_archive.build_drive_archive_manifest(
        root,
        artifact_id="fixture-run-1",
        artifact_kind="CHECKPOINT",
        source_manifest="source-manifest.json",
    )
    path = tmp_path / "drive-archive-manifest.json"
    drive_archive.write_drive_archive_manifest(manifest, path)
    return root, path, manifest


def _env() -> dict[str, str]:
    return {
        "KOSCHEI_RCLONE_REMOTE": "sentinel-drive",
        "KOSCHEI_DRIVE_ARCHIVE_FOLDER_ID": "1XvItGGVxGPu35wqSi80QV_SVZAS0vpeo",
        "KOSCHEI_RCLONE_CONFIG": "/tmp/sentinel-rclone.conf",
    }


def test_manifest_is_deterministic_and_local_verification_passes(tmp_path: Path) -> None:
    root, path, manifest = _manifest(tmp_path)

    rebuilt = drive_archive.build_drive_archive_manifest(
        root,
        artifact_id="fixture-run-1",
        artifact_kind="CHECKPOINT",
        source_manifest="source-manifest.json",
    )
    verification = drive_archive.verify_local_archive(manifest, root)

    assert rebuilt == manifest
    assert drive_archive.load_drive_archive_manifest(path) == manifest
    assert manifest.file_count == 3
    assert [item.path for item in manifest.files] == sorted(item.path for item in manifest.files)
    assert verification.verified is True
    assert verification.violations == []


def test_local_verification_fails_after_artifact_tamper(tmp_path: Path) -> None:
    root, _, manifest = _manifest(tmp_path)
    (root / "weights.bin").write_bytes(b"tampered")

    verification = drive_archive.verify_local_archive(manifest, root)

    assert verification.verified is False
    assert verification.violations


def test_manifest_rejects_symbolic_links(tmp_path: Path) -> None:
    root = _artifact(tmp_path)
    (root / "linked.bin").symlink_to(root / "weights.bin")

    with pytest.raises(ValueError, match="symbolic links"):
        drive_archive.build_drive_archive_manifest(
            root,
            artifact_id="fixture-run-2",
            artifact_kind="GENERIC_ARTIFACT",
        )


def test_drive_binding_fails_closed_and_rejects_remote_injection() -> None:
    with pytest.raises(ValueError, match="missing Drive transport bindings"):
        drive_archive.binding_from_env({})

    env = _env()
    env["KOSCHEI_RCLONE_REMOTE"] = "bad:remote"
    with pytest.raises(ValueError, match="unsupported characters"):
        drive_archive.binding_from_env(env)


def test_upload_is_immutable_and_remote_sha256_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_path, manifest = _manifest(tmp_path)
    commands: list[list[str]] = []
    monkeypatch.setattr(drive_archive, "_resolve_rclone", lambda executable: "/usr/bin/rclone")
    monkeypatch.setattr(drive_archive, "_run_rclone", commands.append)

    receipt = drive_archive.upload_and_verify_with_rclone(
        manifest_path,
        root,
        env=_env(),
    )

    assert receipt.artifact_digest == manifest.artifact_digest
    assert receipt.upload_verified is True
    assert receipt.verification_mode == "REMOTE_SHA256_DOWNLOAD"
    assert len(commands) == 3
    assert commands[0][1] == "copy"
    assert "--immutable" in commands[0]
    assert commands[1][1] == "copyto"
    assert "--immutable" in commands[1]
    assert commands[2][1:3] == ["checksum", "sha256"]
    assert "--download" in commands[2]
    assert _env()["KOSCHEI_DRIVE_ARCHIVE_FOLDER_ID"] not in json.dumps(
        receipt.model_dump(mode="json")
    )


def test_restore_downloads_to_empty_dir_and_rehashes_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_path, _ = _manifest(tmp_path)
    restored = tmp_path / "restored"
    monkeypatch.setattr(drive_archive, "_resolve_rclone", lambda executable: "/usr/bin/rclone")

    def fake_run(command: list[str]) -> None:
        if command[1] == "copy" and command[2].startswith("sentinel-drive:"):
            shutil.copytree(root, Path(command[3]), dirs_exist_ok=True)

    monkeypatch.setattr(drive_archive, "_run_rclone", fake_run)

    verification = drive_archive.restore_and_verify_with_rclone(
        manifest_path,
        restored,
        env=_env(),
    )

    assert verification.verified is True
    assert (restored / "weights.bin").read_bytes() == b"sentinel-weights"


def test_restore_refuses_nonempty_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, manifest_path, _ = _manifest(tmp_path)
    restored = tmp_path / "restored"
    restored.mkdir()
    (restored / "keep.txt").write_text("do not overwrite", encoding="utf-8")
    monkeypatch.setattr(drive_archive, "_resolve_rclone", lambda executable: "/usr/bin/rclone")

    with pytest.raises(FileExistsError, match="must be empty"):
        drive_archive.restore_and_verify_with_rclone(
            manifest_path,
            restored,
            env=_env(),
        )
