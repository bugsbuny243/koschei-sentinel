"""Offline Lang + Sentinel bundle integrity check; no release or execution authority.

This transport contract is also implemented in Sentinel. It uses only the
standard library and never imports either product's runtime or model code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path

SCHEMA = "koschei.commercial-bundle.v1"
COMPONENTS = frozenset({"koschei-lang", "koschei-sentinel"})
MAX_MANIFEST_BYTES = 65536
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,95}\Z")
_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9.-]+)?\Z")
_ARTIFACT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")


def _fields(value: object, required: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{label}: exact contract fields required")
    return value


def _text(value: object, pattern: re.Pattern, label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{label}: invalid format")
    return value


def validate_bundle(value: object) -> dict:
    """Validate packaging metadata without asserting licensing or promotion."""
    bundle = _fields(value, {
        "schema_version", "bundle_id", "bundle_version", "components",
    }, "bundle")
    if bundle["schema_version"] != SCHEMA:
        raise ValueError("unsupported commercial bundle schema")
    _text(bundle["bundle_id"], _IDENTIFIER, "bundle_id")
    _text(bundle["bundle_version"], _VERSION, "bundle_version")
    components = bundle["components"]
    if not isinstance(components, list) or len(components) != 2:
        raise ValueError("every bundle must contain both Lang and Sentinel")
    seen: set[str] = set()
    artifacts: set[str] = set()
    for item in components:
        item = _fields(item, {
            "product", "version", "artifact", "sha256",
        }, "component")
        product = item["product"]
        if not isinstance(product, str) or product not in COMPONENTS or product in seen:
            raise ValueError("bundle products must be exactly Lang and Sentinel")
        seen.add(product)
        _text(item["version"], _VERSION, "component version")
        artifact = _text(item["artifact"], _ARTIFACT, "artifact basename")
        if artifact.casefold() in artifacts:
            raise ValueError("component artifacts must be distinct")
        artifacts.add(artifact.casefold())
        _text(item["sha256"], _SHA256, "artifact sha256")
    if seen != COMPONENTS:
        raise ValueError("every bundle must contain both Lang and Sentinel")
    return bundle


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    value: dict = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON field")
        value[key] = item
    return value


def _reject_constant(_value: str) -> None:
    raise ValueError("non-finite JSON value")


def load_bundle(path: str | Path) -> dict:
    with Path(path).open("rb") as stream:
        payload = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(payload) > MAX_MANIFEST_BYTES:
        raise ValueError("commercial bundle manifest is too large")
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid commercial bundle JSON") from exc
    return validate_bundle(value)


def _artifact_digest(path: Path) -> str:
    # Reject symlinks, directories and special files, including a FIFO swapped
    # in between lstat and open. Verification never executes an artifact.
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("bundle artifact must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or (before.st_dev, before.st_ino) != (
            opened.st_dev, opened.st_ino,
        ):
            raise ValueError("bundle artifact changed before verification")
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.fstat(stream.fileno())
        if (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns,
        ):
            raise ValueError("bundle artifact changed during verification")
    return digest


def check_bundle(value: object, artifacts_dir: str | Path | None = None) -> dict:
    bundle = validate_bundle(value)
    verified = []
    if artifacts_dir is not None:
        root = Path(artifacts_dir).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("artifacts directory required")
        for item in bundle["components"]:
            if _artifact_digest(root / item["artifact"]) != item["sha256"]:
                raise ValueError(f"artifact digest mismatch: {item['product']}")
            verified.append(item["product"])
    canonical = dict(bundle, components=sorted(
        bundle["components"], key=lambda item: item["product"],
    ))
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": "koschei.commercial-bundle-check.v1",
        "bundle_id": bundle["bundle_id"],
        "bundle_version": bundle["bundle_version"],
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "products": sorted(COMPONENTS),
        "status": "integrity_verified" if artifacts_dir is not None else "manifest_valid",
        "verified_artifacts": sorted(verified),
        "release_approved": False,
        "execution_authorized": False,
        "limits": [
            "Integrity does not establish publisher identity, licensing or release trust.",
            "Lang release verification and Sentinel model promotion remain independent gates.",
            "No artifact is downloaded, installed, executed or promoted by this check.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="Lang + Sentinel bundle manifest JSON")
    parser.add_argument("--artifacts-dir", help="Verify both local component artifact digests")
    args = parser.parse_args(argv)
    try:
        report = check_bundle(load_bundle(args.manifest), args.artifacts_dir)
    except (ValueError, OSError) as exc:
        message = str(exc) if isinstance(exc, ValueError) else "bundle files unavailable"
        print(json.dumps({"status": "rejected", "error": message}), file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
