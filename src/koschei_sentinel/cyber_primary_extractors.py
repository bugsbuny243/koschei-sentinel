from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from koschei_sentinel.cyber_corpus_catalog import CyberArtifact, LicenseStatus
from koschei_sentinel.cyber_source_extractors import ExtractedArtifact, snapshot_sha256

_ATTACK_LICENSE = "https://attack.mitre.org/resources/terms-of-use/"
_KUBERNETES_LICENSE = "https://github.com/kubernetes/website/blob/2a031a5f0a9382d26c25cb1e58016001a21ce7b6/LICENSE"
_YARA_LICENSE = "https://github.com/VirusTotal/yara/blob/604822da04103d13812dbcb08f4d7d42b61f94a8/COPYING"

_ATTACK_TYPES = {
    "attack-pattern",
    "campaign",
    "course-of-action",
    "intrusion-set",
    "malware",
    "relationship",
    "tool",
}

_KUBE_PATH_MARKERS = (
    "/security/",
    "/authentication/",
    "/authorization/",
    "/admission-controllers/",
    "/access-authn-authz/",
    "/securing-a-cluster/",
    "/pod-security-",
    "/network-policies/",
    "/service-accounts/",
    "/secrets/",
)

_KUBE_CONTENT_MARKERS = (
    "security",
    "authentication",
    "authorization",
    "rbac",
    "admission",
    "pod security",
    "network policy",
    "service account",
    "secret",
    "certificate",
)

_YARA_DOC_SUFFIXES = {".md", ".rst", ".txt"}
_YARA_SOURCE_SUFFIXES = {".c", ".h"}
_MAX_TEXT_BYTES = 512 * 1024


def _normalise_id(source_id: str, external_id: str) -> str:
    clean = "".join(
        char.lower() if char.isalnum() or char in ".:_-" else "-"
        for char in external_id
    ).strip("-.")
    return f"{source_id}:{clean}"


def _make_artifact(
    *,
    source_id: str,
    external_id: str,
    locator: str,
    source_revision: str,
    source_snapshot_sha256: str,
    text: str,
    license_reference: str,
    benchmark_overlap_risk: str,
) -> CyberArtifact:
    return CyberArtifact(
        artifact_id=_normalise_id(source_id, external_id),
        source_id=source_id,
        locator=locator,
        source_revision=source_revision,
        source_snapshot_sha256=source_snapshot_sha256,
        content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        license_status=LicenseStatus.ALLOW_WITH_ATTRIBUTION,
        license_reference=license_reference,
        inherited_source_license=True,
        training_authorization=True,
        eval_exclusion=True,
        benchmark_overlap_risk=benchmark_overlap_risk,
    )


def _attack_objects(payload: object) -> Iterable[dict[str, object]]:
    if isinstance(payload, dict) and isinstance(payload.get("objects"), list):
        for item in payload["objects"]:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(payload, dict) and isinstance(payload.get("type"), str):
        yield payload


def _attack_text(obj: dict[str, object]) -> str:
    object_type = str(obj.get("type", "unknown"))
    name = obj.get("name") if isinstance(obj.get("name"), str) else ""
    description = obj.get("description") if isinstance(obj.get("description"), str) else ""
    parts = [f"MITRE ATT&CK {object_type}"]
    if name:
        parts.append(f"Name: {name}")
    if description:
        parts.append(description.strip())
    platforms = obj.get("x_mitre_platforms")
    if isinstance(platforms, list) and platforms:
        parts.append("Platforms: " + ", ".join(str(item) for item in platforms))
    permissions = obj.get("x_mitre_permissions_required")
    if isinstance(permissions, list) and permissions:
        parts.append("Permissions required: " + ", ".join(str(item) for item in permissions))
    if object_type == "relationship":
        source_ref = obj.get("source_ref")
        target_ref = obj.get("target_ref")
        relationship_type = obj.get("relationship_type")
        parts.append(
            f"Relationship: {source_ref} --{relationship_type}--> {target_ref}"
        )
    return "\n\n".join(part for part in parts if part).strip()


def extract_attack_stix_snapshot(
    root: str | Path,
    *,
    source_id: str,
    source_revision: str,
    snapshot_digest: str | None = None,
) -> list[ExtractedArtifact]:
    root_path = Path(root)
    snapshot_digest = snapshot_digest or snapshot_sha256(root_path)
    files = [root_path] if root_path.is_file() else sorted(root_path.rglob("*.json"))
    output: list[ExtractedArtifact] = []
    seen_external_ids: set[str] = set()

    for file_path in files:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        for obj in _attack_objects(payload):
            object_type = obj.get("type")
            external_id = obj.get("id")
            if object_type not in _ATTACK_TYPES or not isinstance(external_id, str):
                continue
            if obj.get("revoked") is True or obj.get("x_mitre_deprecated") is True:
                continue
            if external_id in seen_external_ids:
                continue
            text = _attack_text(obj)
            if len(text) < 32:
                continue
            seen_external_ids.add(external_id)
            locator = (
                file_path.name
                if root_path.is_file()
                else file_path.relative_to(root_path).as_posix()
            )
            output.append(
                ExtractedArtifact(
                    artifact=_make_artifact(
                        source_id=source_id,
                        external_id=external_id,
                        locator=f"{locator}#{external_id}",
                        source_revision=source_revision,
                        source_snapshot_sha256=snapshot_digest,
                        text=text,
                        license_reference=_ATTACK_LICENSE,
                        benchmark_overlap_risk="MEDIUM",
                    ),
                    text=text,
                    metadata={
                        "provider": "MITRE_ATTACK",
                        "external_id": external_id,
                        "stix_type": object_type,
                    },
                )
            )
    if not output:
        raise ValueError("ATT&CK snapshot contains no approved STIX knowledge objects")
    return output


def _kubernetes_security_path(path: Path, root: Path) -> bool:
    relative = "/" + path.relative_to(root).as_posix().lower()
    if not relative.startswith("/content/en/docs/"):
        return False
    return any(marker in relative for marker in _KUBE_PATH_MARKERS)


def _kubernetes_security_text(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _KUBE_CONTENT_MARKERS)


def extract_kubernetes_security_snapshot(
    root: str | Path,
    *,
    source_id: str,
    source_revision: str,
    snapshot_digest: str | None = None,
) -> list[ExtractedArtifact]:
    root_path = Path(root)
    snapshot_digest = snapshot_digest or snapshot_sha256(root_path)
    output: list[ExtractedArtifact] = []
    for path in sorted(root_path.rglob("*.md")):
        if not _kubernetes_security_path(path, root_path):
            continue
        payload = path.read_bytes()
        if not payload or len(payload) > _MAX_TEXT_BYTES:
            continue
        text = payload.decode("utf-8")
        if not _kubernetes_security_text(text):
            continue
        locator = path.relative_to(root_path).as_posix()
        output.append(
            ExtractedArtifact(
                artifact=_make_artifact(
                    source_id=source_id,
                    external_id=locator,
                    locator=locator,
                    source_revision=source_revision,
                    source_snapshot_sha256=snapshot_digest,
                    text=text,
                    license_reference=_KUBERNETES_LICENSE,
                    benchmark_overlap_risk="LOW",
                ),
                text=text,
                metadata={"provider": "KUBERNETES", "path": locator},
            )
        )
    if not output:
        raise ValueError("Kubernetes snapshot contains no in-scope English security docs")
    return output


def _yara_in_scope(path: Path, root: Path) -> bool:
    relative = path.relative_to(root).as_posix()
    lowered = relative.lower()
    if any(part in lowered for part in ("third_party/", "vendor/", "build/", ".git/")):
        return False
    if lowered.startswith("docs/") and path.suffix.lower() in _YARA_DOC_SUFFIXES:
        return True
    if lowered == "readme.md":
        return True
    if lowered.startswith("libyara/") and path.suffix.lower() in _YARA_SOURCE_SUFFIXES:
        return True
    return False


def extract_yara_snapshot(
    root: str | Path,
    *,
    source_id: str,
    source_revision: str,
    snapshot_digest: str | None = None,
) -> list[ExtractedArtifact]:
    root_path = Path(root)
    snapshot_digest = snapshot_digest or snapshot_sha256(root_path)
    output: list[ExtractedArtifact] = []
    for path in sorted(item for item in root_path.rglob("*") if item.is_file()):
        if not _yara_in_scope(path, root_path):
            continue
        payload = path.read_bytes()
        if not payload or len(payload) > _MAX_TEXT_BYTES or b"\x00" in payload:
            continue
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if len(text.strip()) < 32:
            continue
        locator = path.relative_to(root_path).as_posix()
        output.append(
            ExtractedArtifact(
                artifact=_make_artifact(
                    source_id=source_id,
                    external_id=locator,
                    locator=locator,
                    source_revision=source_revision,
                    source_snapshot_sha256=snapshot_digest,
                    text=text,
                    license_reference=_YARA_LICENSE,
                    benchmark_overlap_risk="LOW",
                ),
                text=text,
                metadata={
                    "provider": "YARA",
                    "path": locator,
                    "kind": "ENGINE_SOURCE" if locator.startswith("libyara/") else "DOCUMENTATION",
                },
            )
        )
    if not output:
        raise ValueError("YARA snapshot contains no in-scope first-party docs or engine source")
    return output
