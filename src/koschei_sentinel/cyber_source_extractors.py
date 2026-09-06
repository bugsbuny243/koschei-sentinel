from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from koschei_sentinel.cyber_corpus_catalog import CyberArtifact, LicenseStatus

_RUSTSEC_LICENSE_REFERENCE = (
    "https://github.com/RustSec/advisory-db/blob/main/README.md#license"
)
_GHSA_LICENSE_REFERENCE = (
    "https://github.com/github/advisory-database/blob/main/LICENSE.md"
)
_NVD_TERMS_REFERENCE = "https://nvd.nist.gov/developers/terms-of-use"


@dataclass(frozen=True)
class ExtractedArtifact:
    artifact: CyberArtifact
    text: str
    metadata: dict[str, object]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def snapshot_sha256(path: str | Path) -> str:
    source = Path(path)
    if source.is_file():
        return sha256_bytes(source.read_bytes())
    digest = hashlib.sha256()
    for item in sorted(p for p in source.rglob("*") if p.is_file()):
        relative = item.relative_to(source).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        payload = item.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _normalise_artifact_id(source_id: str, external_id: str) -> str:
    clean = "".join(
        character.lower() if character.isalnum() or character in ".:_-" else "-"
        for character in external_id
    ).strip("-.")
    return f"{source_id}:{clean}"


def _artifact(
    *,
    source_id: str,
    external_id: str,
    locator: str,
    source_revision: str,
    source_snapshot_sha256: str,
    text: str,
    license_status: LicenseStatus,
    license_reference: str | None,
    inherited_source_license: bool,
    training_authorization: bool,
    benchmark_overlap_risk: Literal["NONE", "LOW", "MEDIUM", "HIGH", "UNKNOWN"],
) -> CyberArtifact:
    return CyberArtifact(
        artifact_id=_normalise_artifact_id(source_id, external_id),
        source_id=source_id,
        locator=locator,
        source_revision=source_revision,
        source_snapshot_sha256=source_snapshot_sha256,
        content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        license_status=license_status,
        license_reference=license_reference,
        inherited_source_license=inherited_source_license,
        training_authorization=training_authorization,
        eval_exclusion=True,
        benchmark_overlap_risk=benchmark_overlap_risk,
    )


def parse_rustsec_advisory(path: str | Path) -> tuple[dict[str, object], str]:
    raw = Path(path).read_text(encoding="utf-8")
    lines = raw.splitlines()
    if not lines or lines[0].strip().lower() != "```toml":
        raise ValueError(f"RustSec advisory lacks TOML front matter: {path}")
    try:
        end = next(index for index, line in enumerate(lines[1:], 1) if line.strip() == "```")
    except StopIteration as exc:
        raise ValueError(f"RustSec advisory has unterminated TOML front matter: {path}") from exc
    metadata = tomllib.loads("\n".join(lines[1:end]))
    body = "\n".join(lines[end + 1 :]).strip()
    if not body:
        raise ValueError(f"RustSec advisory body is empty: {path}")
    advisory = metadata.get("advisory")
    if not isinstance(advisory, dict):
        raise ValueError(f"RustSec advisory table missing: {path}")
    if not isinstance(advisory.get("id"), str):
        raise ValueError(f"RustSec advisory id missing: {path}")
    return metadata, body


def resolve_rustsec_license(advisory: dict[str, object]) -> tuple[LicenseStatus, str, bool]:
    raw_license = advisory.get("license", "CC0-1.0")
    if raw_license == "CC0-1.0":
        return LicenseStatus.ALLOW_TRAINING, _RUSTSEC_LICENSE_REFERENCE, True
    if raw_license == "CC-BY-4.0":
        source_url = advisory.get("url")
        if not isinstance(source_url, str) or not source_url:
            return LicenseStatus.REVIEW_REQUIRED, _RUSTSEC_LICENSE_REFERENCE, False
        return LicenseStatus.ALLOW_WITH_ATTRIBUTION, source_url, True
    return LicenseStatus.REVIEW_REQUIRED, _RUSTSEC_LICENSE_REFERENCE, False


def extract_rustsec_snapshot(
    root: str | Path,
    *,
    source_id: str,
    source_revision: str,
    snapshot_digest: str | None = None,
) -> list[ExtractedArtifact]:
    root_path = Path(root)
    snapshot_digest = snapshot_digest or snapshot_sha256(root_path)
    output: list[ExtractedArtifact] = []
    for path in sorted(root_path.rglob("RUSTSEC-*.md")):
        metadata, body = parse_rustsec_advisory(path)
        advisory = metadata["advisory"]
        assert isinstance(advisory, dict)
        external_id = str(advisory["id"])
        status, reference, authorized = resolve_rustsec_license(advisory)
        package = str(advisory.get("package", "unknown"))
        text = f"RustSec advisory {external_id}\nPackage: {package}\n\n{body}"
        output.append(
            ExtractedArtifact(
                artifact=_artifact(
                    source_id=source_id,
                    external_id=external_id,
                    locator=path.relative_to(root_path).as_posix(),
                    source_revision=source_revision,
                    source_snapshot_sha256=snapshot_digest,
                    text=text,
                    license_status=status,
                    license_reference=reference,
                    inherited_source_license=False,
                    training_authorization=authorized,
                    benchmark_overlap_risk="LOW",
                ),
                text=text,
                metadata={
                    "provider": "RUSTSEC",
                    "external_id": external_id,
                    "package": package,
                    "declared_license": advisory.get("license", "CC0-1.0"),
                    "attribution_url": advisory.get("url"),
                },
            )
        )
    if not output:
        raise ValueError("RustSec snapshot contains no RUSTSEC-*.md advisories")
    return output


def _read_osv_records(path: Path) -> Iterable[tuple[str, dict[str, object]]]:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for index, record in enumerate(payload):
                if not isinstance(record, dict):
                    raise ValueError(f"OSV list row {index} is not an object")
                yield f"{path.name}#{index}", record
        elif isinstance(payload, dict):
            yield path.name, payload
        else:
            raise ValueError("OSV JSON must be an object or list")
        return
    for item in sorted(path.rglob("*.json")):
        payload = json.loads(item.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        yield item.relative_to(path).as_posix(), payload


def _explicit_osv_license(record: dict[str, object]) -> tuple[LicenseStatus, str | None, bool]:
    candidates: list[object] = [record.get("license")]
    database_specific = record.get("database_specific")
    if isinstance(database_specific, dict):
        candidates.extend([database_specific.get("license"), database_specific.get("license_url")])
    value = next((item for item in candidates if isinstance(item, str) and item), None)
    if value == "CC0-1.0":
        return LicenseStatus.ALLOW_TRAINING, value, True
    if value in {"CC-BY-4.0", "CC BY 4.0"}:
        reference = None
        if isinstance(database_specific, dict):
            raw_ref = database_specific.get("license_url")
            if isinstance(raw_ref, str) and raw_ref:
                reference = raw_ref
        return LicenseStatus.ALLOW_WITH_ATTRIBUTION, reference, bool(reference)
    return LicenseStatus.REVIEW_REQUIRED, None, False


def extract_osv_snapshot(
    path: str | Path,
    *,
    source_id: str,
    source_revision: str,
    snapshot_digest: str | None = None,
) -> list[ExtractedArtifact]:
    source = Path(path)
    snapshot_digest = snapshot_digest or snapshot_sha256(source)
    output: list[ExtractedArtifact] = []
    for locator, record in _read_osv_records(source):
        external_id = record.get("id")
        if not isinstance(external_id, str) or not external_id:
            continue
        summary = record.get("summary") if isinstance(record.get("summary"), str) else ""
        details = record.get("details") if isinstance(record.get("details"), str) else ""
        text = f"OSV record {external_id}\nSummary: {summary}\n\n{details}".strip()
        status, reference, authorized = _explicit_osv_license(record)
        output.append(
            ExtractedArtifact(
                artifact=_artifact(
                    source_id=source_id,
                    external_id=external_id,
                    locator=locator,
                    source_revision=source_revision,
                    source_snapshot_sha256=snapshot_digest,
                    text=text,
                    license_status=status,
                    license_reference=reference,
                    inherited_source_license=False,
                    training_authorization=authorized,
                    benchmark_overlap_risk="MEDIUM",
                ),
                text=text,
                metadata={"provider": "OSV", "external_id": external_id},
            )
        )
    if not output:
        raise ValueError("OSV snapshot contains no records with ids")
    return output


def _nvd_records(payload: object) -> Iterable[dict[str, object]]:
    if isinstance(payload, dict) and isinstance(payload.get("vulnerabilities"), list):
        for wrapper in payload["vulnerabilities"]:
            if isinstance(wrapper, dict) and isinstance(wrapper.get("cve"), dict):
                yield wrapper["cve"]
        return
    if isinstance(payload, dict) and isinstance(payload.get("id"), str):
        yield payload


def extract_nvd_snapshot(
    path: str | Path,
    *,
    source_id: str,
    source_revision: str,
    snapshot_digest: str | None = None,
) -> list[ExtractedArtifact]:
    source = Path(path)
    snapshot_digest = snapshot_digest or snapshot_sha256(source)
    files = [source] if source.is_file() else sorted(source.rglob("*.json"))
    output: list[ExtractedArtifact] = []
    for file_path in files:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        for record in _nvd_records(payload):
            external_id = record.get("id")
            if not isinstance(external_id, str) or not external_id:
                continue
            descriptions = record.get("descriptions")
            description = ""
            if isinstance(descriptions, list):
                for item in descriptions:
                    if isinstance(item, dict) and item.get("lang") == "en" and isinstance(item.get("value"), str):
                        description = item["value"]
                        break
            text = f"NVD record {external_id}\n\n{description}".strip()
            locator = file_path.name if source.is_file() else file_path.relative_to(source).as_posix()
            output.append(
                ExtractedArtifact(
                    artifact=_artifact(
                        source_id=source_id,
                        external_id=external_id,
                        locator=locator,
                        source_revision=source_revision,
                        source_snapshot_sha256=snapshot_digest,
                        text=text,
                        license_status=LicenseStatus.REVIEW_REQUIRED,
                        license_reference=_NVD_TERMS_REFERENCE,
                        inherited_source_license=False,
                        training_authorization=False,
                        benchmark_overlap_risk="MEDIUM",
                    ),
                    text=text,
                    metadata={
                        "provider": "NVD",
                        "external_id": external_id,
                        "rights_note": "NVD use is permitted, but automated training authorization remains fail-closed pending provenance review of embedded CVE description text.",
                    },
                )
            )
    if not output:
        raise ValueError("NVD snapshot contains no CVE records")
    return output


def write_extracted_release(
    rows: list[ExtractedArtifact],
    *,
    manifest_path: str | Path,
    corpus_path: str | Path,
) -> None:
    manifest_destination = Path(manifest_path)
    corpus_destination = Path(corpus_path)
    manifest_destination.parent.mkdir(parents=True, exist_ok=True)
    corpus_destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_destination.write_text(
        "".join(
            json.dumps(row.artifact.model_dump(mode="json"), sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    corpus_destination.write_text(
        "".join(
            json.dumps(
                {
                    "artifact_id": row.artifact.artifact_id,
                    "content_sha256": row.artifact.content_sha256,
                    "text": row.text,
                    "metadata": row.metadata,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for row in rows
            if row.artifact.training_authorization
        ),
        encoding="utf-8",
    )
