from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.blockchain_security_corpus import (
    BlockchainSecurityDocument,
    BlockchainSourceClass,
    ChainFamily,
    ThreatDomain,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.pretraining_corpus import RightsBasis

_DIGEST = r"^[a-f0-9]{64}$"
_PSEUDONYM = r"^[a-z][a-z0-9_]*_[a-f0-9]{24}$"
_ALLOWED_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".go",
        ".h",
        ".hpp",
        ".js",
        ".json",
        ".md",
        ".move",
        ".py",
        ".rs",
        ".sol",
        ".toml",
        ".ts",
        ".txt",
        ".yaml",
        ".yml",
    }
)
_MAX_FILE_BYTES = 100_000
_MAX_FILES = 1_000_000


class BlockchainSourceSnapshotSpec(StrictModel):
    schema_version: Literal["sentinel.blockchain-source-snapshot-spec.v1"] = (
        "sentinel.blockchain-source-snapshot-spec.v1"
    )
    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    source_class: BlockchainSourceClass
    rights_basis: RightsBasis
    snapshot_path: str = Field(min_length=1, max_length=1024)
    expected_snapshot_digest: str = Field(pattern=_DIGEST)
    chain_families: list[ChainFamily] = Field(min_length=1, max_length=16)
    threat_domains: list[ThreatDomain] = Field(min_length=1, max_length=32)
    family_refs: list[str] = Field(default_factory=list, max_length=128)
    include_suffixes: list[str] = Field(default_factory=lambda: sorted(_ALLOWED_SUFFIXES))
    max_file_bytes: int = Field(default=_MAX_FILE_BYTES, ge=1, le=1_000_000)

    @model_validator(mode="after")
    def spec_is_safe(self) -> BlockchainSourceSnapshotSpec:
        _validate_relative_path(self.snapshot_path, "snapshot_path")
        if len(self.chain_families) != len(set(self.chain_families)):
            raise ValueError("chain_families must be unique")
        if len(self.threat_domains) != len(set(self.threat_domains)):
            raise ValueError("threat_domains must be unique")
        if len(self.family_refs) != len(set(self.family_refs)):
            raise ValueError("family_refs must be unique")
        if any(not _is_pseudonym(value) for value in self.family_refs):
            raise ValueError("family_refs must be pseudonymized identifiers")
        suffixes = [value.casefold() for value in self.include_suffixes]
        if len(suffixes) != len(set(suffixes)):
            raise ValueError("include_suffixes must be unique")
        if any(value not in _ALLOWED_SUFFIXES for value in suffixes):
            raise ValueError("include_suffixes contains an unsupported source suffix")
        return self


class SnapshotFile(StrictModel):
    path: str = Field(min_length=1, max_length=2048)
    sha256: str = Field(pattern=_DIGEST)
    bytes: int = Field(ge=1, le=1_000_000)


class BlockchainSourceSnapshotManifest(StrictModel):
    schema_version: Literal["sentinel.blockchain-source-snapshot.v1"] = (
        "sentinel.blockchain-source-snapshot.v1"
    )
    source_id: str
    source_class: BlockchainSourceClass
    rights_basis: RightsBasis
    snapshot_digest: str = Field(pattern=_DIGEST)
    chain_families: list[ChainFamily]
    threat_domains: list[ThreatDomain]
    family_refs: list[str]
    files: list[SnapshotFile] = Field(min_length=1, max_length=_MAX_FILES)
    documents: int = Field(ge=1)
    corpus_file_digest: str = Field(pattern=_DIGEST)


class SnapshotIngestResult(StrictModel):
    manifest: BlockchainSourceSnapshotManifest
    documents: list[BlockchainSecurityDocument]


def load_snapshot_spec(path: str | Path) -> BlockchainSourceSnapshotSpec:
    try:
        return BlockchainSourceSnapshotSpec.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid blockchain source snapshot spec") from exc


def write_snapshot_spec(
    spec: BlockchainSourceSnapshotSpec,
    path: str | Path,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_no_replace(
        destination,
        json.dumps(spec.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )


def inspect_snapshot(
    spec: BlockchainSourceSnapshotSpec,
    *,
    root: str | Path = ".",
) -> tuple[Path, list[SnapshotFile]]:
    root_path = Path(root).resolve()
    snapshot = _resolve_under_root(root_path, spec.snapshot_path)
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise ValueError("snapshot_path must be a real directory")

    suffixes = {value.casefold() for value in spec.include_suffixes}
    rows: list[SnapshotFile] = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_symlink():
            raise ValueError("source snapshot may not contain symbolic links")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("source snapshot contains unsupported filesystem entries")
        if any(part.startswith(".") for part in path.relative_to(snapshot).parts):
            continue
        if path.suffix.casefold() not in suffixes:
            continue
        relative = path.relative_to(snapshot).as_posix()
        _validate_relative_path(relative, "snapshot file")
        size = path.stat().st_size
        if size < 1 or size > spec.max_file_bytes:
            raise ValueError(f"snapshot file size outside policy: {relative}")
        rows.append(SnapshotFile(path=relative, sha256=_hash_file(path), bytes=size))
        if len(rows) > _MAX_FILES:
            raise ValueError("source snapshot contains too many files")
    if not rows:
        raise ValueError("source snapshot contains no eligible text/source files")
    return snapshot, rows


def snapshot_digest(files: list[SnapshotFile]) -> str:
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.sha256.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(item.bytes).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def ingest_snapshot(
    spec: BlockchainSourceSnapshotSpec,
    *,
    root: str | Path = ".",
) -> SnapshotIngestResult:
    snapshot, files = inspect_snapshot(spec, root=root)
    actual_snapshot_digest = snapshot_digest(files)
    if actual_snapshot_digest != spec.expected_snapshot_digest:
        raise ValueError("source snapshot digest does not match trusted expected digest")

    documents: list[BlockchainSecurityDocument] = []
    seen_content: dict[str, str] = {}
    for item in files:
        path = snapshot / item.path
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"snapshot source is not UTF-8: {item.path}") from exc
        content_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if content_digest != item.sha256:
            raise ValueError(f"snapshot file changed during ingest: {item.path}")
        previous = seen_content.setdefault(content_digest, item.path)
        if previous != item.path:
            raise ValueError(
                f"duplicate content inside one snapshot: {previous} and {item.path}"
            )
        document_ref = _pseudonym(
            "doc",
            f"{spec.source_id}\0{item.path}\0{content_digest}",
        )
        documents.append(
            BlockchainSecurityDocument(
                document_ref=document_ref,
                source_class=spec.source_class,
                rights_basis=spec.rights_basis,
                source_snapshot_digest=actual_snapshot_digest,
                content_digest=content_digest,
                family_refs=spec.family_refs,
                chain_families=spec.chain_families,
                threat_domains=spec.threat_domains,
                text=text,
            )
        )

    payload = "".join(
        json.dumps(
            item.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
        for item in sorted(documents, key=lambda value: value.document_ref)
    )
    manifest = BlockchainSourceSnapshotManifest(
        source_id=spec.source_id,
        source_class=spec.source_class,
        rights_basis=spec.rights_basis,
        snapshot_digest=actual_snapshot_digest,
        chain_families=spec.chain_families,
        threat_domains=spec.threat_domains,
        family_refs=spec.family_refs,
        files=files,
        documents=len(documents),
        corpus_file_digest=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    )
    return SnapshotIngestResult(manifest=manifest, documents=documents)


def write_ingest_result(
    result: SnapshotIngestResult,
    *,
    corpus_path: str | Path,
    manifest_path: str | Path,
) -> None:
    corpus = Path(corpus_path)
    manifest = Path(manifest_path)
    if corpus.exists():
        raise FileExistsError(f"snapshot corpus already exists: {corpus}")
    if manifest.exists():
        raise FileExistsError(f"snapshot manifest already exists: {manifest}")

    corpus.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    corpus_payload = "".join(
        json.dumps(
            item.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
        for item in sorted(result.documents, key=lambda value: value.document_ref)
    )
    if hashlib.sha256(corpus_payload.encode("utf-8")).hexdigest() != (
        result.manifest.corpus_file_digest
    ):
        raise ValueError("ingest result corpus digest is inconsistent")

    _atomic_no_replace(corpus, corpus_payload)
    try:
        _atomic_no_replace(
            manifest,
            json.dumps(
                result.manifest.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    except Exception:
        try:
            corpus.unlink()
        except FileNotFoundError:
            pass
        raise


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def _pseudonym(prefix: str, material: str) -> str:
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _is_pseudonym(value: str) -> bool:
    prefix, separator, digest = value.rpartition("_")
    prefix_is_valid = all(
        character.islower() or character.isdigit() or character == "_"
        for character in prefix
    )
    return bool(
        separator
        and prefix
        and prefix[0].isalpha()
        and prefix_is_valid
        and len(digest) == 24
        and all(character in "0123456789abcdef" for character in digest)
    )


def _resolve_under_root(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("path escapes the repository root")
    return candidate


def _validate_relative_path(value: str, field: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or value.startswith("~")
        or "\\" in value
    ):
        raise ValueError(f"{field} must stay within its root")


def _atomic_no_replace(path: Path, payload: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    published = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"artifact already exists: {path}") from None
        published = True
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if published:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
