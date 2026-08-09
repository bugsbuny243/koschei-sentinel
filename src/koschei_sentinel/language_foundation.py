from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel

SOURCE_SCHEMA = "koschei.language-foundation-corpus.v1"
SOURCE_GENERATOR = "koschei-foundation-export/v1"
SOURCE_REPOSITORY = "bugsbuny243/koschei-lang"
RELEASE_SCHEMA = "sentinel.language-foundation-release.v1"
DOCUMENT_SCHEMA = "sentinel.language-foundation-document.v1"
_SPLITS = ("train", "validation", "test")
_COMMIT_RE = re.compile(r"^[a-f0-9]{40}$")


class LanguageFoundationBlocked(ValueError):
    """Raised when source truth or split integrity is not safe for model training."""


class SourceLanguageDocument(StrictModel):
    document_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    family: str = Field(min_length=1, max_length=1024)
    kind: Literal["reference", "koschei_source"]
    path: str = Field(min_length=1, max_length=2048)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    text: str


class SourceLanguageCorpus(StrictModel):
    schema_version: Literal["koschei.language-foundation-corpus.v1"]
    generator_version: Literal["koschei-foundation-export/v1"]
    source_repository: Literal["bugsbuny243/koschei-lang"]
    source_commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    document_count: int = Field(ge=1)
    family_count: int = Field(ge=1)
    total_bytes: int = Field(ge=1)
    corpus_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    documents: list[SourceLanguageDocument] = Field(min_length=1)

    @model_validator(mode="after")
    def shape_matches_documents(self) -> SourceLanguageCorpus:
        if self.document_count != len(self.documents):
            raise ValueError("source corpus document_count mismatch")
        families = {item.family for item in self.documents}
        if self.family_count != len(families):
            raise ValueError("source corpus family_count mismatch")
        paths = [item.path for item in self.documents]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("source corpus document paths must be unique and sorted")
        return self


class LanguageFoundationDocument(StrictModel):
    schema_version: Literal["sentinel.language-foundation-document.v1"] = DOCUMENT_SCHEMA
    document_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    family: str = Field(min_length=1, max_length=1024)
    kind: Literal["reference", "koschei_source"]
    path: str = Field(min_length=1, max_length=2048)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    text: str


class LanguageFoundationSplit(StrictModel):
    path: str
    documents: int = Field(ge=1)
    families: int = Field(ge=1)
    digest: str = Field(pattern=r"^[a-f0-9]{64}$")


class LanguageFoundationReleaseManifest(StrictModel):
    schema_version: Literal["sentinel.language-foundation-release.v1"] = RELEASE_SCHEMA
    source_schema: Literal["koschei.language-foundation-corpus.v1"] = SOURCE_SCHEMA
    source_generator: Literal["koschei-foundation-export/v1"] = SOURCE_GENERATOR
    source_repository: Literal["bugsbuny243/koschei-lang"] = SOURCE_REPOSITORY
    source_commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    source_corpus_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_total_bytes: int = Field(ge=1)
    split_seed: str = Field(min_length=1, max_length=256)
    leakage_detected: Literal[False] = False
    documents: int = Field(ge=3)
    families: int = Field(ge=3)
    splits: dict[str, LanguageFoundationSplit]

    @model_validator(mode="after")
    def has_exact_splits(self) -> LanguageFoundationReleaseManifest:
        if set(self.splits) != set(_SPLITS):
            raise ValueError("language foundation release must contain train/validation/test")
        if sum(item.documents for item in self.splits.values()) != self.documents:
            raise ValueError("language foundation release document count mismatch")
        if sum(item.families for item in self.splits.values()) != self.families:
            raise ValueError("language foundation release family count mismatch")
        return self


def load_source_corpus(path: str | Path) -> SourceLanguageCorpus:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LanguageFoundationBlocked("source corpus is not valid JSON") from exc
    try:
        corpus = SourceLanguageCorpus.model_validate(payload)
    except ValueError as exc:
        raise LanguageFoundationBlocked("source corpus schema validation failed") from exc
    return verify_source_corpus(corpus)


def verify_source_corpus(corpus: SourceLanguageCorpus) -> SourceLanguageCorpus:
    total_bytes = 0
    for document in corpus.documents:
        _verify_relative_path(document.path)
        raw = document.text.encode()
        total_bytes += len(raw)
        if hashlib.sha256(raw).hexdigest() != document.source_sha256:
            raise LanguageFoundationBlocked(f"source document hash mismatch: {document.path}")
        expected_family = _expected_family(document.path, document.kind)
        if document.family != expected_family:
            raise LanguageFoundationBlocked(f"source document family mismatch: {document.path}")
        if document.document_id != _document_id(document):
            raise LanguageFoundationBlocked(f"source document id mismatch: {document.path}")
    if total_bytes != corpus.total_bytes:
        raise LanguageFoundationBlocked("source corpus total_bytes mismatch")

    expected_digest = _source_corpus_digest(
        source_commit=corpus.source_commit,
        document_count=corpus.document_count,
        family_count=corpus.family_count,
        total_bytes=corpus.total_bytes,
        documents=[item.model_dump(mode="json") for item in corpus.documents],
    )
    if corpus.corpus_sha256 != expected_digest:
        raise LanguageFoundationBlocked("source corpus digest mismatch")
    return corpus


def build_language_foundation_release(
    corpus_path: str | Path,
    *,
    output_dir: str | Path,
    split_seed: str = "koschei-language-foundation-v1",
    expected_source_commit: str | None = None,
) -> LanguageFoundationReleaseManifest:
    if not split_seed or len(split_seed) > 256:
        raise LanguageFoundationBlocked("split seed must contain 1..256 characters")
    if expected_source_commit is not None and not _COMMIT_RE.fullmatch(
        expected_source_commit
    ):
        raise LanguageFoundationBlocked(
            "expected source commit must be a lowercase 40-character SHA"
        )

    corpus = load_source_corpus(corpus_path)
    if expected_source_commit is not None and corpus.source_commit != expected_source_commit:
        raise LanguageFoundationBlocked(
            "source corpus commit does not match the expected Koschei commit"
        )
    if corpus.family_count < 3:
        raise LanguageFoundationBlocked(
            "at least three source families are required for leakage-safe splits"
        )

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"language foundation release already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    assignments = _assign_families({item.family for item in corpus.documents}, split_seed)
    rows: dict[str, list[LanguageFoundationDocument]] = {name: [] for name in _SPLITS}
    for source in corpus.documents:
        rows[assignments[source.family]].append(
            LanguageFoundationDocument(
                document_id=source.document_id,
                family=source.family,
                kind=source.kind,
                path=source.path,
                source_sha256=source.source_sha256,
                text=source.text,
            )
        )

    staging = Path(tempfile.mkdtemp(prefix=".language-foundation.", dir=destination.parent))
    release = staging / "release"
    release.mkdir()
    try:
        split_reports: dict[str, LanguageFoundationSplit] = {}
        seen_families: dict[str, str] = {}
        for split_name in _SPLITS:
            split_rows = sorted(rows[split_name], key=lambda item: item.document_id)
            if not split_rows:
                raise LanguageFoundationBlocked(f"{split_name} split is empty")
            payload = "".join(
                canonical_json(item.model_dump(mode="json")) + "\n" for item in split_rows
            )
            split_path = release / f"{split_name}.jsonl"
            split_path.write_text(payload, encoding="utf-8")
            families = {item.family for item in split_rows}
            for family in families:
                previous = seen_families.setdefault(family, split_name)
                if previous != split_name:
                    raise LanguageFoundationBlocked(
                        f"family leakage detected between {previous} and {split_name}: {family}"
                    )
            split_reports[split_name] = LanguageFoundationSplit(
                path=split_path.name,
                documents=len(split_rows),
                families=len(families),
                digest=hashlib.sha256(payload.encode()).hexdigest(),
            )

        manifest = LanguageFoundationReleaseManifest(
            source_commit=corpus.source_commit,
            source_corpus_sha256=corpus.corpus_sha256,
            source_total_bytes=corpus.total_bytes,
            split_seed=split_seed,
            documents=corpus.document_count,
            families=corpus.family_count,
            splits=split_reports,
        )
        manifest_path = release / "language-foundation-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        verify_language_foundation_release(release)
        _fsync_tree(release)
        if destination.exists():
            raise FileExistsError(f"language foundation release already exists: {destination}")
        os.replace(release, destination)
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def verify_language_foundation_release(
    path: str | Path,
) -> LanguageFoundationReleaseManifest:
    root = Path(path)
    manifest_path = root / "language-foundation-manifest.json"
    if not manifest_path.is_file():
        raise LanguageFoundationBlocked(
            "release is missing language-foundation-manifest.json"
        )
    try:
        manifest = LanguageFoundationReleaseManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise LanguageFoundationBlocked(
            "language foundation release manifest is invalid"
        ) from exc

    family_splits: dict[str, str] = {}
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    all_documents: list[LanguageFoundationDocument] = []

    for split_name in _SPLITS:
        report = manifest.splits[split_name]
        split_path = root / report.path
        if split_path.parent != root or split_path.name != f"{split_name}.jsonl":
            raise LanguageFoundationBlocked(f"unsafe {split_name} split path")
        raw = split_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != report.digest:
            raise LanguageFoundationBlocked(f"{split_name} split digest mismatch")
        documents = _load_release_rows(raw, split_name)
        if len(documents) != report.documents:
            raise LanguageFoundationBlocked(f"{split_name} split document count mismatch")
        families = {item.family for item in documents}
        if len(families) != report.families:
            raise LanguageFoundationBlocked(f"{split_name} split family count mismatch")

        for document in documents:
            _verify_released_document(document)
            if document.document_id in seen_ids:
                raise LanguageFoundationBlocked(
                    f"duplicate released document id: {document.document_id}"
                )
            if document.path in seen_paths:
                raise LanguageFoundationBlocked(f"duplicate released path: {document.path}")
            seen_ids.add(document.document_id)
            seen_paths.add(document.path)
            all_documents.append(document)
        for family in families:
            previous = family_splits.setdefault(family, split_name)
            if previous != split_name:
                raise LanguageFoundationBlocked(
                    f"family leakage detected between {previous} and {split_name}: {family}"
                )

    if len(all_documents) != manifest.documents:
        raise LanguageFoundationBlocked("release total document count mismatch")
    if len(family_splits) != manifest.families:
        raise LanguageFoundationBlocked("release total family count mismatch")

    expected_assignments = _assign_families(set(family_splits), manifest.split_seed)
    for family, observed_split in family_splits.items():
        if expected_assignments[family] != observed_split:
            raise LanguageFoundationBlocked(
                f"deterministic split assignment mismatch: {family}"
            )

    ordered = sorted(all_documents, key=lambda item: item.path)
    total_bytes = sum(len(item.text.encode()) for item in ordered)
    if total_bytes != manifest.source_total_bytes:
        raise LanguageFoundationBlocked("release source byte count mismatch")
    source_documents = [_source_document_payload(item) for item in ordered]
    reconstructed = _source_corpus_digest(
        source_commit=manifest.source_commit,
        document_count=len(ordered),
        family_count=len(family_splits),
        total_bytes=total_bytes,
        documents=source_documents,
    )
    if reconstructed != manifest.source_corpus_sha256:
        raise LanguageFoundationBlocked(
            "release documents do not reconstruct the pinned source corpus"
        )
    return manifest


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_release_rows(raw: bytes, split_name: str) -> list[LanguageFoundationDocument]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise LanguageFoundationBlocked(f"{split_name} split is not UTF-8") from exc
    documents: list[LanguageFoundationDocument] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            document = LanguageFoundationDocument.model_validate_json(line)
        except ValueError as exc:
            raise LanguageFoundationBlocked(
                f"invalid {split_name} document at line {number}"
            ) from exc
        documents.append(document)
    return documents


def _assign_families(families: set[str], seed: str) -> dict[str, str]:
    ordered = sorted(
        families,
        key=lambda family: (
            hashlib.sha256(f"{seed}\0{family}".encode()).hexdigest(),
            family,
        ),
    )
    count = len(ordered)
    validation_count = max(1, count // 10)
    test_count = max(1, count // 10)
    if validation_count + test_count >= count:
        validation_count = 1
        test_count = 1
    train_count = count - validation_count - test_count
    if train_count < 1:
        raise LanguageFoundationBlocked("not enough families for train/validation/test")

    assignments: dict[str, str] = {}
    for index, family in enumerate(ordered):
        if index < train_count:
            assignments[family] = "train"
        elif index < train_count + validation_count:
            assignments[family] = "validation"
        else:
            assignments[family] = "test"
    return assignments


def _expected_family(path_value: str, kind: str) -> str:
    _verify_relative_path(path_value)
    parts = Path(path_value).parts
    if kind == "koschei_source":
        if len(parts) < 2 or parts[0] != "examples":
            raise LanguageFoundationBlocked(
                f"Koschei source is outside examples/: {path_value}"
            )
        return f"example:{parts[1]}"
    return f"reference:{path_value}"


def _document_id(document: SourceLanguageDocument | LanguageFoundationDocument) -> str:
    material = (
        f"{document.kind}\0{document.family}\0{document.path}\0{document.source_sha256}"
    )
    return hashlib.sha256(material.encode()).hexdigest()


def _verify_released_document(document: LanguageFoundationDocument) -> None:
    _verify_relative_path(document.path)
    raw = document.text.encode()
    if hashlib.sha256(raw).hexdigest() != document.source_sha256:
        raise LanguageFoundationBlocked(f"released source hash mismatch: {document.path}")
    if document.family != _expected_family(document.path, document.kind):
        raise LanguageFoundationBlocked(f"released family mismatch: {document.path}")
    if document.document_id != _document_id(document):
        raise LanguageFoundationBlocked(f"released document id mismatch: {document.path}")


def _source_document_payload(document: LanguageFoundationDocument) -> dict[str, str]:
    return {
        "document_id": document.document_id,
        "family": document.family,
        "kind": document.kind,
        "path": document.path,
        "source_sha256": document.source_sha256,
        "text": document.text,
    }


def _source_corpus_digest(
    *,
    source_commit: str,
    document_count: int,
    family_count: int,
    total_bytes: int,
    documents: list[dict[str, object]],
) -> str:
    payload = {
        "schema_version": SOURCE_SCHEMA,
        "generator_version": SOURCE_GENERATOR,
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": source_commit,
        "document_count": document_count,
        "family_count": family_count,
        "total_bytes": total_bytes,
        "documents": documents,
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def _verify_relative_path(value: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or value.startswith("~"):
        raise LanguageFoundationBlocked(f"unsafe source path: {value}")


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda value: len(value.parts),
        reverse=True,
    )
    for directory in [*directories, root]:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
