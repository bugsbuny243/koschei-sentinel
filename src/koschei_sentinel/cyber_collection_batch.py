from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.cyber_corpus_catalog import (
    CyberArtifact,
    CyberSource,
    audit_artifacts,
    load_catalog,
)
from koschei_sentinel.models import StrictModel

_DIGEST = r"^[a-f0-9]{64}$"


class CollectionInput(StrictModel):
    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    manifest_path: str = Field(min_length=1)
    corpus_path: str = Field(min_length=1)


class CollectionBatchSpec(StrictModel):
    schema_version: Literal["sentinel.cyber-collection-batch-spec.v3"] = (
        "sentinel.cyber-collection-batch-spec.v3"
    )
    batch_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    approved_catalog_path: str = Field(min_length=1)
    inputs: list[CollectionInput] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def source_ids_are_unique(self) -> CollectionBatchSpec:
        ids = [row.source_id for row in self.inputs]
        if len(ids) != len(set(ids)):
            raise ValueError("collection input source_id values must be unique")
        return self


class CollectionBatchSeal(StrictModel):
    schema_version: Literal["sentinel.cyber-collection-batch-seal.v3"] = (
        "sentinel.cyber-collection-batch-seal.v3"
    )
    batch_id: str
    ready_for_training_pipeline: bool
    sources: int
    artifacts: int
    training_artifacts: int
    rejected_artifacts: int
    artifact_manifest_sha256: str = Field(pattern=_DIGEST)
    training_corpus_sha256: str = Field(pattern=_DIGEST)
    source_ids: list[str]
    violations: list[str]


@dataclass(frozen=True)
class CorpusRow:
    artifact_id: str
    content_sha256: str
    raw: dict[str, object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_artifacts(path: Path) -> list[CyberArtifact]:
    rows: list[CyberArtifact] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank artifact manifest row at {path}:{line_number}")
        rows.append(CyberArtifact.model_validate_json(line))
    if not rows:
        raise ValueError(f"artifact manifest is empty: {path}")
    return rows


def _load_corpus(path: Path) -> list[CorpusRow]:
    rows: list[CorpusRow] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank corpus row at {path}:{line_number}")
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"corpus row is not an object at {path}:{line_number}")
        artifact_id = payload.get("artifact_id")
        content_sha256 = payload.get("content_sha256")
        text = payload.get("text")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError(f"corpus artifact_id missing at {path}:{line_number}")
        if artifact_id in seen_ids:
            raise ValueError(f"duplicate corpus artifact_id: {artifact_id}")
        if not isinstance(content_sha256, str) or len(content_sha256) != 64:
            raise ValueError(f"corpus content_sha256 invalid at {path}:{line_number}")
        if not isinstance(text, str) or not text:
            raise ValueError(f"corpus text missing at {path}:{line_number}")
        actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual != content_sha256:
            raise ValueError(f"corpus text hash mismatch for {artifact_id}")
        seen_ids.add(artifact_id)
        rows.append(CorpusRow(artifact_id, content_sha256, payload))
    return rows


def _source_map(sources: list[CyberSource]) -> dict[str, CyberSource]:
    return {source.source_id: source for source in sources}


def seal_collection_batch(
    spec: CollectionBatchSpec,
    *,
    output_dir: str | Path,
) -> CollectionBatchSeal:
    catalog = load_catalog(spec.approved_catalog_path)
    sources_by_id = _source_map(catalog)
    all_artifacts: list[CyberArtifact] = []
    corpus_rows: dict[str, CorpusRow] = {}
    violations: list[str] = []

    for item in spec.inputs:
        source = sources_by_id.get(item.source_id)
        if source is None:
            violations.append(f"unknown approved source: {item.source_id}")
            continue
        if not source.training_authorization:
            violations.append(f"source is not training-authorized: {item.source_id}")
            continue

        manifest_path = Path(item.manifest_path)
        corpus_path = Path(item.corpus_path)
        artifacts = _load_artifacts(manifest_path)
        rows = _load_corpus(corpus_path)

        for artifact in artifacts:
            if artifact.source_id != item.source_id:
                violations.append(
                    f"artifact {artifact.artifact_id} source_id does not match input {item.source_id}"
                )
            if artifact.source_revision != source.pinned_revision:
                violations.append(
                    f"artifact {artifact.artifact_id} revision does not match approved source pin"
                )
        all_artifacts.extend(artifacts)

        manifest_by_id = {artifact.artifact_id: artifact for artifact in artifacts}
        for row in rows:
            artifact = manifest_by_id.get(row.artifact_id)
            if artifact is None:
                violations.append(f"corpus row lacks manifest artifact: {row.artifact_id}")
                continue
            if not artifact.training_authorization:
                violations.append(f"unauthorized artifact present in corpus: {row.artifact_id}")
            if artifact.content_sha256 != row.content_sha256:
                violations.append(f"manifest/corpus hash mismatch: {row.artifact_id}")
            if row.artifact_id in corpus_rows:
                violations.append(f"duplicate artifact_id across corpus inputs: {row.artifact_id}")
            corpus_rows[row.artifact_id] = row

        authorized_ids = {
            artifact.artifact_id for artifact in artifacts if artifact.training_authorization
        }
        corpus_ids = {row.artifact_id for row in rows}
        missing = sorted(authorized_ids - corpus_ids)
        if missing:
            violations.extend(f"authorized artifact missing from corpus: {artifact_id}" for artifact_id in missing)

    audit = audit_artifacts(all_artifacts, catalog) if all_artifacts else None
    if audit is not None and not audit.ready_for_ingestion:
        violations.extend(audit.violations)
        violations.extend(
            f"duplicate content sha256: {digest}" for digest in audit.duplicate_content_sha256
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_output = output / "artifacts.jsonl"
    corpus_output = output / "training-corpus.jsonl"

    sorted_artifacts = sorted(all_artifacts, key=lambda row: row.artifact_id)
    manifest_output.write_text(
        "".join(
            json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n"
            for row in sorted_artifacts
        ),
        encoding="utf-8",
    )

    allowed_ids = {
        row.artifact_id for row in sorted_artifacts if row.training_authorization
    }
    selected_rows = [corpus_rows[key] for key in sorted(corpus_rows) if key in allowed_ids]
    corpus_output.write_text(
        "".join(
            json.dumps(row.raw, ensure_ascii=False, sort_keys=True) + "\n"
            for row in selected_rows
        ),
        encoding="utf-8",
    )

    rejected = len(sorted_artifacts) - len(allowed_ids)
    ready = not violations and len(selected_rows) == len(allowed_ids) and bool(allowed_ids)
    seal = CollectionBatchSeal(
        batch_id=spec.batch_id,
        ready_for_training_pipeline=ready,
        sources=len(spec.inputs),
        artifacts=len(sorted_artifacts),
        training_artifacts=len(selected_rows),
        rejected_artifacts=rejected,
        artifact_manifest_sha256=_sha256(manifest_output),
        training_corpus_sha256=_sha256(corpus_output),
        source_ids=sorted(item.source_id for item in spec.inputs),
        violations=sorted(set(violations)),
    )
    (output / "seal.json").write_text(
        json.dumps(seal.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return seal
