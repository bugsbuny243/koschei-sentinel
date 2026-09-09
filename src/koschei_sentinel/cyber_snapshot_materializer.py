from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.cyber_corpus_catalog import CyberSource, load_catalog
from koschei_sentinel.cyber_primary_extractors import (
    extract_attack_stix_snapshot,
    extract_kubernetes_security_snapshot,
    extract_yara_snapshot,
)
from koschei_sentinel.cyber_source_extractors import (
    extract_rustsec_snapshot,
    snapshot_sha256,
    write_extracted_release,
)
from koschei_sentinel.models import StrictModel

_DIGEST = r"^[a-f0-9]{64}$"
_PROVIDER_BY_SOURCE = {
    "rustsec.advisory.database": "rustsec",
    "mitre.attack.knowledge": "attack",
    "kubernetes.security.docs": "kubernetes",
    "yara.official.rules.docs": "yara",
}


class SnapshotMaterializationInput(StrictModel):
    source_id: str
    snapshot_path: str


class SnapshotMaterializationSpec(StrictModel):
    schema_version: Literal["sentinel.cyber-snapshot-materialization-spec.v3"] = (
        "sentinel.cyber-snapshot-materialization-spec.v3"
    )
    approved_catalog_path: str
    output_root: str
    inputs: list[SnapshotMaterializationInput] = Field(min_length=1, max_length=64)


class SnapshotReceipt(StrictModel):
    schema_version: Literal["sentinel.cyber-snapshot-receipt.v3"] = (
        "sentinel.cyber-snapshot-receipt.v3"
    )
    source_id: str
    provider: str
    canonical_locator: str
    approved_revision: str
    observed_git_revision: str
    revision_verified: bool
    snapshot_sha256: str = Field(pattern=_DIGEST)
    artifacts: int
    training_authorized_artifacts: int
    blocked_or_review_required_artifacts: int
    manifest_sha256: str = Field(pattern=_DIGEST)
    corpus_sha256: str = Field(pattern=_DIGEST)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_rev_parse(path: Path, revision: str) -> str:
    if not (path / ".git").exists():
        raise ValueError(f"snapshot is not a git checkout: {path}")
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", f"{revision}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"cannot resolve git revision {revision!r} in {path}") from exc
    value = completed.stdout.strip().lower()
    if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"git revision did not resolve to a commit SHA: {revision!r}")
    return value


def _verify_git_revision(path: Path, approved_revision: str) -> tuple[str, bool]:
    observed = _git_rev_parse(path, "HEAD")
    approved_commit = _git_rev_parse(path, approved_revision)
    return observed, observed == approved_commit


def _source_map(catalog_path: str | Path) -> dict[str, CyberSource]:
    return {source.source_id: source for source in load_catalog(catalog_path)}


def materialize_snapshot(
    *,
    source: CyberSource,
    snapshot_path: str | Path,
    output_root: str | Path,
) -> SnapshotReceipt:
    provider = _PROVIDER_BY_SOURCE.get(source.source_id)
    if provider is None:
        raise ValueError(f"no materializer provider registered for {source.source_id}")
    if not source.training_authorization:
        raise ValueError(f"source is not training-authorized: {source.source_id}")
    if not source.pinned_revision or not source.canonical_locator:
        raise ValueError(f"approved source lacks immutable metadata: {source.source_id}")

    snapshot = Path(snapshot_path)
    if not snapshot.exists():
        raise ValueError(f"snapshot path does not exist: {snapshot}")

    observed, verified = _verify_git_revision(snapshot, source.pinned_revision)
    if not verified:
        approved_commit = _git_rev_parse(snapshot, source.pinned_revision)
        raise ValueError(
            f"snapshot git revision mismatch for {source.source_id}: "
            f"approved {source.pinned_revision} -> {approved_commit}, observed HEAD {observed}"
        )

    digest = snapshot_sha256(snapshot)
    output = Path(output_root) / source.source_id
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "artifacts.jsonl"
    corpus = output / "training-corpus.jsonl"

    kwargs = {
        "source_id": source.source_id,
        "source_revision": source.pinned_revision,
        "snapshot_digest": digest,
    }
    if provider == "rustsec":
        rows = extract_rustsec_snapshot(snapshot, **kwargs)
    elif provider == "attack":
        rows = extract_attack_stix_snapshot(snapshot, **kwargs)
    elif provider == "kubernetes":
        rows = extract_kubernetes_security_snapshot(snapshot, **kwargs)
    else:
        rows = extract_yara_snapshot(snapshot, **kwargs)

    write_extracted_release(rows, manifest_path=manifest, corpus_path=corpus)
    trainable = sum(row.artifact.training_authorization for row in rows)
    receipt = SnapshotReceipt(
        source_id=source.source_id,
        provider=provider,
        canonical_locator=source.canonical_locator,
        approved_revision=source.pinned_revision,
        observed_git_revision=observed,
        revision_verified=verified,
        snapshot_sha256=digest,
        artifacts=len(rows),
        training_authorized_artifacts=trainable,
        blocked_or_review_required_artifacts=len(rows) - trainable,
        manifest_sha256=_sha256_file(manifest),
        corpus_sha256=_sha256_file(corpus),
    )
    (output / "snapshot-receipt.json").write_text(
        json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def materialize_spec(spec: SnapshotMaterializationSpec) -> list[SnapshotReceipt]:
    sources = _source_map(spec.approved_catalog_path)
    seen: set[str] = set()
    receipts: list[SnapshotReceipt] = []
    for item in spec.inputs:
        if item.source_id in seen:
            raise ValueError(f"duplicate source_id in materialization spec: {item.source_id}")
        seen.add(item.source_id)
        source = sources.get(item.source_id)
        if source is None:
            raise ValueError(f"source not found in approved catalog: {item.source_id}")
        receipts.append(
            materialize_snapshot(
                source=source,
                snapshot_path=item.snapshot_path,
                output_root=spec.output_root,
            )
        )
    return receipts
