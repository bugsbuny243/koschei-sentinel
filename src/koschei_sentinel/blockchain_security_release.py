from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.blockchain_security_corpus import (
    BlockchainSecurityCorpusAudit,
    BlockchainSecurityDocument,
    ChainFamily,
    ThreatDomain,
    audit_blockchain_security_corpus,
    load_blockchain_security_documents,
    load_blockchain_security_policy,
)
from koschei_sentinel.blockchain_source_catalog import (
    BlockchainSourceCatalogManifest,
    audit_source_catalog,
    load_source_catalog,
    load_source_catalog_policy,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.pretraining_corpus import load_pretraining_holdout

_DIGEST = r"^[a-f0-9]{64}$"
_SPLITS = ("train", "validation", "test")
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


class BlockchainSecurityReleasePolicy(StrictModel):
    schema_version: Literal["sentinel.blockchain-security-release-policy.v1"] = (
        "sentinel.blockchain-security-release-policy.v1"
    )
    policy_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    split_seed: str = Field(min_length=1, max_length=256)
    validation_target_bps: int = Field(default=1000, ge=1, le=4000)
    test_target_bps: int = Field(default=1000, ge=1, le=4000)
    train_min_bps: int = Field(default=7500, ge=1000, le=9998)
    required_chain_families: list[ChainFamily] = Field(min_length=1)
    required_threat_domains: list[ThreatDomain] = Field(min_length=1)
    min_sources_per_required_chain_per_split: int = Field(default=1, ge=1, le=100_000)
    min_sources_per_required_threat_per_split: int = Field(default=1, ge=1, le=100_000)
    require_source_isolation: Literal[True] = True
    require_family_isolation: Literal[True] = True

    @model_validator(mode="after")
    def policy_is_canonical(self) -> BlockchainSecurityReleasePolicy:
        if self.validation_target_bps + self.test_target_bps >= 10_000:
            raise ValueError("validation/test targets leave no training split")
        if len(self.required_chain_families) != len(set(self.required_chain_families)):
            raise ValueError("required_chain_families must be unique")
        if len(self.required_threat_domains) != len(set(self.required_threat_domains)):
            raise ValueError("required_threat_domains must be unique")
        return self


class BlockchainSecurityReleaseSplit(StrictModel):
    path: str
    documents: int = Field(ge=1)
    sources: int = Field(ge=1)
    components: int = Field(ge=1)
    families: int = Field(ge=0)
    digest: str = Field(pattern=_DIGEST)
    source_snapshot_digests: list[str] = Field(min_length=1)
    chain_source_counts: dict[str, int]
    threat_source_counts: dict[str, int]


class BlockchainSecurityReleaseManifest(StrictModel):
    schema_version: Literal["sentinel.blockchain-security-release.v1"] = (
        "sentinel.blockchain-security-release.v1"
    )
    release_policy_id: str
    split_seed: str
    catalog_file_digest: str = Field(pattern=_DIGEST)
    catalog_policy_file_digest: str = Field(pattern=_DIGEST)
    source_catalog_manifest_file_digest: str = Field(pattern=_DIGEST)
    source_catalog_digest: str = Field(pattern=_DIGEST)
    source_corpus_file_digest: str = Field(pattern=_DIGEST)
    source_corpus_digest: str = Field(pattern=_DIGEST)
    corpus_audit_file_digest: str = Field(pattern=_DIGEST)
    corpus_audit_digest: str = Field(pattern=_DIGEST)
    holdout_file_digest: str = Field(pattern=_DIGEST)
    holdout_digest: str = Field(pattern=_DIGEST)
    benchmark_suite_digest: str = Field(pattern=_DIGEST)
    blockchain_policy_file_digest: str = Field(pattern=_DIGEST)
    blockchain_policy_digest: str = Field(pattern=_DIGEST)
    release_policy_file_digest: str = Field(pattern=_DIGEST)
    release_policy_digest: str = Field(pattern=_DIGEST)
    documents: int = Field(ge=3)
    sources: int = Field(ge=3)
    components: int = Field(ge=3)
    family_leakage_detected: Literal[False]
    source_leakage_detected: Literal[False]
    splits: dict[str, BlockchainSecurityReleaseSplit]

    @model_validator(mode="after")
    def exact_splits_and_totals(self) -> BlockchainSecurityReleaseManifest:
        if set(self.splits) != set(_SPLITS):
            raise ValueError("blockchain release must contain train/validation/test")
        if sum(item.documents for item in self.splits.values()) != self.documents:
            raise ValueError("blockchain release document count mismatch")
        if sum(item.sources for item in self.splits.values()) != self.sources:
            raise ValueError("blockchain release source count mismatch")
        if sum(item.components for item in self.splits.values()) != self.components:
            raise ValueError("blockchain release component count mismatch")
        return self


def load_blockchain_security_release_policy(
    path: str | Path,
) -> BlockchainSecurityReleasePolicy:
    try:
        return BlockchainSecurityReleasePolicy.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid blockchain security release policy") from exc


def build_blockchain_security_release(
    *,
    catalog_path: str | Path,
    catalog_policy_path: str | Path,
    source_catalog_manifest_path: str | Path,
    corpus_path: str | Path,
    holdout_path: str | Path,
    blockchain_policy_path: str | Path,
    corpus_audit_path: str | Path,
    release_policy_path: str | Path,
    output_dir: str | Path,
    root: str | Path = ".",
) -> BlockchainSecurityReleaseManifest:
    root_path = Path(root).resolve()
    paths = {
        "catalog": _resolve_file(root_path, catalog_path, "source catalog"),
        "catalog_policy": _resolve_file(
            root_path, catalog_policy_path, "source catalog policy"
        ),
        "source_manifest": _resolve_file(
            root_path, source_catalog_manifest_path, "source catalog manifest"
        ),
        "corpus": _resolve_file(root_path, corpus_path, "composed blockchain corpus"),
        "holdout": _resolve_file(root_path, holdout_path, "blockchain holdout"),
        "blockchain_policy": _resolve_file(
            root_path, blockchain_policy_path, "blockchain corpus policy"
        ),
        "corpus_audit": _resolve_file(
            root_path, corpus_audit_path, "blockchain corpus audit"
        ),
        "release_policy": _resolve_file(
            root_path, release_policy_path, "blockchain release policy"
        ),
    }

    catalog_result = audit_source_catalog(
        load_source_catalog(paths["catalog"]),
        load_source_catalog_policy(paths["catalog_policy"]),
        root=root_path,
    )
    if not catalog_result.manifest.ready:
        raise ValueError("blockchain source catalog is not ready")
    try:
        stored_source_manifest = BlockchainSourceCatalogManifest.model_validate_json(
            paths["source_manifest"].read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid stored source catalog manifest") from exc
    if stored_source_manifest != catalog_result.manifest:
        raise ValueError("source catalog manifest is stale or does not match source artifacts")

    documents = load_blockchain_security_documents(paths["corpus"])
    if documents != catalog_result.documents:
        raise ValueError("composed blockchain corpus does not match verified source catalog")
    corpus_file_digest = _hash_file(paths["corpus"])
    if corpus_file_digest != stored_source_manifest.combined_corpus_digest:
        raise ValueError("composed corpus file digest does not match source catalog manifest")

    holdout = load_pretraining_holdout(paths["holdout"])
    blockchain_policy = load_blockchain_security_policy(paths["blockchain_policy"])
    recomputed_audit = audit_blockchain_security_corpus(documents, holdout, blockchain_policy)
    try:
        stored_audit = BlockchainSecurityCorpusAudit.model_validate_json(
            paths["corpus_audit"].read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid stored blockchain corpus audit") from exc
    if stored_audit != recomputed_audit:
        raise ValueError("blockchain corpus audit is stale or does not match current inputs")
    if not recomputed_audit.ready:
        raise ValueError("blockchain corpus audit must pass before release construction")
    if recomputed_audit.corpus_digest != stored_source_manifest.combined_corpus_digest:
        raise ValueError("corpus audit digest does not match source catalog corpus digest")

    release_policy = load_blockchain_security_release_policy(paths["release_policy"])
    split_rows, split_components = _assign_release_splits(
        documents,
        stored_source_manifest,
        release_policy,
    )
    _validate_split_coverage(split_rows, stored_source_manifest, release_policy)

    destination = _resolve_output(root_path, output_dir)
    if destination.exists():
        raise FileExistsError(f"blockchain security release already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".blockchain-release.", dir=destination.parent))
    release = staging / "release"
    release.mkdir()
    try:
        split_reports: dict[str, BlockchainSecurityReleaseSplit] = {}
        snapshot_to_binding = {
            item.snapshot_digest: item for item in stored_source_manifest.bindings
        }
        for split_name in _SPLITS:
            rows = sorted(split_rows[split_name], key=lambda item: item.document_ref)
            snapshots = sorted({item.source_snapshot_digest for item in rows})
            payload = _document_payload(rows)
            path = release / f"{split_name}.jsonl"
            path.write_text(payload, encoding="utf-8")
            chain_counts: Counter[str] = Counter()
            threat_counts: Counter[str] = Counter()
            for snapshot in snapshots:
                binding = snapshot_to_binding[snapshot]
                for chain in set(binding.chain_families):
                    chain_counts[chain.value] += 1
                for threat in set(binding.threat_domains):
                    threat_counts[threat.value] += 1
            families = {family for item in rows for family in item.family_refs}
            split_reports[split_name] = BlockchainSecurityReleaseSplit(
                path=path.name,
                documents=len(rows),
                sources=len(snapshots),
                components=len(split_components[split_name]),
                families=len(families),
                digest=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                source_snapshot_digests=snapshots,
                chain_source_counts=dict(sorted(chain_counts.items())),
                threat_source_counts=dict(sorted(threat_counts.items())),
            )

        manifest = BlockchainSecurityReleaseManifest(
            release_policy_id=release_policy.policy_id,
            split_seed=release_policy.split_seed,
            catalog_file_digest=_hash_file(paths["catalog"]),
            catalog_policy_file_digest=_hash_file(paths["catalog_policy"]),
            source_catalog_manifest_file_digest=_hash_file(paths["source_manifest"]),
            source_catalog_digest=stored_source_manifest.catalog_digest,
            source_corpus_file_digest=corpus_file_digest,
            source_corpus_digest=stored_source_manifest.combined_corpus_digest,
            corpus_audit_file_digest=_hash_file(paths["corpus_audit"]),
            corpus_audit_digest=_model_digest(stored_audit),
            holdout_file_digest=_hash_file(paths["holdout"]),
            holdout_digest=recomputed_audit.holdout_digest,
            benchmark_suite_digest=recomputed_audit.benchmark_suite_digest,
            blockchain_policy_file_digest=_hash_file(paths["blockchain_policy"]),
            blockchain_policy_digest=recomputed_audit.policy_digest,
            release_policy_file_digest=_hash_file(paths["release_policy"]),
            release_policy_digest=_model_digest(release_policy),
            documents=len(documents),
            sources=stored_source_manifest.sources,
            components=sum(len(value) for value in split_components.values()),
            family_leakage_detected=False,
            source_leakage_detected=False,
            splits=split_reports,
        )
        (release / "blockchain-security-release-manifest.json").write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        verify_blockchain_security_release(release)
        _fsync_tree(release)
        _publish_directory_no_replace(release, destination)
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def verify_blockchain_security_release(
    path: str | Path,
) -> BlockchainSecurityReleaseManifest:
    root = Path(path)
    manifest_path = root / "blockchain-security-release-manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("release is missing blockchain-security-release-manifest.json")
    try:
        manifest = BlockchainSecurityReleaseManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid blockchain security release manifest") from exc

    seen_snapshots: set[str] = set()
    family_split: dict[str, str] = {}
    all_documents: list[BlockchainSecurityDocument] = []
    for split_name in _SPLITS:
        report = manifest.splits[split_name]
        if report.path != f"{split_name}.jsonl":
            raise ValueError(f"unsafe {split_name} release path")
        split_path = root / report.path
        if not split_path.is_file() or split_path.is_symlink():
            raise ValueError(f"missing {split_name} release split")
        raw = split_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != report.digest:
            raise ValueError(f"{split_name} release digest mismatch")
        rows = load_blockchain_security_documents(split_path)
        if len(rows) != report.documents:
            raise ValueError(f"{split_name} release document count mismatch")
        refs = [item.document_ref for item in rows]
        if refs != sorted(refs) or len(refs) != len(set(refs)):
            raise ValueError(f"{split_name} release rows are not canonical and unique")
        snapshots = sorted({item.source_snapshot_digest for item in rows})
        if snapshots != report.source_snapshot_digests:
            raise ValueError(f"{split_name} release source snapshot set mismatch")
        overlap = seen_snapshots.intersection(snapshots)
        if overlap:
            raise ValueError("source snapshot leakage detected across release splits")
        seen_snapshots.update(snapshots)
        for item in rows:
            for family in item.family_refs:
                previous = family_split.setdefault(family, split_name)
                if previous != split_name:
                    raise ValueError("family leakage detected across release splits")
        all_documents.extend(rows)

    ordered = sorted(all_documents, key=lambda item: item.document_ref)
    refs = [item.document_ref for item in ordered]
    if len(refs) != len(set(refs)):
        raise ValueError("duplicate document_ref detected across release splits")
    if len(ordered) != manifest.documents:
        raise ValueError("release total document count mismatch")
    if len(seen_snapshots) != manifest.sources:
        raise ValueError("release total source count mismatch")
    if hashlib.sha256(_document_payload(ordered).encode("utf-8")).hexdigest() != (
        manifest.source_corpus_digest
    ):
        raise ValueError("release splits do not reconstruct the pinned source corpus")
    return manifest


def _assign_release_splits(
    documents: list[BlockchainSecurityDocument],
    source_manifest: BlockchainSourceCatalogManifest,
    policy: BlockchainSecurityReleasePolicy,
) -> tuple[
    dict[str, list[BlockchainSecurityDocument]],
    dict[str, list[tuple[str, ...]]],
]:
    valid_snapshots = {item.snapshot_digest for item in source_manifest.bindings}
    documents_by_snapshot: dict[str, list[BlockchainSecurityDocument]] = defaultdict(list)
    for document in documents:
        if document.source_snapshot_digest not in valid_snapshots:
            raise ValueError("document references a snapshot absent from source catalog")
        documents_by_snapshot[document.source_snapshot_digest].append(document)
    if set(documents_by_snapshot) != valid_snapshots:
        raise ValueError("source catalog contains snapshot without composed corpus documents")

    components = _family_components(documents_by_snapshot)
    if len(components) < 3:
        raise ValueError("at least three family-connected source components are required")
    ordered_components = sorted(
        components,
        key=lambda component: (
            hashlib.sha256(
                (policy.split_seed + "\0" + "\0".join(component)).encode("utf-8")
            ).hexdigest(),
            component,
        ),
    )
    total_sources = len(valid_snapshots)
    validation_target = max(
        1, (total_sources * policy.validation_target_bps + 9_999) // 10_000
    )
    test_target = max(1, (total_sources * policy.test_target_bps + 9_999) // 10_000)

    split_components: dict[str, list[tuple[str, ...]]] = {name: [] for name in _SPLITS}
    validation_sources = 0
    test_sources = 0
    for component in ordered_components:
        if validation_sources < validation_target:
            split = "validation"
            validation_sources += len(component)
        elif test_sources < test_target:
            split = "test"
            test_sources += len(component)
        else:
            split = "train"
        split_components[split].append(component)

    if any(not split_components[name] for name in _SPLITS):
        raise ValueError("deterministic split assignment produced an empty split")
    train_sources = sum(len(item) for item in split_components["train"])
    if _share_bps(train_sources, total_sources) < policy.train_min_bps:
        raise ValueError("family-connected components leave too little source diversity for train")

    split_rows: dict[str, list[BlockchainSecurityDocument]] = {name: [] for name in _SPLITS}
    for split_name, groups in split_components.items():
        for group in groups:
            for snapshot in group:
                split_rows[split_name].extend(documents_by_snapshot[snapshot])
    return split_rows, split_components


def _family_components(
    documents_by_snapshot: dict[str, list[BlockchainSecurityDocument]],
) -> list[tuple[str, ...]]:
    snapshots = sorted(documents_by_snapshot)
    parent = {snapshot: snapshot for snapshot in snapshots}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    family_owner: dict[str, str] = {}
    for snapshot in snapshots:
        families = {
            family
            for document in documents_by_snapshot[snapshot]
            for family in document.family_refs
        }
        for family in sorted(families):
            previous = family_owner.setdefault(family, snapshot)
            union(previous, snapshot)

    groups: dict[str, list[str]] = defaultdict(list)
    for snapshot in snapshots:
        groups[find(snapshot)].append(snapshot)
    return sorted(tuple(sorted(values)) for values in groups.values())


def _validate_split_coverage(
    split_rows: dict[str, list[BlockchainSecurityDocument]],
    source_manifest: BlockchainSourceCatalogManifest,
    policy: BlockchainSecurityReleasePolicy,
) -> None:
    binding_by_snapshot = {
        item.snapshot_digest: item for item in source_manifest.bindings
    }
    required_chains = {item.value for item in policy.required_chain_families}
    required_threats = {item.value for item in policy.required_threat_domains}
    for split_name in _SPLITS:
        snapshots = {item.source_snapshot_digest for item in split_rows[split_name]}
        chain_counts: Counter[str] = Counter()
        threat_counts: Counter[str] = Counter()
        for snapshot in snapshots:
            binding = binding_by_snapshot[snapshot]
            for chain in set(binding.chain_families):
                chain_counts[chain.value] += 1
            for threat in set(binding.threat_domains):
                threat_counts[threat.value] += 1
        for chain in sorted(required_chains):
            if chain_counts[chain] < policy.min_sources_per_required_chain_per_split:
                raise ValueError(
                    f"{split_name} split lacks independent source coverage for chain {chain}"
                )
        for threat in sorted(required_threats):
            if threat_counts[threat] < policy.min_sources_per_required_threat_per_split:
                raise ValueError(
                    f"{split_name} split lacks independent source coverage for threat {threat}"
                )


def _document_payload(documents: list[BlockchainSecurityDocument]) -> str:
    return "".join(
        json.dumps(
            item.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
        for item in documents
    )


def _model_digest(model: StrictModel) -> str:
    return hashlib.sha256(
        json.dumps(
            model.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def _share_bps(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        return 0
    return (numerator * 10_000 + denominator - 1) // denominator


def _resolve_file(root: Path, value: str | Path, label: str) -> Path:
    relative = _relative_path(root, value, label)
    path = (root / relative).resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing or not a regular file: {relative}")
    return path


def _resolve_output(root: Path, value: str | Path) -> Path:
    relative = _relative_path(root, value, "output_dir")
    return (root / relative).resolve()


def _relative_path(root: Path, value: str | Path, field: str) -> str:
    path = Path(value)
    if path.is_absolute():
        try:
            relative = path.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{field} must stay within the repository root") from exc
        value = relative.as_posix()
    else:
        value = PurePosixPath(str(path)).as_posix()
    parsed = PurePosixPath(value)
    if (
        not value
        or parsed.is_absolute()
        or ".." in parsed.parts
        or value.startswith("~")
        or "\\" in value
    ):
        raise ValueError(f"{field} must stay within the repository root")
    return value


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    if not source.is_dir() or source.is_symlink():
        raise ValueError("staged release must be a real directory")
    if os.name != "posix":
        raise ValueError("atomic no-replace blockchain release publication requires POSIX")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError) as exc:
        raise ValueError("atomic no-replace publication requires renameat2") from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(f"blockchain security release already exists: {destination}")
        if error_number in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
            raise ValueError("atomic no-replace publication unsupported by filesystem")
        raise OSError(error_number, os.strerror(error_number), destination)
    _fsync_directory(destination.parent)


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
        _fsync_directory(directory)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
