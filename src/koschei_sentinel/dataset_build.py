from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from koschei_sentinel.anonymize import require_dataset_salt
from koschei_sentinel.dataset import ExportManifest, export_jsonl, export_record, load_records
from koschei_sentinel.models import StrictModel
from koschei_sentinel.readiness import (
    DatasetReadinessReport,
    ReadinessPolicy,
    evaluate_release_readiness,
    write_readiness_report,
)
from koschei_sentinel.split import SplitConfig, split_dataset


class DatasetBuildManifest(StrictModel):
    schema_version: Literal["sentinel.dataset-build.v1"] = "sentinel.dataset-build.v1"
    status: Literal["ready", "not_ready", "rejected"]
    dry_run: bool
    materialized: bool
    salt_version: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    input_files: int = Field(ge=1)
    input_records: int = Field(ge=0)
    export_manifest: ExportManifest
    split_seed: str
    output_dir: str | None = None
    readiness: DatasetReadinessReport | None = None


def build_dataset_release(
    input_paths: list[str | Path],
    *,
    output_dir: str | Path | None,
    salt_version: str,
    policy: ReadinessPolicy | None = None,
    split_config: SplitConfig | None = None,
    dry_run: bool = False,
) -> DatasetBuildManifest:
    paths = sorted((Path(value) for value in input_paths), key=lambda value: str(value))
    if not paths:
        raise ValueError("at least one ARVIS export input is required")
    records = [record for path in paths for record in load_records(path)]
    return build_dataset_release_from_records(
        records,
        input_files=len(paths),
        output_dir=output_dir,
        salt_version=salt_version,
        policy=policy,
        split_config=split_config,
        dry_run=dry_run,
    )


def build_dataset_release_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    input_files: int,
    output_dir: str | Path | None,
    salt_version: str,
    policy: ReadinessPolicy | None = None,
    split_config: SplitConfig | None = None,
    dry_run: bool = False,
) -> DatasetBuildManifest:
    if input_files < 1:
        raise ValueError("input_files must describe at least one private source")
    if not dry_run and output_dir is None:
        raise ValueError("output_dir is required unless dry_run is enabled")

    destination = Path(output_dir) if output_dir is not None else None
    if destination is not None and destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")

    source_records = [dict(record) for record in records]
    export_manifest = export_jsonl(
        source_records,
        output_path=None,
        dry_run=True,
    ).model_copy(update={"dry_run": dry_run})
    active_split = split_config or SplitConfig()
    if export_manifest.rejected_records:
        return DatasetBuildManifest(
            status="rejected",
            dry_run=dry_run,
            materialized=False,
            salt_version=salt_version,
            input_files=input_files,
            input_records=len(source_records),
            export_manifest=export_manifest,
            split_seed=active_split.seed,
        )

    salt = require_dataset_salt()
    examples = [export_record(record, salt=salt) for record in source_records]
    active_policy = policy or ReadinessPolicy()
    staging_parent = destination.parent if destination is not None else Path(tempfile.gettempdir())
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".sentinel-dataset-build.", dir=staging_parent)
    )
    release = staging / "release"
    try:
        split_dataset(
            examples,
            output_dir=release,
            config=active_split,
            dry_run=False,
        )
        readiness = evaluate_release_readiness(release, policy=active_policy)
        status: Literal["ready", "not_ready"] = (
            "ready" if readiness.ready else "not_ready"
        )
        materialized = bool(readiness.ready and not dry_run)
        manifest = DatasetBuildManifest(
            status=status,
            dry_run=dry_run,
            materialized=materialized,
            salt_version=salt_version,
            input_files=input_files,
            input_records=len(source_records),
            export_manifest=export_manifest,
            split_seed=active_split.seed,
            output_dir=str(destination) if materialized and destination is not None else None,
            readiness=readiness,
        )
        if readiness.ready:
            write_readiness_report(readiness, release / "readiness-report.json")
            _write_manifest(manifest, release / "build-manifest.json")
            _fsync_tree(release)
            if materialized:
                assert destination is not None
                os.replace(release, destination)
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _write_manifest(manifest: DatasetBuildManifest, path: Path) -> None:
    payload = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")


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
