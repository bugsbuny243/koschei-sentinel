from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from koschei_sentinel.cyber_sft_export_verify import (
    CyberSFTExportVerification,
    verify_cyber_sft_export,
)
from koschei_sentinel.cyber_sft_run_attestation import (
    CyberSFTRunAttestation,
    build_cyber_sft_run_attestation,
)
from koschei_sentinel.cyber_sft_trainer import CyberSFTAdapterManifest
from koschei_sentinel.cyber_sft_training import (
    CyberSFTPlan,
    load_cyber_sft_config,
    load_cyber_sft_examples,
    load_cyber_sft_validation_examples,
)

_RUN_METADATA_FILES = (
    "adapter-manifest.json",
    "training-receipt.json",
    "model-runtime.json",
    "resume-runtime.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_path(
    root: Path,
    value: str | Path,
    *,
    label: str,
    directory: bool = False,
) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    lexical = Path(os.path.abspath(candidate))
    if lexical != root and root not in lexical.parents:
        raise ValueError(f"{label} escapes repository root")

    current = lexical
    while current != root:
        if current.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink")
        current = current.parent

    if directory:
        if not lexical.is_dir():
            raise ValueError(f"{label} directory is missing")
    elif not lexical.is_file():
        raise ValueError(f"{label} file is missing")
    return lexical


def _copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


def _validate_adapter_relative_path(relative: str) -> Path:
    if "\\" in relative:
        raise ValueError(f"adapter file path is not portable: {relative}")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or not parsed.parts or ".." in parsed.parts:
        raise ValueError(f"adapter file path is not portable: {relative}")
    if any(part in {"", "."} for part in parsed.parts):
        raise ValueError(f"adapter file path is not canonical: {relative}")
    if parsed.parts[0] != "adapter":
        raise ValueError(f"adapter file must remain under adapter/: {relative}")
    return Path(*parsed.parts)


def build_cyber_sft_export(
    *,
    config_path: str | Path,
    plan_path: str | Path,
    training_source_path: str | Path,
    model_preflight_path: str | Path,
    verification_path: str | Path,
    attestation_path: str | Path,
    output_dir: str | Path,
    root: str | Path = ".",
) -> CyberSFTExportVerification:
    """Build an atomic, minimal, independently verifiable Cyber SFT candidate export."""

    root_path = Path(root).resolve()
    config_source = _source_path(root_path, config_path, label="training config")
    plan_source = _source_path(root_path, plan_path, label="training plan")
    training_source = _source_path(
        root_path,
        training_source_path,
        label="training source binding",
    )
    preflight_source = _source_path(
        root_path,
        model_preflight_path,
        label="model preflight",
    )
    verification_source = _source_path(
        root_path,
        verification_path,
        label="artifact verification",
    )
    attestation_source = _source_path(
        root_path,
        attestation_path,
        label="run attestation",
    )

    config = load_cyber_sft_config(config_source)
    plan = CyberSFTPlan.model_validate_json(plan_source.read_bytes())
    supplied_attestation = CyberSFTRunAttestation.model_validate_json(
        attestation_source.read_bytes()
    )

    fresh_attestation = build_cyber_sft_run_attestation(
        config_path=config_source,
        plan_path=plan_source,
        training_source_path=training_source,
        run_dir=config.output_dir,
        model_preflight_path=preflight_source,
        verification_path=verification_source,
        selected_profile=supplied_attestation.selected_profile,
        repository_commit=supplied_attestation.repository_commit,
        root=root_path,
    )
    if fresh_attestation.model_dump(mode="json") != supplied_attestation.model_dump(
        mode="json"
    ):
        raise ValueError("supplied run attestation differs from a fresh source rebuild")

    train_rows, train_examples_sha, train_manifest_sha, _ = load_cyber_sft_examples(
        config,
        root=root_path,
    )
    if train_examples_sha != supplied_attestation.corpus_examples_sha256:
        raise ValueError("source TRAIN examples SHA-256 differs from run attestation")
    if train_manifest_sha != supplied_attestation.corpus_manifest_sha256:
        raise ValueError("source TRAIN manifest SHA-256 differs from run attestation")
    if len(train_rows) != plan.training_examples:
        raise ValueError("source TRAIN example count differs from training plan")

    validation = load_cyber_sft_validation_examples(config, root=root_path)
    if plan.explicit_validation:
        if validation is None:
            raise ValueError("training plan requires an explicit VALIDATION corpus")
        validation_rows, validation_examples_sha, validation_manifest_sha, _ = validation
        if validation_examples_sha != plan.validation_corpus_examples_sha256:
            raise ValueError("source VALIDATION examples SHA-256 differs from training plan")
        if validation_manifest_sha != plan.validation_corpus_manifest_sha256:
            raise ValueError("source VALIDATION manifest SHA-256 differs from training plan")
        if len(validation_rows) != plan.validation_examples:
            raise ValueError("source VALIDATION example count differs from training plan")
    elif validation is not None:
        raise ValueError("source config has VALIDATION corpus but training plan does not")

    run_source = _source_path(
        root_path,
        config.output_dir,
        label="Cyber SFT run",
        directory=True,
    )
    manifest_source = _source_path(
        root_path,
        run_source / "adapter-manifest.json",
        label="adapter manifest",
    )
    manifest = CyberSFTAdapterManifest.model_validate_json(manifest_source.read_bytes())

    run_files: list[tuple[Path, Path]] = []
    for name in _RUN_METADATA_FILES:
        source = _source_path(
            root_path,
            run_source / name,
            label=f"run artifact {name}",
        )
        run_files.append((source, Path("run") / name))
    seen_adapter_files: set[str] = set()
    for relative in manifest.adapter_files:
        if relative in seen_adapter_files:
            raise ValueError(f"adapter manifest contains duplicate file path: {relative}")
        seen_adapter_files.add(relative)
        portable_relative = _validate_adapter_relative_path(relative)
        source = _source_path(
            root_path,
            run_source / portable_relative,
            label=f"adapter file {relative}",
        )
        run_files.append((source, Path("run") / portable_relative))

    corpus_source = _source_path(
        root_path,
        config.corpus_dir,
        label="TRAIN corpus",
        directory=True,
    )
    corpus_examples = _source_path(
        root_path,
        corpus_source / "examples.jsonl",
        label="TRAIN examples",
    )
    corpus_manifest = _source_path(
        root_path,
        corpus_source / "manifest.json",
        label="TRAIN manifest",
    )
    if _sha256(corpus_examples) != supplied_attestation.corpus_examples_sha256:
        raise ValueError("TRAIN examples changed during export preflight")
    if _sha256(corpus_manifest) != supplied_attestation.corpus_manifest_sha256:
        raise ValueError("TRAIN manifest changed during export preflight")

    output = Path(output_dir).expanduser().absolute()
    if output.exists():
        raise ValueError("candidate export destination already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    try:
        flat_files = (
            (config_source, "training-config.json"),
            (plan_source, "training-plan.json"),
            (training_source, "training-source.json"),
            (preflight_source, "model-preflight.json"),
            (verification_source, "verification.json"),
            (attestation_source, "run-attestation.json"),
            (corpus_examples, "corpus-examples.jsonl"),
            (corpus_manifest, "corpus-manifest.json"),
        )
        for source, name in flat_files:
            _copy_exact(source, staging / name)
        (staging / "selected-profile.txt").write_text(
            supplied_attestation.selected_profile + "\n",
            encoding="utf-8",
        )
        (staging / "repository-commit.txt").write_text(
            supplied_attestation.repository_commit + "\n",
            encoding="utf-8",
        )
        for source, relative in run_files:
            _copy_exact(source, staging / relative)

        report = verify_cyber_sft_export(staging)
        if not report.valid:
            details = "; ".join(report.violations) or "unknown verification failure"
            raise ValueError(f"fresh candidate export verification failed: {details}")
        staging.rename(output)
        return report
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
