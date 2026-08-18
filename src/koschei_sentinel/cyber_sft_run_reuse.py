from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from koschei_sentinel.cyber_sft_artifact_verify import verify_cyber_sft_run
from koschei_sentinel.cyber_sft_trainer import CyberSFTAdapterManifest
from koschei_sentinel.cyber_sft_training_source import (
    CyberSFTTrainingSourceBinding,
    verify_training_source_binding,
)
from koschei_sentinel.models import StrictModel


class CyberSFTRunReuseReport(StrictModel):
    schema_version: Literal["sentinel.cyber-sft-run-reuse-report.v1"] = (
        "sentinel.cyber-sft-run-reuse-report.v1"
    )
    run_dir: str
    run_exists: bool
    run_verification_valid: bool
    source_binding_valid: bool
    run_source_identity_valid: bool
    reusable: bool
    violations: list[str]


def _expected_resume_binding(source: CyberSFTTrainingSourceBinding) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "sentinel.cyber-sft-resume-binding.v1",
        "run_id": source.run_id,
        "base_model": source.base_model,
        "base_revision": source.base_revision,
        "corpus_examples_sha256": source.corpus_examples_sha256,
        "corpus_manifest_sha256": source.corpus_manifest_sha256,
        "config_sha256": source.config_sha256,
    }
    if source.explicit_validation:
        payload.update(
            {
                "explicit_validation": True,
                "validation_corpus_examples_sha256": source.validation_corpus_examples_sha256,
                "validation_corpus_manifest_sha256": source.validation_corpus_manifest_sha256,
            }
        )
    return payload


def _resume_binding_matches_source(
    run_path: Path,
    source: CyberSFTTrainingSourceBinding,
) -> tuple[bool, str | None]:
    path = run_path / "resume-runtime.json"
    if not path.is_file():
        return False, "completed run resume-runtime.json is missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"completed run resume-runtime.json is invalid: {exc}"
    if not isinstance(payload, dict):
        return False, "completed run resume-runtime.json must contain a JSON object"
    if payload.get("resume_binding") != _expected_resume_binding(source):
        return False, "completed run resume binding differs from training source"
    return True, None


def evaluate_completed_run_reuse(
    *,
    run_dir: str,
    source_binding_path: str | Path,
    config_path: str | Path,
    plan_path: str | Path,
    repository_commit: str,
    root: str | Path = ".",
) -> CyberSFTRunReuseReport:
    root_path = Path(root).resolve()
    run_path = (root_path / run_dir).resolve()
    if run_path != root_path and root_path not in run_path.parents:
        raise ValueError("run_dir escapes reuse root")
    if not run_path.is_dir():
        return CyberSFTRunReuseReport(
            run_dir=run_dir,
            run_exists=False,
            run_verification_valid=False,
            source_binding_valid=False,
            run_source_identity_valid=False,
            reusable=False,
            violations=["completed run directory does not exist"],
        )

    violations: list[str] = []
    verification = verify_cyber_sft_run(run_dir, root=root_path)
    verification_valid = verification.valid
    if not verification_valid:
        violations.extend(
            f"completed run verification: {row}" for row in verification.violations
        )
        if not verification.violations:
            violations.append("completed run verification is invalid")

    source_valid = False
    source: CyberSFTTrainingSourceBinding | None = None
    source_path = Path(source_binding_path)
    if not source_path.is_file():
        violations.append("training source binding is missing")
    else:
        try:
            source = CyberSFTTrainingSourceBinding.model_validate_json(
                source_path.read_bytes()
            )
            verify_training_source_binding(
                source,
                config_path=config_path,
                plan_path=plan_path,
                repository_commit=repository_commit,
            )
            source_valid = True
        except (OSError, TypeError, ValueError) as exc:
            violations.append(f"training source binding is invalid: {exc}")

    run_source_valid = False
    manifest_path = run_path / "adapter-manifest.json"
    if source is not None and manifest_path.is_file():
        try:
            manifest = CyberSFTAdapterManifest.model_validate_json(
                manifest_path.read_bytes()
            )
        except ValueError as exc:
            violations.append(f"completed run adapter manifest is invalid: {exc}")
        else:
            checks = (
                ("run_id", manifest.run_id, source.run_id),
                ("base_model", manifest.base_model, source.base_model),
                ("base_revision", manifest.base_revision, source.base_revision),
                (
                    "corpus_examples_sha256",
                    manifest.corpus_examples_sha256,
                    source.corpus_examples_sha256,
                ),
                (
                    "corpus_manifest_sha256",
                    manifest.corpus_manifest_sha256,
                    source.corpus_manifest_sha256,
                ),
            )
            mismatches = [
                label for label, observed, expected in checks if observed != expected
            ]
            resume_matches, resume_violation = _resume_binding_matches_source(
                run_path,
                source,
            )
            if mismatches:
                violations.append(
                    "completed run identity differs from training source: "
                    + ", ".join(mismatches)
                )
            if resume_violation is not None:
                violations.append(resume_violation)
            if not mismatches and resume_matches:
                run_source_valid = True
    elif not manifest_path.is_file():
        violations.append("completed run adapter manifest is missing")

    reusable = verification_valid and source_valid and run_source_valid and not violations
    return CyberSFTRunReuseReport(
        run_dir=run_dir,
        run_exists=True,
        run_verification_valid=verification_valid,
        source_binding_valid=source_valid,
        run_source_identity_valid=run_source_valid,
        reusable=reusable,
        violations=violations,
    )
