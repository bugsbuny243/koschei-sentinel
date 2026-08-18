from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.cyber_sft_trainer import (
    CyberSFTAdapterManifest,
    CyberSFTTrainingReceipt,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import canonical_json, resolve_under_root


class CyberSFTArtifactVerification(StrictModel):
    schema_version: Literal["sentinel.cyber-sft-artifact-verification.v1"] = (
        "sentinel.cyber-sft-artifact-verification.v1"
    )
    run_id: str | None
    valid: bool
    smoke_only: bool | None
    adapter_digest_verified: bool
    receipt_digest_verified: bool
    receipt_bindings_verified: bool
    global_step: int | None = Field(default=None, ge=0)
    violations: list[str]


def _adapter_digest(run_path: Path, files: list[str]) -> str:
    digest = hashlib.sha256()
    for relative in files:
        candidate = (run_path / relative).resolve()
        if candidate != run_path and run_path not in candidate.parents:
            raise ValueError(f"adapter file escapes run directory: {relative}")
        if not candidate.is_file():
            raise ValueError(f"adapter file is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(candidate.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _receipt_digest(receipt: CyberSFTTrainingReceipt) -> str:
    payload = receipt.model_dump(mode="json")
    payload.pop("receipt_sha256", None)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def verify_cyber_sft_run(
    run_dir: str,
    *,
    root: str | Path = ".",
) -> CyberSFTArtifactVerification:
    root_path = Path(root).resolve()
    run_path = resolve_under_root(root_path, run_dir)
    violations: list[str] = []
    manifest_path = run_path / "adapter-manifest.json"
    receipt_path = run_path / "training-receipt.json"
    if not manifest_path.is_file():
        violations.append("adapter-manifest.json is missing")
    if not receipt_path.is_file():
        violations.append("training-receipt.json is missing")
    if violations:
        return CyberSFTArtifactVerification(
            run_id=None,
            valid=False,
            smoke_only=None,
            adapter_digest_verified=False,
            receipt_digest_verified=False,
            receipt_bindings_verified=False,
            global_step=None,
            violations=violations,
        )

    try:
        manifest = CyberSFTAdapterManifest.model_validate_json(manifest_path.read_bytes())
    except ValueError as exc:
        violations.append(f"adapter manifest is invalid: {exc}")
        return CyberSFTArtifactVerification(
            run_id=None,
            valid=False,
            smoke_only=None,
            adapter_digest_verified=False,
            receipt_digest_verified=False,
            receipt_bindings_verified=False,
            global_step=None,
            violations=violations,
        )
    try:
        receipt = CyberSFTTrainingReceipt.model_validate_json(receipt_path.read_bytes())
    except ValueError as exc:
        violations.append(f"training receipt is invalid: {exc}")
        return CyberSFTArtifactVerification(
            run_id=manifest.run_id,
            valid=False,
            smoke_only=(manifest.corpus_promotion_eligible is False),
            adapter_digest_verified=False,
            receipt_digest_verified=False,
            receipt_bindings_verified=False,
            global_step=None,
            violations=violations,
        )

    adapter_verified = False
    try:
        observed_adapter_digest = _adapter_digest(run_path, manifest.adapter_files)
        adapter_verified = observed_adapter_digest == manifest.adapter_digest
        if not adapter_verified:
            violations.append("adapter directory digest differs from adapter manifest")
    except ValueError as exc:
        violations.append(str(exc))

    receipt_verified = _receipt_digest(receipt) == receipt.receipt_sha256
    if not receipt_verified:
        violations.append("training receipt self-hash does not verify")

    binding_pairs = [
        ("run_id", receipt.run_id, manifest.run_id),
        ("stage", receipt.stage, manifest.stage),
        ("base_model", receipt.base_model, manifest.base_model),
        ("base_revision", receipt.base_revision, manifest.base_revision),
        (
            "corpus_examples_sha256",
            receipt.corpus_examples_sha256,
            manifest.corpus_examples_sha256,
        ),
        (
            "corpus_manifest_sha256",
            receipt.corpus_manifest_sha256,
            manifest.corpus_manifest_sha256,
        ),
        (
            "corpus_promotion_eligible",
            receipt.corpus_promotion_eligible,
            manifest.corpus_promotion_eligible,
        ),
        ("adapter_digest", receipt.adapter_digest, manifest.adapter_digest),
        ("optimizer", receipt.optimizer, manifest.optimizer),
    ]
    bindings_verified = True
    for field, receipt_value, manifest_value in binding_pairs:
        if receipt_value != manifest_value:
            bindings_verified = False
            violations.append(f"training receipt binding mismatch: {field}")

    if receipt.global_step <= 0:
        violations.append("training receipt reports zero optimizer steps")

    valid = (
        adapter_verified
        and receipt_verified
        and bindings_verified
        and receipt.global_step > 0
        and not violations
    )
    return CyberSFTArtifactVerification(
        run_id=manifest.run_id,
        valid=valid,
        smoke_only=(manifest.corpus_promotion_eligible is False),
        adapter_digest_verified=adapter_verified,
        receipt_digest_verified=receipt_verified,
        receipt_bindings_verified=bindings_verified,
        global_step=receipt.global_step,
        violations=violations,
    )
