from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.cyber_sft_export_verify import verify_cyber_sft_export
from koschei_sentinel.cyber_sft_run_attestation import CyberSFTRunAttestation
from koschei_sentinel.cyber_sft_trainer import CyberSFTAdapterManifest
from koschei_sentinel.cyber_sft_training import (
    CyberSFTPlan,
    CyberSFTStage,
    load_cyber_sft_config,
)
from koschei_sentinel.cyber_sft_training_source import CyberSFTTrainingSourceBinding
from koschei_sentinel.defense_reflex_gold_release_audit import audit_gold_defense_release
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import canonical_json

_DIGEST = r"^[a-f0-9]{64}$"


class GoldCandidateTrainingBindingVerification(StrictModel):
    schema_version: Literal["sentinel.gold-candidate-training-binding-verification.v1"] = (
        "sentinel.gold-candidate-training-binding-verification.v1"
    )
    run_id: str | None
    adapter_digest: str | None = Field(default=None, pattern=_DIGEST)
    source_gold_audit_sha256: str | None = Field(default=None, pattern=_DIGEST)
    candidate_export_verification_sha256: str | None = Field(
        default=None,
        pattern=_DIGEST,
    )
    gold_train_examples_sha256: str | None = Field(default=None, pattern=_DIGEST)
    gold_train_manifest_sha256: str | None = Field(default=None, pattern=_DIGEST)
    gold_validation_examples_sha256: str | None = Field(default=None, pattern=_DIGEST)
    gold_validation_manifest_sha256: str | None = Field(default=None, pattern=_DIGEST)
    stage_verified: bool
    promotion_eligible_verified: bool
    explicit_validation_verified: bool
    exported_training_bytes_verified: bool
    training_hashes_verified: bool
    validation_hashes_verified: bool
    valid: bool
    violations: list[str]
    verification_sha256: str = Field(pattern=_DIGEST)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _stable_sha(value: StrictModel) -> str:
    return _sha256_bytes(canonical_json(value.model_dump(mode="json")).encode("utf-8"))


def _verification_digest(payload: dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("verification_sha256", None)
    return _sha256_bytes(canonical_json(unsigned).encode("utf-8"))


def verify_gold_candidate_training_binding(
    gold_release_dir: str | Path,
    candidate_export_dir: str | Path,
) -> GoldCandidateTrainingBindingVerification:
    release = Path(gold_release_dir)
    export = Path(candidate_export_dir)
    violations: list[str] = []

    run_id: str | None = None
    adapter_digest: str | None = None
    source_gold_audit_sha256: str | None = None
    export_verification_sha256: str | None = None
    gold_train_examples_sha256: str | None = None
    gold_train_manifest_sha256: str | None = None
    gold_validation_examples_sha256: str | None = None
    gold_validation_manifest_sha256: str | None = None
    stage_verified = False
    promotion_eligible_verified = False
    explicit_validation_verified = False
    exported_training_bytes_verified = False
    training_hashes_verified = False
    validation_hashes_verified = False

    try:
        release_audit = audit_gold_defense_release(release)
        if not release_audit.valid:
            raise ValueError("Gold release structural audit is invalid")
        source_gold_audit_sha256 = release_audit.audit_sha256

        export_verification = verify_cyber_sft_export(export)
        export_verification_sha256 = _stable_sha(export_verification)
        run_id = export_verification.run_id
        if not export_verification.valid:
            detail = "; ".join(export_verification.violations[:5])
            raise ValueError(
                "candidate export verification is invalid"
                + (f": {detail}" if detail else "")
            )

        config = load_cyber_sft_config(export / "training-config.json")
        plan = CyberSFTPlan.model_validate_json((export / "training-plan.json").read_bytes())
        source = CyberSFTTrainingSourceBinding.model_validate_json(
            (export / "training-source.json").read_bytes()
        )
        attestation = CyberSFTRunAttestation.model_validate_json(
            (export / "run-attestation.json").read_bytes()
        )
        adapter_manifest = CyberSFTAdapterManifest.model_validate_json(
            (export / "run" / "adapter-manifest.json").read_bytes()
        )
        adapter_digest = adapter_manifest.adapter_digest
        run_id = config.run_id

        stage_verified = (
            config.stage is CyberSFTStage.DEFENSE_REFLEX
            and plan.stage is CyberSFTStage.DEFENSE_REFLEX
        )
        if not stage_verified:
            violations.append("Gold candidate must be a Defense Reflex SFT run")

        promotion_eligible_verified = (
            plan.corpus_promotion_eligible is True
            and source.run_id == config.run_id
            and attestation.promotion_eligible is True
            and attestation.smoke_only is False
            and adapter_manifest.corpus_promotion_eligible is True
        )
        if not promotion_eligible_verified:
            violations.append("Gold candidate training provenance is not promotion-eligible")

        explicit_validation_verified = (
            config.validation_corpus_dir is not None
            and config.validation_ratio == 0.0
            and plan.explicit_validation is True
            and source.explicit_validation is True
            and plan.validation_examples > 0
            and plan.validation_corpus_examples_sha256 is not None
            and plan.validation_corpus_manifest_sha256 is not None
            and source.validation_corpus_examples_sha256 is not None
            and source.validation_corpus_manifest_sha256 is not None
        )
        if not explicit_validation_verified:
            violations.append(
                "Gold candidate requires non-empty explicit validation with validation_ratio=0.0"
            )

        train_examples = release / "train" / "examples.jsonl"
        train_manifest = release / "train" / "manifest.json"
        validation_examples = release / "validation" / "examples.jsonl"
        validation_manifest = release / "validation" / "manifest.json"
        gold_train_examples_sha256 = _sha256_file(train_examples)
        gold_train_manifest_sha256 = _sha256_file(train_manifest)
        gold_validation_examples_sha256 = _sha256_file(validation_examples)
        gold_validation_manifest_sha256 = _sha256_file(validation_manifest)

        exported_training_bytes_verified = (
            _sha256_file(export / "corpus-examples.jsonl") == gold_train_examples_sha256
            and _sha256_file(export / "corpus-manifest.json") == gold_train_manifest_sha256
        )
        if not exported_training_bytes_verified:
            violations.append(
                "candidate export training corpus bytes differ from Gold TRAIN split"
            )

        training_hashes_verified = (
            plan.corpus_examples_sha256 == gold_train_examples_sha256
            and plan.corpus_manifest_sha256 == gold_train_manifest_sha256
            and source.corpus_examples_sha256 == gold_train_examples_sha256
            and source.corpus_manifest_sha256 == gold_train_manifest_sha256
            and attestation.corpus_examples_sha256 == gold_train_examples_sha256
            and attestation.corpus_manifest_sha256 == gold_train_manifest_sha256
        )
        if not training_hashes_verified:
            violations.append(
                "candidate plan/source/attestation do not bind the Gold TRAIN split"
            )

        validation_hashes_verified = (
            plan.validation_corpus_examples_sha256 == gold_validation_examples_sha256
            and plan.validation_corpus_manifest_sha256 == gold_validation_manifest_sha256
            and source.validation_corpus_examples_sha256 == gold_validation_examples_sha256
            and source.validation_corpus_manifest_sha256 == gold_validation_manifest_sha256
        )
        if not validation_hashes_verified:
            violations.append(
                "candidate plan/source do not bind the Gold VALIDATION split"
            )
    except (OSError, TypeError, ValueError) as exc:
        violations.append(str(exc))

    valid = (
        stage_verified
        and promotion_eligible_verified
        and explicit_validation_verified
        and exported_training_bytes_verified
        and training_hashes_verified
        and validation_hashes_verified
        and not violations
    )
    payload: dict[str, object] = {
        "schema_version": "sentinel.gold-candidate-training-binding-verification.v1",
        "run_id": run_id,
        "adapter_digest": adapter_digest,
        "source_gold_audit_sha256": source_gold_audit_sha256,
        "candidate_export_verification_sha256": export_verification_sha256,
        "gold_train_examples_sha256": gold_train_examples_sha256,
        "gold_train_manifest_sha256": gold_train_manifest_sha256,
        "gold_validation_examples_sha256": gold_validation_examples_sha256,
        "gold_validation_manifest_sha256": gold_validation_manifest_sha256,
        "stage_verified": stage_verified,
        "promotion_eligible_verified": promotion_eligible_verified,
        "explicit_validation_verified": explicit_validation_verified,
        "exported_training_bytes_verified": exported_training_bytes_verified,
        "training_hashes_verified": training_hashes_verified,
        "validation_hashes_verified": validation_hashes_verified,
        "valid": valid,
        "violations": violations,
    }
    payload["verification_sha256"] = _verification_digest(payload)
    return GoldCandidateTrainingBindingVerification.model_validate(payload)
