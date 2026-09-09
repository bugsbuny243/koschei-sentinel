from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.cyber_megatron_candidate import (
    CyberMegatronCandidateManifest,
    verify_cyber_megatron_candidate,
)
from koschei_sentinel.cyber_megatron_training import (
    QWEN35_397B_MODEL,
    QWEN35_397B_REVISION,
)
from koschei_sentinel.gold_holdout_inference_runner import (
    GoldHoldoutGenerationPolicy,
    _load_inference_pack,
    _policy_sha256,
)
from koschei_sentinel.gold_holdout_pack_admission import (
    admit_owner_trusted_signed_gold_holdout_pack,
)
from koschei_sentinel.gold_review_signing import reviewer_public_key_fingerprint
from koschei_sentinel.gold_reviewer_trust import load_gold_reviewer_trust_policy
from koschei_sentinel.models import StrictModel
from koschei_sentinel.promotion import load_owner_public_key, public_key_fingerprint
from koschei_sentinel.training import atomic_write, canonical_json, resolve_under_root

_DIGEST = r"^[a-f0-9]{64}$"
_PRODUCTION_MINIMUM_HOLDOUT_CASES = 50


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_canonical(payload: object) -> str:
    return _sha256_bytes(canonical_json(payload).encode("utf-8"))


def _read_regular_bytes(path: str | Path, label: str) -> bytes:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    try:
        return source.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {label}") from exc


class CyberMegatronHoldoutPlan(StrictModel):
    schema_version: Literal["sentinel.cyber-megatron-holdout-plan.v1"] = (
        "sentinel.cyber-megatron-holdout-plan.v1"
    )
    backend: Literal["megatron-swift-mcore"] = "megatron-swift-mcore"
    model: Literal["Qwen/Qwen3.5-397B-A17B"]
    model_revision: Literal["8472618112abcbd45acbcdc58436aff4233c23f7"]
    run_id: str
    candidate_manifest_sha256: str = Field(pattern=_DIGEST)
    candidate_sha256: str = Field(pattern=_DIGEST)
    checkpoint_tree_sha256: str = Field(pattern=_DIGEST)
    gold_release_audit_sha256: str = Field(pattern=_DIGEST)
    minimum_case_count: int = Field(ge=1)
    case_count: int = Field(gt=0)
    case_ids: list[str] = Field(min_length=1)
    case_ids_sha256: str = Field(pattern=_DIGEST)
    inputs_sha256: str = Field(pattern=_DIGEST)
    inference_manifest_sha256: str = Field(pattern=_DIGEST)
    pack_signature_file_sha256: str = Field(pattern=_DIGEST)
    pack_proof_sha256: str = Field(pattern=_DIGEST)
    review_signature_audit_sha256: str = Field(pattern=_DIGEST)
    reviewer_key_fingerprint: str = Field(pattern=_DIGEST)
    reviewer_trust_policy_file_sha256: str = Field(pattern=_DIGEST)
    reviewer_trust_policy_digest: str = Field(pattern=_DIGEST)
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    generation_policy: GoldHoldoutGenerationPolicy
    generation_policy_sha256: str = Field(pattern=_DIGEST)
    answer_key_isolated: Literal[True] = True
    deterministic_generation: Literal[True] = True
    prediction_identity_digest: str = Field(pattern=_DIGEST)
    plan_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def plan_contract_verifies(self) -> CyberMegatronHoldoutPlan:
        if self.case_ids != sorted(self.case_ids):
            raise ValueError("397B HOLDOUT case_ids must be sorted")
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("397B HOLDOUT case_ids must be unique")
        if self.case_count != len(self.case_ids):
            raise ValueError("397B HOLDOUT case_count differs from case_ids")
        if self.case_count < self.minimum_case_count:
            raise ValueError("397B HOLDOUT case_count is below the bound minimum")
        if _sha256_canonical(self.case_ids) != self.case_ids_sha256:
            raise ValueError("397B HOLDOUT case_ids_sha256 does not verify")
        if _policy_sha256(self.generation_policy) != self.generation_policy_sha256:
            raise ValueError("397B HOLDOUT generation policy SHA does not verify")
        if self.prediction_identity_digest != self.checkpoint_tree_sha256:
            raise ValueError(
                "397B HOLDOUT prediction identity must equal checkpoint tree identity"
            )
        unsigned = self.model_dump(mode="json")
        observed = str(unsigned.pop("plan_sha256"))
        if _sha256_canonical(unsigned) != observed:
            raise ValueError("397B HOLDOUT plan_sha256 does not verify")
        return self


class CyberMegatronHoldoutPlanVerification(StrictModel):
    valid: bool
    plan: CyberMegatronHoldoutPlan | None
    plan_file_sha256: str | None = Field(default=None, pattern=_DIGEST)
    violations: list[str]


def _expected_holdout_plan(
    *,
    root: Path,
    candidate_manifest_path: Path,
    candidate_config_path: Path,
    candidate_training_plan_path: Path,
    checkpoint_dir: Path,
    inference_pack: Path,
    signature_path: Path,
    reviewer_public_key_path: Path,
    reviewer_trust_policy_path: Path,
    owner_public_key_path: Path,
    generation_policy: GoldHoldoutGenerationPolicy,
    minimum_case_count: int,
) -> CyberMegatronHoldoutPlan:
    if minimum_case_count < 1:
        raise ValueError("397B HOLDOUT minimum_case_count must be positive")
    candidate_verification = verify_cyber_megatron_candidate(
        root=root,
        manifest_path=candidate_manifest_path,
        config_path=candidate_config_path,
        plan_path=candidate_training_plan_path,
        checkpoint_dir=checkpoint_dir,
    )
    if not candidate_verification.valid or candidate_verification.manifest is None:
        detail = "; ".join(candidate_verification.violations[:5])
        raise ValueError(
            "397B HOLDOUT requires a freshly verified Megatron candidate"
            + (f": {detail}" if detail else "")
        )
    candidate = candidate_verification.manifest
    if candidate.model != QWEN35_397B_MODEL:
        raise ValueError(
            "397B HOLDOUT candidate model is not the pinned external bootstrap candidate"
        )
    if candidate.model_revision != QWEN35_397B_REVISION:
        raise ValueError(
            "397B HOLDOUT candidate revision is not the pinned external bootstrap revision"
        )

    candidate_raw = _read_regular_bytes(
        candidate_manifest_path,
        "397B candidate manifest",
    )
    observed_candidate = CyberMegatronCandidateManifest.model_validate_json(candidate_raw)
    if observed_candidate != candidate:
        raise ValueError("397B candidate manifest differs from fresh candidate verification")

    signature_raw = _read_regular_bytes(
        signature_path,
        "Gold HOLDOUT pack signature proof",
    )
    trust_policy_raw = _read_regular_bytes(
        reviewer_trust_policy_path,
        "Gold reviewer trust policy",
    )
    _read_regular_bytes(reviewer_public_key_path, "Gold reviewer public key")
    _read_regular_bytes(owner_public_key_path, "Gold owner public key")

    admission = admit_owner_trusted_signed_gold_holdout_pack(
        inference_pack=inference_pack,
        signature_path=signature_path,
        reviewer_public_key_path=reviewer_public_key_path,
        reviewer_trust_policy_path=reviewer_trust_policy_path,
        owner_public_key_path=owner_public_key_path,
    )
    cases, inference_manifest, inference_manifest_raw = _load_inference_pack(inference_pack)
    trust_policy = load_gold_reviewer_trust_policy(reviewer_trust_policy_path)
    owner_public_key = load_owner_public_key(owner_public_key_path)

    reviewer_fingerprint = reviewer_public_key_fingerprint(admission.reviewer_public_key)
    owner_fingerprint = public_key_fingerprint(owner_public_key)
    proof = admission.proof

    if proof.reviewer_key_fingerprint != reviewer_fingerprint:
        raise ValueError("Gold HOLDOUT pack proof reviewer fingerprint changed after admission")
    if trust_policy.reviewer_key_fingerprint != reviewer_fingerprint:
        raise ValueError("Gold reviewer trust policy does not bind admitted reviewer key")
    if trust_policy.owner_key_fingerprint != owner_fingerprint:
        raise ValueError("Gold reviewer trust policy does not bind supplied owner root")

    if candidate.gold_release_audit_sha256 != inference_manifest.source_gold_audit_sha256:
        raise ValueError(
            "397B candidate Gold audit differs from signed HOLDOUT inference pack"
        )
    if proof.source_gold_audit_sha256 != candidate.gold_release_audit_sha256:
        raise ValueError("Gold HOLDOUT signature proof differs from candidate Gold audit")
    if proof.inputs_sha256 != inference_manifest.inputs_sha256:
        raise ValueError("Gold HOLDOUT signature proof differs from inference inputs")
    if proof.inference_manifest_sha256 != _sha256_bytes(inference_manifest_raw):
        raise ValueError("Gold HOLDOUT signature proof differs from inference manifest bytes")

    case_ids = [case.case_id for case in cases]
    if case_ids != inference_manifest.case_ids:
        raise ValueError("Gold HOLDOUT admitted case IDs differ from inference manifest")
    if len(case_ids) < minimum_case_count:
        raise ValueError(
            "397B HOLDOUT admitted pack is below minimum case count: "
            f"{len(case_ids)} < {minimum_case_count}"
        )

    unsigned: dict[str, object] = {
        "schema_version": "sentinel.cyber-megatron-holdout-plan.v1",
        "backend": "megatron-swift-mcore",
        "model": candidate.model,
        "model_revision": candidate.model_revision,
        "run_id": candidate.run_id,
        "candidate_manifest_sha256": _sha256_bytes(candidate_raw),
        "candidate_sha256": candidate.candidate_sha256,
        "checkpoint_tree_sha256": candidate.checkpoint_tree_sha256,
        "gold_release_audit_sha256": candidate.gold_release_audit_sha256,
        "minimum_case_count": minimum_case_count,
        "case_count": len(case_ids),
        "case_ids": case_ids,
        "case_ids_sha256": _sha256_canonical(case_ids),
        "inputs_sha256": inference_manifest.inputs_sha256,
        "inference_manifest_sha256": _sha256_bytes(inference_manifest_raw),
        "pack_signature_file_sha256": _sha256_bytes(signature_raw),
        "pack_proof_sha256": proof.proof_sha256,
        "review_signature_audit_sha256": proof.review_signature_audit_sha256,
        "reviewer_key_fingerprint": reviewer_fingerprint,
        "reviewer_trust_policy_file_sha256": _sha256_bytes(trust_policy_raw),
        "reviewer_trust_policy_digest": trust_policy.policy_digest,
        "owner_key_fingerprint": owner_fingerprint,
        "generation_policy": generation_policy.model_dump(mode="json"),
        "generation_policy_sha256": _policy_sha256(generation_policy),
        "answer_key_isolated": True,
        "deterministic_generation": True,
        "prediction_identity_digest": candidate.checkpoint_tree_sha256,
    }
    return CyberMegatronHoldoutPlan(
        **unsigned,
        plan_sha256=_sha256_canonical(unsigned),
    )


def _plan_text(plan: CyberMegatronHoldoutPlan) -> str:
    return json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def build_cyber_megatron_holdout_plan(
    *,
    candidate_manifest_path: str | Path,
    candidate_config_path: str | Path,
    candidate_training_plan_path: str | Path,
    checkpoint_dir: str | Path,
    inference_pack: str | Path,
    signature_path: str | Path,
    reviewer_public_key_path: str | Path,
    reviewer_trust_policy_path: str | Path,
    owner_public_key_path: str | Path,
    output_path: str | Path,
    generation_policy: GoldHoldoutGenerationPolicy | None = None,
    minimum_case_count: int = _PRODUCTION_MINIMUM_HOLDOUT_CASES,
    root: str | Path = ".",
) -> CyberMegatronHoldoutPlan:
    root_path = Path(root).resolve()
    destination = resolve_under_root(root_path, str(output_path))
    if destination.exists():
        raise FileExistsError(f"397B HOLDOUT plan already exists: {output_path}")
    selected_policy = generation_policy or GoldHoldoutGenerationPolicy()
    plan = _expected_holdout_plan(
        root=root_path,
        candidate_manifest_path=resolve_under_root(root_path, str(candidate_manifest_path)),
        candidate_config_path=resolve_under_root(root_path, str(candidate_config_path)),
        candidate_training_plan_path=resolve_under_root(
            root_path,
            str(candidate_training_plan_path),
        ),
        checkpoint_dir=resolve_under_root(root_path, str(checkpoint_dir)),
        inference_pack=Path(inference_pack),
        signature_path=Path(signature_path),
        reviewer_public_key_path=Path(reviewer_public_key_path),
        reviewer_trust_policy_path=Path(reviewer_trust_policy_path),
        owner_public_key_path=Path(owner_public_key_path),
        generation_policy=selected_policy,
        minimum_case_count=minimum_case_count,
    )
    atomic_write(destination, _plan_text(plan))
    return plan


def verify_cyber_megatron_holdout_plan(
    *,
    plan_path: str | Path,
    candidate_manifest_path: str | Path,
    candidate_config_path: str | Path,
    candidate_training_plan_path: str | Path,
    checkpoint_dir: str | Path,
    inference_pack: str | Path,
    signature_path: str | Path,
    reviewer_public_key_path: str | Path,
    reviewer_trust_policy_path: str | Path,
    owner_public_key_path: str | Path,
    generation_policy: GoldHoldoutGenerationPolicy | None = None,
    minimum_case_count: int = _PRODUCTION_MINIMUM_HOLDOUT_CASES,
    root: str | Path = ".",
) -> CyberMegatronHoldoutPlanVerification:
    root_path = Path(root).resolve()
    source = resolve_under_root(root_path, str(plan_path))
    if source.is_symlink() or not source.is_file():
        return CyberMegatronHoldoutPlanVerification(
            valid=False,
            plan=None,
            plan_file_sha256=None,
            violations=["397B HOLDOUT plan is missing or is a symlink"],
        )
    raw = source.read_bytes()
    try:
        observed = CyberMegatronHoldoutPlan.model_validate_json(raw)
    except ValueError as exc:
        return CyberMegatronHoldoutPlanVerification(
            valid=False,
            plan=None,
            plan_file_sha256=_sha256_bytes(raw),
            violations=[f"397B HOLDOUT plan cannot be verified: {exc}"],
        )

    violations: list[str] = []
    selected_policy = generation_policy or GoldHoldoutGenerationPolicy()
    if observed.minimum_case_count != minimum_case_count:
        violations.append("397B HOLDOUT plan minimum case floor differs from verifier policy")
    if observed.generation_policy != selected_policy:
        violations.append("397B HOLDOUT generation policy differs from verifier policy")
    if raw != _plan_text(observed).encode("utf-8"):
        violations.append("397B HOLDOUT plan is not canonical byte-for-byte")
    try:
        expected = _expected_holdout_plan(
            root=root_path,
            candidate_manifest_path=resolve_under_root(
                root_path,
                str(candidate_manifest_path),
            ),
            candidate_config_path=resolve_under_root(
                root_path,
                str(candidate_config_path),
            ),
            candidate_training_plan_path=resolve_under_root(
                root_path,
                str(candidate_training_plan_path),
            ),
            checkpoint_dir=resolve_under_root(root_path, str(checkpoint_dir)),
            inference_pack=Path(inference_pack),
            signature_path=Path(signature_path),
            reviewer_public_key_path=Path(reviewer_public_key_path),
            reviewer_trust_policy_path=Path(reviewer_trust_policy_path),
            owner_public_key_path=Path(owner_public_key_path),
            generation_policy=selected_policy,
            minimum_case_count=minimum_case_count,
        )
    except (OSError, TypeError, ValueError) as exc:
        violations.append(f"397B HOLDOUT source revalidation failed: {exc}")
    else:
        if observed != expected:
            violations.append(
                "397B HOLDOUT plan differs from freshly revalidated candidate/trust/input sources"
            )

    return CyberMegatronHoldoutPlanVerification(
        valid=not violations,
        plan=observed,
        plan_file_sha256=_sha256_bytes(raw),
        violations=violations,
    )
