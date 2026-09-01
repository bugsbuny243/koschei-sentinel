from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.cyber_range import CyberRangeScenario
from koschei_sentinel.defense_reflex_gold_capacity import (
    GoldReviewCapacityReport,
    _exact_scenario_set,
    _load_models_by_scenario,
    build_gold_review_capacity_report,
)
from koschei_sentinel.defense_reflex_gold_queue import GoldDefenseReviewPacket
from koschei_sentinel.defense_reflex_gold_release_audit import audit_gold_defense_release
from koschei_sentinel.defense_reflex_gold_review import GoldReviewedPacket
from koschei_sentinel.gold_reviewer_trust import (
    load_gold_reviewer_trust_policy,
    load_trusted_reviewer_public_key,
)
from koschei_sentinel.gold_review_signing import (
    GoldReviewSignatureProof,
    audit_gold_release_review_signatures,
    write_signed_gold_defense_release,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import atomic_write, canonical_json

_DIGEST = r"^[a-f0-9]{64}$"
_PRODUCTION_MINIMUM_HOLDOUT_CASES = 50


def _sha256_canonical(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class GoldProductionReleaseReceipt(StrictModel):
    schema_version: Literal["sentinel.gold-production-release-receipt.v1"] = (
        "sentinel.gold-production-release-receipt.v1"
    )
    split_policy_sha256: str = Field(pattern=_DIGEST)
    capacity_sha256: str = Field(pattern=_DIGEST)
    release_manifest_sha256: str = Field(pattern=_DIGEST)
    release_sha256: str = Field(pattern=_DIGEST)
    release_audit_sha256: str = Field(pattern=_DIGEST)
    review_signature_audit_sha256: str = Field(pattern=_DIGEST)
    reviewer_trust_policy_digest: str = Field(pattern=_DIGEST)
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    train_examples: int = Field(gt=0)
    validation_examples: int = Field(gt=0)
    holdout_cases: int = Field(ge=_PRODUCTION_MINIMUM_HOLDOUT_CASES)
    minimum_holdout_cases: Literal[50] = 50
    production_ready: Literal[True] = True
    receipt_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def receipt_contract_verifies(self) -> GoldProductionReleaseReceipt:
        unsigned = self.model_dump(mode="json")
        observed = str(unsigned.pop("receipt_sha256"))
        if _sha256_canonical(unsigned) != observed:
            raise ValueError("Gold production release receipt self-hash does not verify")
        return self


def _preflight_output_paths(
    *,
    output_dir: str | Path,
    receipt_output: str | Path | None,
) -> None:
    release_destination = Path(output_dir)
    if release_destination.exists():
        raise FileExistsError(
            f"Gold production release already exists: {release_destination}"
        )
    if receipt_output is None:
        return

    receipt_destination = Path(receipt_output)
    if receipt_destination.exists():
        raise FileExistsError(
            f"Gold production release receipt exists: {receipt_destination}"
        )
    release_resolved = release_destination.resolve(strict=False)
    receipt_resolved = receipt_destination.resolve(strict=False)
    if (
        receipt_resolved == release_resolved
        or release_resolved in receipt_resolved.parents
    ):
        raise ValueError("Gold production release receipt must be outside the release directory")


def _load_release_rows(
    *,
    scenario_dir: str | Path,
    packet_dir: str | Path,
    review_dir: str | Path,
    signature_dir: str | Path,
):
    scenarios = _load_models_by_scenario(
        Path(scenario_dir),
        CyberRangeScenario,
        "Gold scenarios",
    )
    packets = _load_models_by_scenario(
        Path(packet_dir),
        GoldDefenseReviewPacket,
        "Gold review packets",
    )
    reviews = _load_models_by_scenario(
        Path(review_dir),
        GoldReviewedPacket,
        "Gold reviewed packets",
    )
    signatures = _load_models_by_scenario(
        Path(signature_dir),
        GoldReviewSignatureProof,
        "Gold review signatures",
    )
    scenario_ids = _exact_scenario_set(scenarios, packets, reviews, signatures)
    rows = [
        (scenarios[scenario_id], packets[scenario_id], reviews[scenario_id])
        for scenario_id in scenario_ids
    ]
    proofs = [signatures[scenario_id] for scenario_id in scenario_ids]
    return rows, proofs


def _write_capacity_gated_release(
    *,
    capacity: GoldReviewCapacityReport,
    scenario_dir: str | Path,
    packet_dir: str | Path,
    review_dir: str | Path,
    signature_dir: str | Path,
    reviewer_public_key_path: str | Path,
    reviewer_trust_policy_path: str | Path,
    owner_public_key_path: str | Path,
    output_dir: str | Path,
) -> GoldProductionReleaseReceipt:
    if not capacity.ready_for_release:
        detail = "; ".join(capacity.blockers[:5])
        raise ValueError(
            "Gold production release capacity gate is not ready"
            + (f": {detail}" if detail else "")
        )
    if capacity.minimum_holdout_cases < _PRODUCTION_MINIMUM_HOLDOUT_CASES:
        raise ValueError("Gold production release capacity floor is below 50 HOLDOUT cases")

    reviewer_public_key = load_trusted_reviewer_public_key(
        reviewer_public_key_path=reviewer_public_key_path,
        trust_policy_path=reviewer_trust_policy_path,
        owner_public_key_path=owner_public_key_path,
    )
    trust_policy = load_gold_reviewer_trust_policy(reviewer_trust_policy_path)
    if trust_policy.policy_digest != capacity.reviewer_trust_policy_digest:
        raise ValueError("Gold production capacity trust policy digest drifted")
    if trust_policy.owner_key_fingerprint != capacity.owner_key_fingerprint:
        raise ValueError("Gold production capacity owner trust root drifted")

    rows, proofs = _load_release_rows(
        scenario_dir=scenario_dir,
        packet_dir=packet_dir,
        review_dir=review_dir,
        signature_dir=signature_dir,
    )
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Gold production release already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    staging_release = staging_parent / "release"
    try:
        manifest = write_signed_gold_defense_release(
            rows,
            proofs,
            staging_release,
            reviewer_public_key,
        )
        audit = audit_gold_defense_release(staging_release)
        if not audit.valid:
            detail = "; ".join(audit.violations[:5])
            raise ValueError(
                "Gold production release audit failed"
                + (f": {detail}" if detail else "")
            )
        signature_audit = audit_gold_release_review_signatures(
            staging_release,
            reviewer_public_key,
        )
        if not signature_audit.valid:
            detail = "; ".join(signature_audit.violations[:5])
            raise ValueError(
                "Gold production review-signature audit failed"
                + (f": {detail}" if detail else "")
            )
        count_checks = (
            ("TRAIN", manifest.train_examples, capacity.train_count),
            ("VALIDATION", manifest.validation_examples, capacity.validation_count),
            ("HOLDOUT", manifest.holdout_cases, capacity.holdout_count),
        )
        for label, observed, expected in count_checks:
            if observed != expected:
                raise ValueError(
                    f"Gold production {label} count differs from capacity receipt"
                )
        if manifest.holdout_cases < _PRODUCTION_MINIMUM_HOLDOUT_CASES:
            raise ValueError("Gold production release contains fewer than 50 HOLDOUT cases")
        if manifest.split_policy_sha256 != capacity.split_policy_sha256:
            raise ValueError("Gold production release split policy differs from capacity receipt")
        if signature_audit.expected_review_count != capacity.scenario_count:
            raise ValueError("Gold production signature audit count differs from capacity receipt")
        if audit.release_manifest_sha256 is None:
            raise ValueError("Gold production release audit lacks manifest identity")

        unsigned: dict[str, object] = {
            "schema_version": "sentinel.gold-production-release-receipt.v1",
            "split_policy_sha256": capacity.split_policy_sha256,
            "capacity_sha256": capacity.capacity_sha256,
            "release_manifest_sha256": audit.release_manifest_sha256,
            "release_sha256": manifest.release_sha256,
            "release_audit_sha256": audit.audit_sha256,
            "review_signature_audit_sha256": signature_audit.audit_sha256,
            "reviewer_trust_policy_digest": capacity.reviewer_trust_policy_digest,
            "owner_key_fingerprint": capacity.owner_key_fingerprint,
            "train_examples": manifest.train_examples,
            "validation_examples": manifest.validation_examples,
            "holdout_cases": manifest.holdout_cases,
            "minimum_holdout_cases": 50,
            "production_ready": True,
        }
        receipt = GoldProductionReleaseReceipt(
            **unsigned,
            receipt_sha256=_sha256_canonical(unsigned),
        )
        os.replace(staging_release, destination)
        return receipt
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


def build_gold_production_release(
    *,
    scenario_dir: str | Path,
    packet_dir: str | Path,
    review_dir: str | Path,
    signature_dir: str | Path,
    split_policy_path: str | Path,
    reviewer_public_key_path: str | Path,
    reviewer_trust_policy_path: str | Path,
    owner_public_key_path: str | Path,
    output_dir: str | Path,
    receipt_output: str | Path | None = None,
) -> GoldProductionReleaseReceipt:
    _preflight_output_paths(
        output_dir=output_dir,
        receipt_output=receipt_output,
    )
    capacity = build_gold_review_capacity_report(
        scenario_dir=scenario_dir,
        packet_dir=packet_dir,
        review_dir=review_dir,
        signature_dir=signature_dir,
        split_policy_path=split_policy_path,
        reviewer_public_key_path=reviewer_public_key_path,
        reviewer_trust_policy_path=reviewer_trust_policy_path,
        owner_public_key_path=owner_public_key_path,
        minimum_train_cases=1,
        minimum_validation_cases=1,
        minimum_holdout_cases=50,
    )
    receipt = _write_capacity_gated_release(
        capacity=capacity,
        scenario_dir=scenario_dir,
        packet_dir=packet_dir,
        review_dir=review_dir,
        signature_dir=signature_dir,
        reviewer_public_key_path=reviewer_public_key_path,
        reviewer_trust_policy_path=reviewer_trust_policy_path,
        owner_public_key_path=owner_public_key_path,
        output_dir=output_dir,
    )
    if receipt_output is not None:
        destination = Path(receipt_output)
        atomic_write(
            destination,
            canonical_json(receipt.model_dump(mode="json")) + "\n",
        )
    return receipt
