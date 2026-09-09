from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, TypeVar

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import Field, model_validator

from koschei_sentinel.cyber_range import CyberRangeScenario
from koschei_sentinel.defense_reflex_gold_queue import (
    GoldDefenseReviewPacket,
    GoldReviewSplit,
    GoldReviewSplitPolicy,
    _report_sha256,
    _split_basis,
    _split_for_basis,
    _split_policy_sha256,
)
from koschei_sentinel.defense_reflex_gold_release import _verify_review_binding
from koschei_sentinel.defense_reflex_gold_review import GoldReviewedPacket
from koschei_sentinel.gold_reviewer_trust import (
    GoldReviewerTrustPolicy,
    load_gold_reviewer_trust_policy,
    verify_gold_reviewer_trust_policy,
)
from koschei_sentinel.gold_review_signing import (
    GoldReviewSignatureProof,
    load_reviewer_public_key,
    reviewer_public_key_fingerprint,
    verify_gold_review_signature,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.promotion import load_owner_public_key
from koschei_sentinel.training import canonical_json

_DIGEST = r"^[a-f0-9]{64}$"
_PRODUCTION_MINIMUM_HOLDOUT_CASES = 50
_ModelT = TypeVar("_ModelT", bound=StrictModel)


def _sha256_canonical(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")


def _require_flat_json_directory(path: Path, label: str) -> list[Path]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a real directory, not a symlink")
    entries = sorted(path.iterdir(), key=lambda item: item.name)
    if not entries:
        raise ValueError(f"{label} is empty")
    files: list[Path] = []
    for entry in entries:
        if entry.is_symlink():
            raise ValueError(f"{label} must not contain symlinks: {entry.name}")
        if not entry.is_file():
            raise ValueError(f"{label} must contain only flat JSON files: {entry.name}")
        if entry.suffix != ".json":
            raise ValueError(f"{label} contains a non-JSON file: {entry.name}")
        files.append(entry)
    return files


def _load_model(path: Path, model_type: type[_ModelT], label: str) -> _ModelT:
    _require_regular_file(path, label)
    try:
        return model_type.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {path.name}") from exc


def _load_models_by_scenario(
    directory: Path,
    model_type: type[_ModelT],
    label: str,
) -> dict[str, _ModelT]:
    rows: dict[str, _ModelT] = {}
    for path in _require_flat_json_directory(directory, label):
        row = _load_model(path, model_type, label)
        scenario_id = getattr(row, "scenario_id", None)
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError(f"{label} row lacks scenario_id: {path.name}")
        if scenario_id in rows:
            raise ValueError(f"duplicate {label} scenario_id: {scenario_id}")
        rows[scenario_id] = row
    return rows


def _exact_scenario_set(
    scenarios: dict[str, CyberRangeScenario],
    packets: dict[str, GoldDefenseReviewPacket],
    reviews: dict[str, GoldReviewedPacket],
    signatures: dict[str, GoldReviewSignatureProof],
) -> list[str]:
    expected = set(scenarios)
    observed = {
        "packets": set(packets),
        "reviews": set(reviews),
        "signatures": set(signatures),
    }
    for label, scenario_ids in observed.items():
        if scenario_ids != expected:
            missing = sorted(expected - scenario_ids)
            extra = sorted(scenario_ids - expected)
            detail: list[str] = []
            if missing:
                detail.append("missing=" + ",".join(missing[:8]))
            if extra:
                detail.append("extra=" + ",".join(extra[:8]))
            raise ValueError(
                f"Gold capacity {label} scenario set differs from scenarios"
                + (f": {'; '.join(detail)}" if detail else "")
            )
    return sorted(expected)


class GoldReviewCapacityReport(StrictModel):
    schema_version: Literal["sentinel.gold-review-capacity.v1"] = (
        "sentinel.gold-review-capacity.v1"
    )
    split_policy_sha256: str = Field(pattern=_DIGEST)
    reviewer_trust_policy_digest: str = Field(pattern=_DIGEST)
    reviewer_key_fingerprint: str = Field(pattern=_DIGEST)
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    scenario_count: int = Field(gt=0)
    packet_count: int = Field(gt=0)
    reviewed_count: int = Field(gt=0)
    signature_count: int = Field(gt=0)
    train_count: int = Field(ge=0)
    validation_count: int = Field(ge=0)
    holdout_count: int = Field(ge=0)
    minimum_train_cases: int = Field(ge=1)
    minimum_validation_cases: int = Field(ge=1)
    minimum_holdout_cases: int = Field(ge=1)
    holdout_shortfall: int = Field(ge=0)
    train_scenario_ids: list[str]
    validation_scenario_ids: list[str]
    holdout_scenario_ids: list[str]
    all_scenario_ids_sha256: str = Field(pattern=_DIGEST)
    artifact_bindings_sha256: str = Field(pattern=_DIGEST)
    ready_for_release: bool
    blockers: list[str]
    capacity_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def capacity_contract_verifies(self) -> GoldReviewCapacityReport:
        split_lists = (
            self.train_scenario_ids,
            self.validation_scenario_ids,
            self.holdout_scenario_ids,
        )
        for rows in split_lists:
            if rows != sorted(rows):
                raise ValueError("Gold capacity scenario IDs must be sorted")
            if len(rows) != len(set(rows)):
                raise ValueError("Gold capacity split contains duplicate scenario IDs")
        train = set(self.train_scenario_ids)
        validation = set(self.validation_scenario_ids)
        holdout = set(self.holdout_scenario_ids)
        if train & validation or train & holdout or validation & holdout:
            raise ValueError("Gold capacity split scenario IDs must be disjoint")
        all_ids = sorted(train | validation | holdout)
        if self.scenario_count != len(all_ids):
            raise ValueError("Gold capacity scenario_count differs from split identities")
        if self.packet_count != self.scenario_count:
            raise ValueError("Gold capacity packet_count differs from scenario_count")
        if self.reviewed_count != self.scenario_count:
            raise ValueError("Gold capacity reviewed_count differs from scenario_count")
        if self.signature_count != self.scenario_count:
            raise ValueError("Gold capacity signature_count differs from scenario_count")
        if self.train_count != len(self.train_scenario_ids):
            raise ValueError("Gold capacity train_count differs from scenario identities")
        if self.validation_count != len(self.validation_scenario_ids):
            raise ValueError("Gold capacity validation_count differs from scenario identities")
        if self.holdout_count != len(self.holdout_scenario_ids):
            raise ValueError("Gold capacity holdout_count differs from scenario identities")
        expected_shortfall = max(0, self.minimum_holdout_cases - self.holdout_count)
        if self.holdout_shortfall != expected_shortfall:
            raise ValueError("Gold capacity holdout_shortfall does not verify")
        if self.all_scenario_ids_sha256 != _sha256_canonical(all_ids):
            raise ValueError("Gold capacity all_scenario_ids_sha256 does not verify")
        expected_ready = (
            self.train_count >= self.minimum_train_cases
            and self.validation_count >= self.minimum_validation_cases
            and self.holdout_count >= self.minimum_holdout_cases
            and not self.blockers
        )
        if self.ready_for_release != expected_ready:
            raise ValueError("Gold capacity readiness does not match required split capacity")
        unsigned = self.model_dump(mode="json")
        observed = str(unsigned.pop("capacity_sha256"))
        if _sha256_canonical(unsigned) != observed:
            raise ValueError("Gold capacity report self-hash does not verify")
        return self


def _evaluate_gold_review_capacity(
    *,
    scenarios: dict[str, CyberRangeScenario],
    packets: dict[str, GoldDefenseReviewPacket],
    reviews: dict[str, GoldReviewedPacket],
    signatures: dict[str, GoldReviewSignatureProof],
    split_policy: GoldReviewSplitPolicy,
    reviewer_public_key: Ed25519PublicKey,
    reviewer_trust_policy: GoldReviewerTrustPolicy,
    owner_public_key: Ed25519PublicKey,
    minimum_train_cases: int = 1,
    minimum_validation_cases: int = 1,
    minimum_holdout_cases: int = _PRODUCTION_MINIMUM_HOLDOUT_CASES,
) -> GoldReviewCapacityReport:
    if minimum_train_cases < 1 or minimum_validation_cases < 1 or minimum_holdout_cases < 1:
        raise ValueError("Gold capacity minimum split sizes must be positive")
    verify_gold_reviewer_trust_policy(
        reviewer_trust_policy,
        reviewer_public_key,
        owner_public_key,
    )
    reviewer_fingerprint = reviewer_public_key_fingerprint(reviewer_public_key)
    policy_sha = _split_policy_sha256(split_policy)
    scenario_ids = _exact_scenario_set(scenarios, packets, reviews, signatures)

    by_split: dict[GoldReviewSplit, list[str]] = {
        GoldReviewSplit.TRAIN: [],
        GoldReviewSplit.VALIDATION: [],
        GoldReviewSplit.HOLDOUT: [],
    }
    artifact_bindings: list[dict[str, str]] = []
    for scenario_id in scenario_ids:
        scenario = scenarios[scenario_id]
        packet = packets[scenario_id]
        reviewed = reviews[scenario_id]
        proof = signatures[scenario_id]

        _report, report_sha = _report_sha256(scenario)
        split_basis = _split_basis(scenario_id, report_sha, split_policy)
        split = _split_for_basis(split_basis, split_policy)
        if packet.split_policy_sha256 != policy_sha:
            raise ValueError(f"Gold capacity split policy mismatch: {scenario_id}")
        if packet.source_report_sha256 != report_sha:
            raise ValueError(f"Gold capacity source report drift: {scenario_id}")
        if packet.split_basis_sha256 != split_basis:
            raise ValueError(f"Gold capacity split basis drift: {scenario_id}")
        if packet.split is not split:
            raise ValueError(f"Gold capacity pre-review split drift: {scenario_id}")

        _verify_review_binding(scenario, packet, reviewed)
        verify_gold_review_signature(reviewed, proof, reviewer_public_key)
        by_split[split].append(scenario_id)
        artifact_bindings.append(
            {
                "scenario_id": scenario_id,
                "source_report_sha256": report_sha,
                "packet_sha256": packet.packet_sha256,
                "review_sha256": reviewed.review_sha256,
                "signature_proof_sha256": proof.proof_sha256,
                "split": split.value,
            }
        )

    for rows in by_split.values():
        rows.sort()
    blockers: list[str] = []
    train_count = len(by_split[GoldReviewSplit.TRAIN])
    validation_count = len(by_split[GoldReviewSplit.VALIDATION])
    holdout_count = len(by_split[GoldReviewSplit.HOLDOUT])
    if train_count < minimum_train_cases:
        blockers.append(
            f"Gold TRAIN capacity below minimum: {train_count} < {minimum_train_cases}"
        )
    if validation_count < minimum_validation_cases:
        blockers.append(
            "Gold VALIDATION capacity below minimum: "
            f"{validation_count} < {minimum_validation_cases}"
        )
    if holdout_count < minimum_holdout_cases:
        blockers.append(
            f"Gold HOLDOUT capacity below minimum: {holdout_count} < {minimum_holdout_cases}"
        )

    unsigned: dict[str, object] = {
        "schema_version": "sentinel.gold-review-capacity.v1",
        "split_policy_sha256": policy_sha,
        "reviewer_trust_policy_digest": reviewer_trust_policy.policy_digest,
        "reviewer_key_fingerprint": reviewer_fingerprint,
        "owner_key_fingerprint": reviewer_trust_policy.owner_key_fingerprint,
        "scenario_count": len(scenario_ids),
        "packet_count": len(packets),
        "reviewed_count": len(reviews),
        "signature_count": len(signatures),
        "train_count": train_count,
        "validation_count": validation_count,
        "holdout_count": holdout_count,
        "minimum_train_cases": minimum_train_cases,
        "minimum_validation_cases": minimum_validation_cases,
        "minimum_holdout_cases": minimum_holdout_cases,
        "holdout_shortfall": max(0, minimum_holdout_cases - holdout_count),
        "train_scenario_ids": by_split[GoldReviewSplit.TRAIN],
        "validation_scenario_ids": by_split[GoldReviewSplit.VALIDATION],
        "holdout_scenario_ids": by_split[GoldReviewSplit.HOLDOUT],
        "all_scenario_ids_sha256": _sha256_canonical(scenario_ids),
        "artifact_bindings_sha256": _sha256_canonical(artifact_bindings),
        "ready_for_release": not blockers,
        "blockers": blockers,
    }
    return GoldReviewCapacityReport(
        **unsigned,
        capacity_sha256=_sha256_canonical(unsigned),
    )


def build_gold_review_capacity_report(
    *,
    scenario_dir: str | Path,
    packet_dir: str | Path,
    review_dir: str | Path,
    signature_dir: str | Path,
    split_policy_path: str | Path,
    reviewer_public_key_path: str | Path,
    reviewer_trust_policy_path: str | Path,
    owner_public_key_path: str | Path,
    minimum_train_cases: int = 1,
    minimum_validation_cases: int = 1,
    minimum_holdout_cases: int = _PRODUCTION_MINIMUM_HOLDOUT_CASES,
) -> GoldReviewCapacityReport:
    if minimum_holdout_cases < _PRODUCTION_MINIMUM_HOLDOUT_CASES:
        raise ValueError("production Gold capacity requires at least 50 HOLDOUT cases")

    policy_path = Path(split_policy_path)
    reviewer_key_path = Path(reviewer_public_key_path)
    trust_policy_path = Path(reviewer_trust_policy_path)
    owner_key_path = Path(owner_public_key_path)
    for path, label in (
        (policy_path, "Gold split policy"),
        (reviewer_key_path, "Gold reviewer public key"),
        (trust_policy_path, "Gold reviewer trust policy"),
        (owner_key_path, "Gold owner public key"),
    ):
        _require_regular_file(path, label)

    split_policy = _load_model(policy_path, GoldReviewSplitPolicy, "Gold split policy")
    reviewer_public_key = load_reviewer_public_key(reviewer_key_path)
    reviewer_trust_policy = load_gold_reviewer_trust_policy(trust_policy_path)
    owner_public_key = load_owner_public_key(owner_key_path)

    return _evaluate_gold_review_capacity(
        scenarios=_load_models_by_scenario(
            Path(scenario_dir),
            CyberRangeScenario,
            "Gold scenarios",
        ),
        packets=_load_models_by_scenario(
            Path(packet_dir),
            GoldDefenseReviewPacket,
            "Gold review packets",
        ),
        reviews=_load_models_by_scenario(
            Path(review_dir),
            GoldReviewedPacket,
            "Gold reviewed packets",
        ),
        signatures=_load_models_by_scenario(
            Path(signature_dir),
            GoldReviewSignatureProof,
            "Gold review signatures",
        ),
        split_policy=split_policy,
        reviewer_public_key=reviewer_public_key,
        reviewer_trust_policy=reviewer_trust_policy,
        owner_public_key=owner_public_key,
        minimum_train_cases=minimum_train_cases,
        minimum_validation_cases=minimum_validation_cases,
        minimum_holdout_cases=minimum_holdout_cases,
    )
