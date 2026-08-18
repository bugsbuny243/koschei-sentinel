from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.defense_reflex_corpus_v3 import (
    DefenseReflexCorpusManifestV3,
    DefenseReflexTrainingExampleV3,
    DefenseReviewMethod,
    build_defense_reflex_v3_manifest,
    serialize_defense_reflex_v3,
)
from koschei_sentinel.defense_reflex_gold_release import (
    GoldDefenseReleaseManifest,
    GoldHoldoutEvaluationCase,
    GoldHoldoutManifest,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import canonical_json


class GoldDefenseReleaseAudit(StrictModel):
    schema_version: Literal["sentinel.gold-defense-release-audit.v1"] = (
        "sentinel.gold-defense-release-audit.v1"
    )
    release_dir: str
    release_manifest_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    train_examples: int = Field(ge=0)
    validation_examples: int = Field(ge=0)
    holdout_cases: int = Field(ge=0)
    train_validation_overlap: list[str]
    train_holdout_overlap: list[str]
    validation_holdout_overlap: list[str]
    hashes_verified: bool
    human_review_only_verified: bool
    holdout_isolation_verified: bool
    release_digest_verified: bool
    valid: bool
    violations: list[str]
    audit_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(payload: str) -> str:
    return _sha256_bytes(payload.encode("utf-8"))


def _audit_digest(payload: dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("audit_sha256", None)
    return _sha256_text(canonical_json(unsigned))


def _load_json_model(path: Path, model_type, label: str):
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {label}: {path}: {exc}") from exc
    try:
        return model_type.model_validate_json(raw), raw
    except ValueError as exc:
        raise ValueError(f"invalid {label}: {path}") from exc


def _load_training_split(
    split_dir: Path,
    *,
    label: str,
) -> tuple[list[DefenseReflexTrainingExampleV3], DefenseReflexCorpusManifestV3, bytes, list[str]]:
    violations: list[str] = []
    manifest, manifest_raw = _load_json_model(
        split_dir / "manifest.json",
        DefenseReflexCorpusManifestV3,
        f"{label} manifest",
    )
    try:
        payload = (split_dir / "examples.jsonl").read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read {label} examples: {exc}") from exc

    rows: list[DefenseReflexTrainingExampleV3] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(DefenseReflexTrainingExampleV3.model_validate_json(line))
        except ValueError:
            violations.append(f"{label} line {line_number} is not a Defense Reflex v3 example")

    if not rows:
        violations.append(f"{label} split is empty")
        return rows, manifest, manifest_raw, violations

    canonical_payload = serialize_defense_reflex_v3(rows)
    rebuilt = build_defense_reflex_v3_manifest(rows)
    if payload != canonical_payload:
        violations.append(f"{label} examples are not in canonical serialized order/format")
    if manifest.model_dump(mode="json") != rebuilt.model_dump(mode="json"):
        violations.append(f"{label} manifest does not match parsed examples")
    if manifest.examples_sha256 != _sha256_text(payload):
        violations.append(f"{label} examples SHA does not verify")
    if not manifest.promotion_eligible:
        violations.append(f"{label} manifest is not promotion-eligible")
    if manifest.synthetic_policy_reviewed_examples != 0:
        violations.append(f"{label} contains synthetic-policy reviewed examples")
    if manifest.human_reviewed_examples != len(rows):
        violations.append(f"{label} is not entirely human-reviewed")

    for row in rows:
        if row.provenance.get("review_method") != DefenseReviewMethod.HUMAN.value:
            violations.append(f"{label} example {row.example_id} is not human-reviewed")
        if not row.promotion_eligible:
            violations.append(f"{label} example {row.example_id} is not promotion-eligible")

    return rows, manifest, manifest_raw, violations


def _holdout_case_digest(case: GoldHoldoutEvaluationCase) -> str:
    payload = case.model_dump(mode="json")
    payload.pop("case_sha256", None)
    return _sha256_text(canonical_json(payload))


def _load_holdout(
    holdout_dir: Path,
) -> tuple[list[GoldHoldoutEvaluationCase], GoldHoldoutManifest, bytes, str, list[str]]:
    violations: list[str] = []
    manifest, manifest_raw = _load_json_model(
        holdout_dir / "manifest.json",
        GoldHoldoutManifest,
        "Gold holdout manifest",
    )
    try:
        payload = (holdout_dir / "cases.jsonl").read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read Gold holdout cases: {exc}") from exc

    cases: list[GoldHoldoutEvaluationCase] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = GoldHoldoutEvaluationCase.model_validate_json(line)
        except ValueError:
            violations.append(f"HOLDOUT line {line_number} is not a Gold evaluation case")
            continue
        cases.append(case)
        if _holdout_case_digest(case) != case.case_sha256:
            violations.append(f"HOLDOUT case {case.case_id} self-hash does not verify")
        if case.training_authorization:
            violations.append(f"HOLDOUT case {case.case_id} is training-authorized")
        if not case.evaluation_authorization:
            violations.append(f"HOLDOUT case {case.case_id} lacks evaluation authorization")
        try:
            DefenseReflexTrainingExampleV3.model_validate_json(line)
        except ValueError:
            pass
        else:
            violations.append(f"HOLDOUT case {case.case_id} is parseable as a training example")

    if not cases:
        violations.append("HOLDOUT split is empty")
    if manifest.case_count != len(cases):
        violations.append("HOLDOUT manifest case count does not match cases.jsonl")
    if manifest.cases_sha256 != _sha256_text(payload):
        violations.append("HOLDOUT cases SHA does not verify")
    if manifest.scenario_ids != sorted(row.scenario_id for row in cases):
        violations.append("HOLDOUT manifest scenario IDs do not match cases")
    if manifest.review_sha256s != sorted(row.review_sha256 for row in cases):
        violations.append("HOLDOUT manifest review SHA list does not match cases")
    if manifest.training_authorization:
        violations.append("HOLDOUT manifest is training-authorized")
    if not manifest.evaluation_ready:
        violations.append("HOLDOUT manifest is not evaluation-ready")
    if (holdout_dir / "examples.jsonl").exists():
        violations.append("HOLDOUT directory contains forbidden examples.jsonl")

    return cases, manifest, manifest_raw, payload, violations


def audit_gold_defense_release(
    release_dir: str | Path,
) -> GoldDefenseReleaseAudit:
    root = Path(release_dir)
    violations: list[str] = []
    release_manifest_sha: str | None = None
    train_rows: list[DefenseReflexTrainingExampleV3] = []
    validation_rows: list[DefenseReflexTrainingExampleV3] = []
    holdout_cases: list[GoldHoldoutEvaluationCase] = []
    hashes_verified = False
    human_verified = False
    holdout_verified = False
    release_digest_verified = False

    try:
        release_manifest, release_manifest_raw = _load_json_model(
            root / "release-manifest.json",
            GoldDefenseReleaseManifest,
            "Gold release manifest",
        )
        release_manifest_sha = _sha256_bytes(release_manifest_raw)
        train_rows, train_manifest, train_manifest_raw, train_violations = _load_training_split(
            root / "train",
            label="TRAIN",
        )
        validation_rows, validation_manifest, validation_manifest_raw, validation_violations = (
            _load_training_split(root / "validation", label="VALIDATION")
        )
        holdout_cases, holdout_manifest, holdout_manifest_raw, _holdout_payload, holdout_violations = (
            _load_holdout(root / "holdout")
        )
        violations.extend(train_violations)
        violations.extend(validation_violations)
        violations.extend(holdout_violations)

        manifest_hash_checks = (
            (
                "TRAIN manifest SHA",
                _sha256_bytes(train_manifest_raw),
                release_manifest.train_manifest_sha256,
            ),
            (
                "VALIDATION manifest SHA",
                _sha256_bytes(validation_manifest_raw),
                release_manifest.validation_manifest_sha256,
            ),
            (
                "HOLDOUT manifest SHA",
                _sha256_bytes(holdout_manifest_raw),
                release_manifest.holdout_manifest_sha256,
            ),
        )
        for label, observed, expected in manifest_hash_checks:
            if observed != expected:
                violations.append(f"{label} does not match release manifest")

        if release_manifest.train_examples != len(train_rows):
            violations.append("release TRAIN count does not match parsed examples")
        if release_manifest.validation_examples != len(validation_rows):
            violations.append("release VALIDATION count does not match parsed examples")
        if release_manifest.holdout_cases != len(holdout_cases):
            violations.append("release HOLDOUT count does not match parsed cases")
        if not release_manifest.human_review_only:
            violations.append("Gold release is not marked human-review-only")
        if release_manifest.holdout_training_authorization:
            violations.append("Gold release authorizes HOLDOUT for training")
        if not release_manifest.promotion_ready:
            violations.append("Gold release is not promotion-ready")

        release_payload = {
            "split_policy_sha256": release_manifest.split_policy_sha256,
            "train_manifest_sha256": release_manifest.train_manifest_sha256,
            "validation_manifest_sha256": release_manifest.validation_manifest_sha256,
            "holdout_manifest_sha256": release_manifest.holdout_manifest_sha256,
            "train_examples": release_manifest.train_examples,
            "validation_examples": release_manifest.validation_examples,
            "holdout_cases": release_manifest.holdout_cases,
        }
        expected_release_sha = _sha256_text(canonical_json(release_payload))
        release_digest_verified = expected_release_sha == release_manifest.release_sha256
        if not release_digest_verified:
            violations.append("Gold release self-digest does not verify")

        hashes_verified = not any("SHA" in row or "hash" in row for row in violations)
        human_verified = (
            train_manifest.human_reviewed_examples == len(train_rows)
            and validation_manifest.human_reviewed_examples == len(validation_rows)
            and train_manifest.synthetic_policy_reviewed_examples == 0
            and validation_manifest.synthetic_policy_reviewed_examples == 0
            and all(
                row.provenance.get("review_method") == DefenseReviewMethod.HUMAN.value
                and row.promotion_eligible
                for row in train_rows + validation_rows
            )
        )
        holdout_verified = (
            not holdout_manifest.training_authorization
            and holdout_manifest.evaluation_ready
            and all(
                not row.training_authorization and row.evaluation_authorization
                for row in holdout_cases
            )
            and not (root / "holdout" / "examples.jsonl").exists()
        )
    except (OSError, TypeError, ValueError) as exc:
        violations.append(str(exc))

    train_ids = {row.scenario_id for row in train_rows}
    validation_ids = {row.scenario_id for row in validation_rows}
    holdout_ids = {row.scenario_id for row in holdout_cases}
    train_validation_overlap = sorted(train_ids & validation_ids)
    train_holdout_overlap = sorted(train_ids & holdout_ids)
    validation_holdout_overlap = sorted(validation_ids & holdout_ids)
    if train_validation_overlap:
        violations.append("TRAIN and VALIDATION share scenario IDs")
    if train_holdout_overlap:
        violations.append("TRAIN and HOLDOUT share scenario IDs")
    if validation_holdout_overlap:
        violations.append("VALIDATION and HOLDOUT share scenario IDs")

    valid = (
        not violations
        and hashes_verified
        and human_verified
        and holdout_verified
        and release_digest_verified
        and bool(train_rows)
        and bool(validation_rows)
        and bool(holdout_cases)
    )
    payload: dict[str, object] = {
        "schema_version": "sentinel.gold-defense-release-audit.v1",
        "release_dir": str(release_dir),
        "release_manifest_sha256": release_manifest_sha,
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "holdout_cases": len(holdout_cases),
        "train_validation_overlap": train_validation_overlap,
        "train_holdout_overlap": train_holdout_overlap,
        "validation_holdout_overlap": validation_holdout_overlap,
        "hashes_verified": hashes_verified,
        "human_review_only_verified": human_verified,
        "holdout_isolation_verified": holdout_verified,
        "release_digest_verified": release_digest_verified,
        "valid": valid,
        "violations": violations,
    }
    payload["audit_sha256"] = _audit_digest(payload)
    return GoldDefenseReleaseAudit.model_validate(payload)
