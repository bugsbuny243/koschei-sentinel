from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.promotion import public_key_fingerprint
from koschei_sentinel.training import atomic_write, canonical_json
from koschei_sentinel.web4_benchmark_intake import (
    Web4BenchmarkIntakePacket,
    Web4BenchmarkSplit,
)
from koschei_sentinel.web4_benchmark_review import (
    Web4BenchmarkAdjudication,
    Web4BenchmarkHumanReview,
    verify_web4_benchmark_adjudication,
)
from koschei_sentinel.web4_reviewer_trust import Web4ReviewerTrustPolicy

_DIGEST = r"^[a-f0-9]{64}$"
_RELEASE_ID = r"^[a-z0-9][a-z0-9._-]{2,127}$"
_SIGNATURE_CONTEXT = b"koschei-sentinel-web4-holdout-release-v1\0"


@dataclass(frozen=True)
class Web4HoldoutReleaseMaterial:
    packet: Web4BenchmarkIntakePacket
    review: Web4BenchmarkHumanReview
    adjudication: Web4BenchmarkAdjudication
    answer_key_path: Path
    reviewer_public_key: Ed25519PublicKey
    reviewer_trust_policy: Web4ReviewerTrustPolicy
    adjudicator_public_key: Ed25519PublicKey
    adjudicator_trust_policy: Web4ReviewerTrustPolicy


class Web4HoldoutReleaseCase(StrictModel):
    schema_version: Literal["sentinel.web4-holdout-release-case.v1"] = (
        "sentinel.web4-holdout-release-case.v1"
    )
    case_id: str = Field(min_length=3, max_length=256)
    family: str = Field(min_length=3, max_length=256)
    split: Literal["HOLDOUT"] = "HOLDOUT"
    created_at: str
    packet_sha256: str = Field(pattern=_DIGEST)
    model_input: dict[str, object]
    model_input_sha256: str = Field(pattern=_DIGEST)
    answer_key_sha256: str = Field(pattern=_DIGEST)
    intake_policy_sha256: str = Field(pattern=_DIGEST)
    split_seed: str
    split_material_sha256: str = Field(pattern=_DIGEST)
    source_refs: list[str]
    source_revision_status: dict[str, str]
    source_snapshot_sha256s: dict[str, str]
    review_sha256: str = Field(pattern=_DIGEST)
    review_artifact_sha256: str = Field(pattern=_DIGEST)
    reviewer_id: str
    reviewer_key_fingerprint: str = Field(pattern=_DIGEST)
    reviewer_trust_policy_digest: str = Field(pattern=_DIGEST)
    adjudication_sha256: str = Field(pattern=_DIGEST)
    adjudication_artifact_sha256: str = Field(pattern=_DIGEST)
    adjudicator_id: str
    adjudicator_key_fingerprint: str = Field(pattern=_DIGEST)
    adjudicator_trust_policy_digest: str = Field(pattern=_DIGEST)
    human_reviewed: Literal[True] = True
    independent_adjudication: Literal[True] = True
    final_review_approved: Literal[True] = True
    contains_answer_key: Literal[False] = False
    research_evaluation_authorization: Literal[True] = True
    training_authorization: Literal[False] = False
    promotion_eligible: Literal[False] = False
    production_activation_allowed: Literal[False] = False
    case_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def case_self_hash_verifies(self) -> Web4HoldoutReleaseCase:
        if set(self.source_refs) != set(self.source_revision_status):
            raise ValueError("Web4 HOLDOUT release source revision bindings differ")
        if set(self.source_refs) != set(self.source_snapshot_sha256s):
            raise ValueError("Web4 HOLDOUT release source snapshot bindings differ")
        payload = self.model_dump(mode="json")
        observed = str(payload.pop("case_sha256"))
        if _digest(payload) != observed:
            raise ValueError("Web4 HOLDOUT release case self-hash does not verify")
        return self


class Web4HoldoutRelease(StrictModel):
    schema_version: Literal["sentinel.web4-holdout-release.v1"] = (
        "sentinel.web4-holdout-release.v1"
    )
    release_id: str = Field(pattern=_RELEASE_ID)
    state: Literal["owner_signed_research_holdout"] = "owner_signed_research_holdout"
    benchmark_policy_sha256: str = Field(pattern=_DIGEST)
    source_registry_sha256: str = Field(pattern=_DIGEST)
    case_ids: list[str] = Field(min_length=1)
    cases: list[Web4HoldoutReleaseCase] = Field(min_length=1)
    case_count: int = Field(ge=1)
    answer_keys_isolated: Literal[True] = True
    all_cases_human_reviewed: Literal[True] = True
    all_cases_independently_adjudicated: Literal[True] = True
    deterministic_holdout_only: Literal[True] = True
    research_evaluation_authorization: Literal[True] = True
    training_authorization: Literal[False] = False
    promotion_eligible: Literal[False] = False
    production_activation_allowed: Literal[False] = False
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    release_sha256: str = Field(pattern=_DIGEST)
    signature_algorithm: Literal["ed25519"] = "ed25519"
    owner_signature_base64: str = Field(min_length=80, max_length=128)
    owner_signature_verified: Literal[True] = True
    artifact_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def release_contract_verifies(self) -> Web4HoldoutRelease:
        if self.case_count != len(self.cases):
            raise ValueError("Web4 HOLDOUT release case_count differs from cases")
        if self.case_ids != sorted(self.case_ids):
            raise ValueError("Web4 HOLDOUT release case_ids must be sorted")
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("Web4 HOLDOUT release contains duplicate case IDs")
        if self.case_ids != [case.case_id for case in self.cases]:
            raise ValueError("Web4 HOLDOUT release case_ids differ from case records")
        if any(case.split != "HOLDOUT" for case in self.cases):
            raise ValueError("Web4 HOLDOUT release contains a non-HOLDOUT case")
        if any(not case.research_evaluation_authorization for case in self.cases):
            raise ValueError("Web4 HOLDOUT release case lacks research evaluation authorization")
        payload = self.model_dump(mode="json")
        observed_artifact = str(payload.pop("artifact_sha256"))
        if _digest(payload) != observed_artifact:
            raise ValueError("Web4 HOLDOUT release artifact self-hash does not verify")
        signed = dict(payload)
        signed.pop("signature_algorithm")
        signed.pop("owner_signature_base64")
        signed.pop("owner_signature_verified")
        observed_release = str(signed.pop("release_sha256"))
        if _digest(signed) != observed_release:
            raise ValueError("Web4 HOLDOUT release digest does not verify")
        return self


def _digest(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _signature_message(release_sha256: str) -> bytes:
    return _SIGNATURE_CONTEXT + release_sha256.encode("ascii")


def _read_benchmark_policy(path: str | Path) -> tuple[dict[str, object], str]:
    policy_path = Path(path)
    if policy_path.is_symlink() or not policy_path.is_file():
        raise ValueError("Web4 benchmark policy must be a regular non-symlink file")
    raw = policy_path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid Web4 benchmark policy JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Web4 benchmark policy must contain one JSON object")
    if payload.get("schema_version") != "sentinel.web4-eval-policy.v1":
        raise ValueError("unsupported Web4 benchmark policy schema")
    if payload.get("status") != "research_only":
        raise ValueError("Web4 HOLDOUT release requires research-only benchmark policy")
    if payload.get("promotion_eligible") is not False:
        raise ValueError("Web4 benchmark policy must keep promotion disabled")
    if payload.get("training_overlap_allowed") is not False:
        raise ValueError("Web4 benchmark policy must forbid training overlap")
    if payload.get("requires_human_reviewed_holdout") is not True:
        raise ValueError("Web4 benchmark policy must require human-reviewed HOLDOUT")
    required_families = payload.get("required_families")
    if not isinstance(required_families, list) or not required_families:
        raise ValueError("Web4 benchmark policy required families are invalid")
    return payload, hashlib.sha256(raw).hexdigest()


def _case_payload(
    packet: Web4BenchmarkIntakePacket,
    review: Web4BenchmarkHumanReview,
    adjudication: Web4BenchmarkAdjudication,
) -> dict[str, object]:
    return {
        "schema_version": "sentinel.web4-holdout-release-case.v1",
        "case_id": packet.case_id,
        "family": packet.family,
        "split": "HOLDOUT",
        "created_at": packet.created_at,
        "packet_sha256": packet.packet_sha256,
        "model_input": packet.model_input,
        "model_input_sha256": packet.model_input_sha256,
        "answer_key_sha256": packet.answer_key_sha256,
        "intake_policy_sha256": packet.intake_policy_sha256,
        "split_seed": packet.split_seed,
        "split_material_sha256": packet.split_material_sha256,
        "source_refs": packet.source_refs,
        "source_revision_status": packet.source_revision_status,
        "source_snapshot_sha256s": packet.source_snapshot_sha256s,
        "review_sha256": review.review_sha256,
        "review_artifact_sha256": review.artifact_sha256,
        "reviewer_id": review.reviewer_id,
        "reviewer_key_fingerprint": review.reviewer_key_fingerprint,
        "reviewer_trust_policy_digest": review.reviewer_trust_policy_digest,
        "adjudication_sha256": adjudication.adjudication_sha256,
        "adjudication_artifact_sha256": adjudication.artifact_sha256,
        "adjudicator_id": adjudication.adjudicator_id,
        "adjudicator_key_fingerprint": adjudication.adjudicator_key_fingerprint,
        "adjudicator_trust_policy_digest": adjudication.adjudicator_trust_policy_digest,
        "human_reviewed": True,
        "independent_adjudication": True,
        "final_review_approved": True,
        "contains_answer_key": False,
        "research_evaluation_authorization": True,
        "training_authorization": False,
        "promotion_eligible": False,
        "production_activation_allowed": False,
    }


def _build_release_case(
    material: Web4HoldoutReleaseMaterial,
    owner_public_key: Ed25519PublicKey,
    required_families: set[str],
) -> Web4HoldoutReleaseCase:
    packet = Web4BenchmarkIntakePacket.model_validate(material.packet.model_dump(mode="json"))
    if packet.split is not Web4BenchmarkSplit.HOLDOUT:
        raise ValueError("Web4 signed HOLDOUT release rejects non-HOLDOUT intake packet")
    if packet.family not in required_families:
        raise ValueError("Web4 HOLDOUT release case family is absent from benchmark policy")
    adjudication = verify_web4_benchmark_adjudication(
        adjudication=material.adjudication,
        review=material.review,
        packet=packet,
        answer_key_path=material.answer_key_path,
        reviewer_public_key=material.reviewer_public_key,
        reviewer_trust_policy=material.reviewer_trust_policy,
        adjudicator_public_key=material.adjudicator_public_key,
        adjudicator_trust_policy=material.adjudicator_trust_policy,
        owner_public_key=owner_public_key,
    )
    review = Web4BenchmarkHumanReview.model_validate(material.review.model_dump(mode="json"))
    if not adjudication.final_review_approved:
        raise ValueError("Web4 signed HOLDOUT release requires final approved review")
    if not adjudication.eligible_for_signed_holdout_release:
        raise ValueError("Web4 adjudication is not eligible for signed HOLDOUT release")
    payload = _case_payload(packet, review, adjudication)
    return Web4HoldoutReleaseCase.model_validate(
        {**payload, "case_sha256": _digest(payload)}
    )


def build_web4_holdout_release(
    *,
    release_id: str,
    materials: list[Web4HoldoutReleaseMaterial],
    benchmark_policy_path: str | Path,
    owner_private_key: Ed25519PrivateKey,
) -> Web4HoldoutRelease:
    if not materials:
        raise ValueError("Web4 HOLDOUT release requires at least one case")
    policy, benchmark_policy_sha = _read_benchmark_policy(benchmark_policy_path)
    required_families = {str(item) for item in policy["required_families"]}
    owner_public_key = owner_private_key.public_key()

    cases = [
        _build_release_case(material, owner_public_key, required_families)
        for material in materials
    ]
    cases.sort(key=lambda case: case.case_id)
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Web4 HOLDOUT release contains duplicate case IDs")

    packet_policy_shas = {material.packet.benchmark_policy_sha256 for material in materials}
    if packet_policy_shas != {benchmark_policy_sha}:
        raise ValueError("Web4 HOLDOUT packets do not bind supplied benchmark policy")
    source_registry_shas = {material.packet.source_registry_sha256 for material in materials}
    if len(source_registry_shas) != 1:
        raise ValueError("Web4 HOLDOUT release requires one source registry identity")

    payload: dict[str, object] = {
        "schema_version": "sentinel.web4-holdout-release.v1",
        "release_id": release_id,
        "state": "owner_signed_research_holdout",
        "benchmark_policy_sha256": benchmark_policy_sha,
        "source_registry_sha256": next(iter(source_registry_shas)),
        "case_ids": [case.case_id for case in cases],
        "cases": [case.model_dump(mode="json") for case in cases],
        "case_count": len(cases),
        "answer_keys_isolated": True,
        "all_cases_human_reviewed": True,
        "all_cases_independently_adjudicated": True,
        "deterministic_holdout_only": True,
        "research_evaluation_authorization": True,
        "training_authorization": False,
        "promotion_eligible": False,
        "production_activation_allowed": False,
        "owner_key_fingerprint": public_key_fingerprint(owner_public_key),
    }
    release_sha = _digest(payload)
    signature = owner_private_key.sign(_signature_message(release_sha))
    owner_public_key.verify(signature, _signature_message(release_sha))
    signed: dict[str, object] = {
        **payload,
        "release_sha256": release_sha,
        "signature_algorithm": "ed25519",
        "owner_signature_base64": base64.b64encode(signature).decode("ascii"),
        "owner_signature_verified": True,
    }
    signed["artifact_sha256"] = _digest(signed)
    release = Web4HoldoutRelease.model_validate(signed)
    return verify_web4_holdout_release(
        release,
        owner_public_key=owner_public_key,
        benchmark_policy_path=benchmark_policy_path,
    )


def verify_web4_holdout_release(
    release: Web4HoldoutRelease,
    *,
    owner_public_key: Ed25519PublicKey,
    benchmark_policy_path: str | Path,
    materials: list[Web4HoldoutReleaseMaterial] | None = None,
) -> Web4HoldoutRelease:
    release = Web4HoldoutRelease.model_validate(release.model_dump(mode="json"))
    _, benchmark_policy_sha = _read_benchmark_policy(benchmark_policy_path)
    if release.benchmark_policy_sha256 != benchmark_policy_sha:
        raise ValueError("Web4 HOLDOUT release does not bind supplied benchmark policy")
    if release.owner_key_fingerprint != public_key_fingerprint(owner_public_key):
        raise ValueError("Web4 HOLDOUT release owner key fingerprint differs")
    try:
        signature = base64.b64decode(release.owner_signature_base64, validate=True)
        owner_public_key.verify(signature, _signature_message(release.release_sha256))
    except (InvalidSignature, ValueError) as exc:
        raise ValueError("Web4 HOLDOUT release owner signature verification failed") from exc

    if materials is not None:
        policy, _ = _read_benchmark_policy(benchmark_policy_path)
        required_families = {str(item) for item in policy["required_families"]}
        rebuilt = [
            _build_release_case(material, owner_public_key, required_families)
            for material in materials
        ]
        rebuilt.sort(key=lambda case: case.case_id)
        if [case.model_dump(mode="json") for case in rebuilt] != [
            case.model_dump(mode="json") for case in release.cases
        ]:
            raise ValueError("Web4 HOLDOUT release cases differ from supplied review chain")
    return release


def write_web4_holdout_release(release: Web4HoldoutRelease, path: str | Path) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Web4 HOLDOUT release already exists: {destination}")
    payload = json.dumps(release.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    atomic_write(destination, payload)
