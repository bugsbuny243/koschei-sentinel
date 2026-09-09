from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import Field

from koschei_sentinel.gold_review_signing import (
    load_reviewer_private_key,
    load_reviewer_public_key,
    reviewer_public_key_fingerprint,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.promotion import (
    load_owner_public_key,
    public_key_fingerprint,
)

_DIGEST = r"^[a-f0-9]{64}$"
_POLICY_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_SIGNATURE_CONTEXT = b"koschei-sentinel-gold-reviewer-trust-v1\0"


class GoldReviewerTrustPolicy(StrictModel):
    schema_version: Literal["sentinel.gold-reviewer-trust-policy.v1"] = (
        "sentinel.gold-reviewer-trust-policy.v1"
    )
    policy_id: str = Field(pattern=_POLICY_ID)
    state: Literal["active"] = "active"
    authority: Literal["gold_review_signing_only"] = "gold_review_signing_only"
    reviewer_key_fingerprint: str = Field(pattern=_DIGEST)
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    human_review_signature_required: Literal[True] = True
    holdout_pack_signature_required: Literal[True] = True
    automatic_key_rotation_allowed: Literal[False] = False
    production_deployment_allowed: Literal[False] = False
    policy_digest: str = Field(pattern=_DIGEST)
    signature_algorithm: Literal["ed25519"] = "ed25519"
    owner_signature_base64: str = Field(min_length=80, max_length=128)
    owner_signature_verified: Literal[True] = True


def _digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _policy_payload(
    *,
    policy_id: str,
    reviewer_key_fingerprint: str,
    owner_key_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": "sentinel.gold-reviewer-trust-policy.v1",
        "policy_id": policy_id,
        "state": "active",
        "authority": "gold_review_signing_only",
        "reviewer_key_fingerprint": reviewer_key_fingerprint,
        "owner_key_fingerprint": owner_key_fingerprint,
        "human_review_signature_required": True,
        "holdout_pack_signature_required": True,
        "automatic_key_rotation_allowed": False,
        "production_deployment_allowed": False,
    }


def _signature_message(policy_digest: str) -> bytes:
    return _SIGNATURE_CONTEXT + policy_digest.encode("ascii")


def build_gold_reviewer_trust_policy(
    reviewer_public_key: Ed25519PublicKey,
    owner_private_key: Ed25519PrivateKey,
    *,
    policy_id: str,
) -> GoldReviewerTrustPolicy:
    payload = _policy_payload(
        policy_id=policy_id,
        reviewer_key_fingerprint=reviewer_public_key_fingerprint(reviewer_public_key),
        owner_key_fingerprint=public_key_fingerprint(owner_private_key.public_key()),
    )
    policy_digest = _digest(payload)
    signature = owner_private_key.sign(_signature_message(policy_digest))
    artifact = GoldReviewerTrustPolicy.model_validate(
        {
            **payload,
            "policy_digest": policy_digest,
            "signature_algorithm": "ed25519",
            "owner_signature_base64": base64.b64encode(signature).decode("ascii"),
            "owner_signature_verified": True,
        }
    )
    return verify_gold_reviewer_trust_policy(
        artifact,
        reviewer_public_key,
        owner_private_key.public_key(),
    )


def verify_gold_reviewer_trust_policy(
    policy: GoldReviewerTrustPolicy,
    reviewer_public_key: Ed25519PublicKey,
    owner_public_key: Ed25519PublicKey,
) -> GoldReviewerTrustPolicy:
    expected_payload = _policy_payload(
        policy_id=policy.policy_id,
        reviewer_key_fingerprint=policy.reviewer_key_fingerprint,
        owner_key_fingerprint=policy.owner_key_fingerprint,
    )
    if _digest(expected_payload) != policy.policy_digest:
        raise ValueError("Gold reviewer trust policy digest does not verify")

    owner_fingerprint = public_key_fingerprint(owner_public_key)
    if owner_fingerprint != policy.owner_key_fingerprint:
        raise ValueError("owner public key does not match Gold reviewer trust policy")

    reviewer_fingerprint = reviewer_public_key_fingerprint(reviewer_public_key)
    if reviewer_fingerprint != policy.reviewer_key_fingerprint:
        raise ValueError("reviewer public key does not match owner-signed Gold trust policy")

    try:
        signature = base64.b64decode(policy.owner_signature_base64, validate=True)
        owner_public_key.verify(signature, _signature_message(policy.policy_digest))
    except (InvalidSignature, ValueError) as exc:
        raise ValueError("Gold reviewer trust policy owner signature verification failed") from exc
    return policy


def load_gold_reviewer_trust_policy(path: str | Path) -> GoldReviewerTrustPolicy:
    try:
        return GoldReviewerTrustPolicy.model_validate_json(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError("invalid Gold reviewer trust policy") from exc


def load_trusted_reviewer_public_key(
    *,
    reviewer_public_key_path: str | Path,
    trust_policy_path: str | Path,
    owner_public_key_path: str | Path,
) -> Ed25519PublicKey:
    reviewer_public_key = load_reviewer_public_key(reviewer_public_key_path)
    policy = load_gold_reviewer_trust_policy(trust_policy_path)
    owner_public_key = load_owner_public_key(owner_public_key_path)
    verify_gold_reviewer_trust_policy(
        policy,
        reviewer_public_key,
        owner_public_key,
    )
    return reviewer_public_key


def load_trusted_reviewer_private_key(
    *,
    reviewer_private_key_path: str | Path,
    trust_policy_path: str | Path,
    owner_public_key_path: str | Path,
) -> Ed25519PrivateKey:
    reviewer_private_key = load_reviewer_private_key(reviewer_private_key_path)
    policy = load_gold_reviewer_trust_policy(trust_policy_path)
    owner_public_key = load_owner_public_key(owner_public_key_path)
    verify_gold_reviewer_trust_policy(
        policy,
        reviewer_private_key.public_key(),
        owner_public_key,
    )
    return reviewer_private_key


def write_gold_reviewer_trust_policy(
    policy: GoldReviewerTrustPolicy,
    path: str | Path,
) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Gold reviewer trust policy already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(policy.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
