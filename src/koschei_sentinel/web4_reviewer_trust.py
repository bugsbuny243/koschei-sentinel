from __future__ import annotations

import base64
import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import Field

from koschei_sentinel.models import StrictModel
from koschei_sentinel.promotion import load_owner_public_key, public_key_fingerprint
from koschei_sentinel.training import atomic_write, canonical_json

_DIGEST = r"^[a-f0-9]{64}$"
_POLICY_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_DELEGATE_ID = r"^[A-Za-z0-9][A-Za-z0-9._@-]{2,127}$"
_SIGNATURE_CONTEXT = b"koschei-sentinel-web4-reviewer-trust-v1\0"


class Web4ReviewRole(StrEnum):
    PRIMARY_REVIEWER = "PRIMARY_REVIEWER"
    ADJUDICATOR = "ADJUDICATOR"


class Web4ReviewerTrustPolicy(StrictModel):
    schema_version: Literal["sentinel.web4-reviewer-trust-policy.v1"] = (
        "sentinel.web4-reviewer-trust-policy.v1"
    )
    policy_id: str = Field(pattern=_POLICY_ID)
    state: Literal["active"] = "active"
    role: Web4ReviewRole
    delegate_id: str = Field(pattern=_DELEGATE_ID)
    authority: Literal["benchmark_review_only"] = "benchmark_review_only"
    delegate_key_fingerprint: str = Field(pattern=_DIGEST)
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    independent_adjudication_required: Literal[True] = True
    automatic_key_rotation_allowed: Literal[False] = False
    training_authorization_allowed: Literal[False] = False
    evaluation_authorization_allowed: Literal[False] = False
    production_deployment_allowed: Literal[False] = False
    policy_digest: str = Field(pattern=_DIGEST)
    signature_algorithm: Literal["ed25519"] = "ed25519"
    owner_signature_base64: str = Field(min_length=80, max_length=128)
    owner_signature_verified: Literal[True] = True


def _digest(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _policy_payload(
    *,
    policy_id: str,
    role: Web4ReviewRole,
    delegate_id: str,
    delegate_key_fingerprint: str,
    owner_key_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": "sentinel.web4-reviewer-trust-policy.v1",
        "policy_id": policy_id,
        "state": "active",
        "role": role.value,
        "delegate_id": delegate_id,
        "authority": "benchmark_review_only",
        "delegate_key_fingerprint": delegate_key_fingerprint,
        "owner_key_fingerprint": owner_key_fingerprint,
        "independent_adjudication_required": True,
        "automatic_key_rotation_allowed": False,
        "training_authorization_allowed": False,
        "evaluation_authorization_allowed": False,
        "production_deployment_allowed": False,
    }


def _signature_message(policy_digest: str) -> bytes:
    return _SIGNATURE_CONTEXT + policy_digest.encode("ascii")


def delegate_public_key_fingerprint(public_key: Ed25519PublicKey) -> str:
    return public_key_fingerprint(public_key)


def load_web4_review_private_key(path: str | Path) -> Ed25519PrivateKey:
    try:
        key = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("invalid Web4 review private key") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("Web4 review private key must be unencrypted Ed25519 PEM")
    return key


def load_web4_review_public_key(path: str | Path) -> Ed25519PublicKey:
    try:
        key = serialization.load_pem_public_key(Path(path).read_bytes())
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("invalid Web4 review public key") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("Web4 review public key must be Ed25519")
    return key


def build_web4_reviewer_trust_policy(
    delegate_public_key: Ed25519PublicKey,
    owner_private_key: Ed25519PrivateKey,
    *,
    policy_id: str,
    role: Web4ReviewRole,
    delegate_id: str,
) -> Web4ReviewerTrustPolicy:
    payload = _policy_payload(
        policy_id=policy_id,
        role=role,
        delegate_id=delegate_id,
        delegate_key_fingerprint=delegate_public_key_fingerprint(delegate_public_key),
        owner_key_fingerprint=public_key_fingerprint(owner_private_key.public_key()),
    )
    policy_digest = _digest(payload)
    signature = owner_private_key.sign(_signature_message(policy_digest))
    artifact = Web4ReviewerTrustPolicy.model_validate(
        {
            **payload,
            "policy_digest": policy_digest,
            "signature_algorithm": "ed25519",
            "owner_signature_base64": base64.b64encode(signature).decode("ascii"),
            "owner_signature_verified": True,
        }
    )
    return verify_web4_reviewer_trust_policy(
        artifact,
        delegate_public_key,
        owner_private_key.public_key(),
    )


def verify_web4_reviewer_trust_policy(
    policy: Web4ReviewerTrustPolicy,
    delegate_public_key: Ed25519PublicKey,
    owner_public_key: Ed25519PublicKey,
) -> Web4ReviewerTrustPolicy:
    expected = _policy_payload(
        policy_id=policy.policy_id,
        role=policy.role,
        delegate_id=policy.delegate_id,
        delegate_key_fingerprint=policy.delegate_key_fingerprint,
        owner_key_fingerprint=policy.owner_key_fingerprint,
    )
    if _digest(expected) != policy.policy_digest:
        raise ValueError("Web4 reviewer trust policy digest does not verify")

    owner_fingerprint = public_key_fingerprint(owner_public_key)
    if owner_fingerprint != policy.owner_key_fingerprint:
        raise ValueError("owner public key does not match Web4 reviewer trust policy")

    delegate_fingerprint = delegate_public_key_fingerprint(delegate_public_key)
    if delegate_fingerprint != policy.delegate_key_fingerprint:
        raise ValueError("delegate public key does not match owner-signed Web4 trust policy")

    try:
        signature = base64.b64decode(policy.owner_signature_base64, validate=True)
        owner_public_key.verify(signature, _signature_message(policy.policy_digest))
    except (InvalidSignature, ValueError) as exc:
        raise ValueError("Web4 reviewer trust policy owner signature verification failed") from exc
    return policy


def load_web4_reviewer_trust_policy(path: str | Path) -> Web4ReviewerTrustPolicy:
    try:
        return Web4ReviewerTrustPolicy.model_validate_json(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError("invalid Web4 reviewer trust policy") from exc


def load_trusted_web4_review_public_key(
    *,
    delegate_public_key_path: str | Path,
    trust_policy_path: str | Path,
    owner_public_key_path: str | Path,
    required_role: Web4ReviewRole,
) -> Ed25519PublicKey:
    delegate_public_key = load_web4_review_public_key(delegate_public_key_path)
    policy = load_web4_reviewer_trust_policy(trust_policy_path)
    owner_public_key = load_owner_public_key(owner_public_key_path)
    if policy.role is not required_role:
        raise ValueError(f"Web4 reviewer trust policy role must be {required_role.value}")
    verify_web4_reviewer_trust_policy(policy, delegate_public_key, owner_public_key)
    return delegate_public_key


def write_web4_reviewer_trust_policy(
    policy: Web4ReviewerTrustPolicy,
    path: str | Path,
) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Web4 reviewer trust policy already exists: {destination}")
    payload = json.dumps(policy.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    atomic_write(destination, payload)
