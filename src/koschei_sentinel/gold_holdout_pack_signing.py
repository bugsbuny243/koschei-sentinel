from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import Field

from koschei_sentinel.gold_holdout_evaluation import GoldHoldoutInferenceManifest
from koschei_sentinel.gold_review_signing import reviewer_public_key_fingerprint
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import canonical_json

_DIGEST = r"^[a-f0-9]{64}$"
_SIGNATURE_CONTEXT = b"koschei-sentinel-gold-holdout-pack-v1\0"


class GoldHoldoutPackSignatureProof(StrictModel):
    schema_version: Literal["sentinel.gold-holdout-pack-signature-proof.v1"] = (
        "sentinel.gold-holdout-pack-signature-proof.v1"
    )
    reviewer_key_fingerprint: str = Field(pattern=_DIGEST)
    source_gold_audit_sha256: str = Field(pattern=_DIGEST)
    review_signature_audit_sha256: str = Field(pattern=_DIGEST)
    inputs_sha256: str = Field(pattern=_DIGEST)
    inference_manifest_sha256: str = Field(pattern=_DIGEST)
    binding_sha256: str = Field(pattern=_DIGEST)
    signature_algorithm: Literal["ed25519"] = "ed25519"
    signature_base64: str = Field(min_length=80, max_length=128)
    signature_verified: Literal[True] = True
    proof_sha256: str = Field(pattern=_DIGEST)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(payload: dict[str, object], field_name: str | None = None) -> str:
    unsigned = dict(payload)
    if field_name is not None:
        unsigned.pop(field_name, None)
    return _sha256_bytes(canonical_json(unsigned).encode("utf-8"))


def _binding_payload(
    *,
    manifest: GoldHoldoutInferenceManifest,
    manifest_sha256: str,
    reviewer_key_fingerprint_value: str,
    review_signature_audit_sha256: str,
) -> dict[str, object]:
    return {
        "reviewer_key_fingerprint": reviewer_key_fingerprint_value,
        "source_gold_audit_sha256": manifest.source_gold_audit_sha256,
        "review_signature_audit_sha256": review_signature_audit_sha256,
        "inputs_sha256": manifest.inputs_sha256,
        "inference_manifest_sha256": manifest_sha256,
    }


def _signature_message(binding_sha256: str) -> bytes:
    return _SIGNATURE_CONTEXT + binding_sha256.encode("ascii")


def sign_gold_holdout_inference_pack(
    manifest_path: str | Path,
    reviewer_private_key: Ed25519PrivateKey,
    *,
    review_signature_audit_sha256: str,
) -> GoldHoldoutPackSignatureProof:
    if len(review_signature_audit_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in review_signature_audit_sha256
    ):
        raise ValueError("Gold HOLDOUT pack signing requires a valid review-signature audit SHA")
    raw = Path(manifest_path).read_bytes()
    manifest = GoldHoldoutInferenceManifest.model_validate_json(raw)
    fingerprint = reviewer_public_key_fingerprint(reviewer_private_key.public_key())
    manifest_sha = _sha256_bytes(raw)
    binding = _binding_payload(
        manifest=manifest,
        manifest_sha256=manifest_sha,
        reviewer_key_fingerprint_value=fingerprint,
        review_signature_audit_sha256=review_signature_audit_sha256,
    )
    binding_sha = _digest(binding)
    signature = reviewer_private_key.sign(_signature_message(binding_sha))
    reviewer_private_key.public_key().verify(signature, _signature_message(binding_sha))
    payload: dict[str, object] = {
        "schema_version": "sentinel.gold-holdout-pack-signature-proof.v1",
        **binding,
        "binding_sha256": binding_sha,
        "signature_algorithm": "ed25519",
        "signature_base64": base64.b64encode(signature).decode("ascii"),
        "signature_verified": True,
    }
    payload["proof_sha256"] = _digest(payload)
    return GoldHoldoutPackSignatureProof.model_validate(payload)


def verify_gold_holdout_inference_pack_signature(
    proof: GoldHoldoutPackSignatureProof,
    manifest_path: str | Path,
    reviewer_public_key: Ed25519PublicKey,
) -> None:
    payload = proof.model_dump(mode="json")
    if _digest(payload, "proof_sha256") != proof.proof_sha256:
        raise ValueError("Gold HOLDOUT pack signature proof self-hash does not verify")

    fingerprint = reviewer_public_key_fingerprint(reviewer_public_key)
    if proof.reviewer_key_fingerprint != fingerprint:
        raise ValueError("Gold HOLDOUT pack signature uses an untrusted reviewer key")

    raw = Path(manifest_path).read_bytes()
    manifest = GoldHoldoutInferenceManifest.model_validate_json(raw)
    manifest_sha = _sha256_bytes(raw)
    binding = _binding_payload(
        manifest=manifest,
        manifest_sha256=manifest_sha,
        reviewer_key_fingerprint_value=fingerprint,
        review_signature_audit_sha256=proof.review_signature_audit_sha256,
    )
    if _digest(binding) != proof.binding_sha256:
        raise ValueError(
            "Gold HOLDOUT pack signature proof does not bind this inference manifest"
        )

    observed = {
        "reviewer_key_fingerprint": proof.reviewer_key_fingerprint,
        "source_gold_audit_sha256": proof.source_gold_audit_sha256,
        "review_signature_audit_sha256": proof.review_signature_audit_sha256,
        "inputs_sha256": proof.inputs_sha256,
        "inference_manifest_sha256": proof.inference_manifest_sha256,
    }
    if observed != binding:
        raise ValueError("Gold HOLDOUT pack signature proof does not bind this inference manifest")

    try:
        signature = base64.b64decode(proof.signature_base64, validate=True)
        reviewer_public_key.verify(signature, _signature_message(proof.binding_sha256))
    except (InvalidSignature, ValueError) as exc:
        raise ValueError("Gold HOLDOUT pack Ed25519 signature verification failed") from exc


def load_gold_holdout_pack_signature(
    path: str | Path,
) -> GoldHoldoutPackSignatureProof:
    try:
        return GoldHoldoutPackSignatureProof.model_validate_json(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid Gold HOLDOUT pack signature proof: {path}") from exc
