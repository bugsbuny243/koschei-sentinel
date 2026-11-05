from __future__ import annotations

import base64
import hashlib
import json
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_checkpoint_load_verify import (
    RankCheckpointLoadVerification,
    verify_rank_checkpoint_load,
)
from koschei_sentinel.production_checkpoint_manifest import RankCheckpointManifest
from koschei_sentinel.production_target_binding import (
    ProductionTargetBinding,
    require_verified_production_target_binding,
)
from koschei_sentinel.production_target_provenance import (
    ProductionTargetProvenance,
    owner_key_fingerprint,
    production_target_binding_digest,
    verify_production_target_provenance,
)

_DIGEST = r"^[a-f0-9]{64}$"
_SIGNATURE_CONTEXT = b"koschei-sentinel-approved-rank-checkpoint-v1\\0"


def _canonical_digest(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def checkpoint_approval_signature_message(receipt_sha256: str) -> bytes:
    if len(receipt_sha256) != 64 or any(c not in "0123456789abcdef" for c in receipt_sha256):
        raise ValueError("checkpoint approval digest must be lowercase SHA-256 hex")
    return _SIGNATURE_CONTEXT + receipt_sha256.encode("ascii")


class ApprovedRankCheckpointReceipt(StrictModel):
    """Owner-signed file identity; a claim that must be checked against a trusted key."""

    schema_version: Literal["sentinel.approved-rank-checkpoint.v1"] = (
        "sentinel.approved-rank-checkpoint.v1"
    )
    model_ref: str = Field(min_length=1, max_length=512)
    model_revision: str = Field(pattern=_DIGEST)
    target_binding_sha256: str = Field(pattern=_DIGEST)
    target_provenance_sha256: str = Field(pattern=_DIGEST)
    initialization_manifest_sha256: str = Field(pattern=_DIGEST)
    pipeline_rank: int = Field(ge=0)
    tensor_rank: int = Field(ge=0)
    expert_rank: int = Field(ge=0)
    checkpoint_file_sha256: str = Field(pattern=_DIGEST)
    checkpoint_file_size_bytes: int = Field(gt=0)
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    receipt_sha256: str = Field(pattern=_DIGEST)
    owner_signature_b64: str = Field(min_length=1)

    @model_validator(mode="after")
    def verify_receipt_digest(self) -> ApprovedRankCheckpointReceipt:
        unsigned = self.model_dump(
            mode="json", exclude={"receipt_sha256", "owner_signature_b64"}
        )
        if self.receipt_sha256 != _canonical_digest(unsigned):
            raise ValueError("checkpoint approval receipt self-hash mismatch")
        return self


class ApprovedRankCheckpointLoadVerification(StrictModel):
    schema_version: Literal["sentinel.approved-rank-checkpoint-load.v1"] = (
        "sentinel.approved-rank-checkpoint-load.v1"
    )
    status: Literal["approved_metadata_verified"] = "approved_metadata_verified"
    model_ref: str
    model_revision: str
    checkpoint_file_sha256: str = Field(pattern=_DIGEST)
    receipt_sha256: str = Field(pattern=_DIGEST)
    metadata: RankCheckpointLoadVerification
    source_provenance_verified: Literal[True] = True
    owner_signature_verified: Literal[True] = True
    artifact_bytes_verified: Literal[True] = True
    execution_authorized: Literal[False] = False


def verify_approved_rank_checkpoint_load(
    manifest: RankCheckpointManifest,
    *,
    checkpoint_path: str,
    receipt: ApprovedRankCheckpointReceipt,
    target_binding: ProductionTargetBinding,
    target_provenance: ProductionTargetProvenance,
    trusted_owner_public_key: Ed25519PublicKey,
) -> ApprovedRankCheckpointLoadVerification:
    """Verify an approved model identity and file digest *before* tensor loading.

    The caller must supply a pinned, independently trusted owner public key.
    This function verifies metadata only and never grants execution authority.
    """
    require_verified_production_target_binding(target_binding)
    verify_production_target_provenance(
        target_provenance, target_binding, trusted_owner_public_key
    )

    if (receipt.model_ref, receipt.model_revision) != (
        target_binding.model_ref,
        target_binding.model_revision,
    ):
        raise ValueError("checkpoint approval model identity differs from verified target")
    if receipt.target_binding_sha256 != production_target_binding_digest(target_binding):
        raise ValueError("checkpoint approval target binding differs from verified target")
    if receipt.target_provenance_sha256 != target_provenance.provenance_sha256:
        raise ValueError("checkpoint approval source provenance differs from verified target")
    if receipt.initialization_manifest_sha256 != manifest.manifest_sha256:
        raise ValueError("checkpoint approval initialization manifest mismatch")
    if (
        receipt.pipeline_rank,
        receipt.tensor_rank,
        receipt.expert_rank,
    ) != (
        manifest.pipeline_rank,
        manifest.tensor_rank,
        manifest.expert_rank,
    ):
        raise ValueError("checkpoint approval rank identity mismatch")
    if receipt.owner_key_fingerprint != owner_key_fingerprint(trusted_owner_public_key):
        raise ValueError("checkpoint approval owner trust root mismatch")

    try:
        signature = base64.b64decode(receipt.owner_signature_b64, validate=True)
    except ValueError as exc:
        raise ValueError("checkpoint approval signature is invalid base64") from exc
    try:
        trusted_owner_public_key.verify(
            signature, checkpoint_approval_signature_message(receipt.receipt_sha256)
        )
    except InvalidSignature as exc:
        raise ValueError("checkpoint approval signature verification failed") from exc

    metadata = verify_rank_checkpoint_load(
        manifest,
        checkpoint_path=checkpoint_path,
        expected_file_sha256=receipt.checkpoint_file_sha256,
        expected_file_size_bytes=receipt.checkpoint_file_size_bytes,
    )
    return ApprovedRankCheckpointLoadVerification(
        model_ref=target_binding.model_ref,
        model_revision=target_binding.model_revision,
        checkpoint_file_sha256=receipt.checkpoint_file_sha256,
        receipt_sha256=receipt.receipt_sha256,
        metadata=metadata,
    )
