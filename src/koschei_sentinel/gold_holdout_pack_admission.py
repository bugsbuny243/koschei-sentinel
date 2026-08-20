from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from koschei_sentinel.gold_holdout_pack_preflight import (
    preflight_gold_holdout_inference_pack,
)
from koschei_sentinel.gold_holdout_pack_signing import (
    GoldHoldoutPackSignatureProof,
    load_gold_holdout_pack_signature,
    verify_gold_holdout_inference_pack_signature,
)
from koschei_sentinel.gold_reviewer_trust import load_trusted_reviewer_public_key
from koschei_sentinel.gold_review_signing import load_reviewer_public_key


@dataclass(frozen=True)
class GoldHoldoutPackAdmission:
    proof: GoldHoldoutPackSignatureProof
    reviewer_public_key: Ed25519PublicKey


def verify_admitted_gold_holdout_pack(
    admission: GoldHoldoutPackAdmission,
    inference_pack: str | Path,
) -> None:
    pack = Path(inference_pack)
    preflight_gold_holdout_inference_pack(pack)
    verify_gold_holdout_inference_pack_signature(
        admission.proof,
        pack / "manifest.json",
        admission.reviewer_public_key,
    )


def snapshot_admitted_gold_holdout_pack(
    admission: GoldHoldoutPackAdmission,
    inference_pack: str | Path,
    destination: str | Path,
) -> Path:
    source = Path(inference_pack)
    snapshot = Path(destination)
    if snapshot.exists():
        raise FileExistsError(f"Gold HOLDOUT pack snapshot already exists: {snapshot}")
    try:
        shutil.copytree(source, snapshot, symlinks=True)
        verify_admitted_gold_holdout_pack(admission, snapshot)
        return snapshot
    except (OSError, TypeError, ValueError):
        shutil.rmtree(snapshot, ignore_errors=True)
        raise


def admit_signed_gold_holdout_pack(
    *,
    inference_pack: str | Path,
    signature_path: str | Path,
    reviewer_public_key_path: str | Path,
) -> GoldHoldoutPackAdmission:
    """Low-level admission using an already-trusted reviewer public-key path."""
    proof = load_gold_holdout_pack_signature(signature_path)
    reviewer_public_key = load_reviewer_public_key(reviewer_public_key_path)
    admission = GoldHoldoutPackAdmission(
        proof=proof,
        reviewer_public_key=reviewer_public_key,
    )
    verify_admitted_gold_holdout_pack(admission, inference_pack)
    return admission


def admit_owner_trusted_signed_gold_holdout_pack(
    *,
    inference_pack: str | Path,
    signature_path: str | Path,
    reviewer_public_key_path: str | Path,
    reviewer_trust_policy_path: str | Path,
    owner_public_key_path: str | Path,
) -> GoldHoldoutPackAdmission:
    """Production admission anchored to an owner-signed reviewer trust policy."""
    proof = load_gold_holdout_pack_signature(signature_path)
    reviewer_public_key = load_trusted_reviewer_public_key(
        reviewer_public_key_path=reviewer_public_key_path,
        trust_policy_path=reviewer_trust_policy_path,
        owner_public_key_path=owner_public_key_path,
    )
    admission = GoldHoldoutPackAdmission(
        proof=proof,
        reviewer_public_key=reviewer_public_key,
    )
    verify_admitted_gold_holdout_pack(admission, inference_pack)
    return admission
