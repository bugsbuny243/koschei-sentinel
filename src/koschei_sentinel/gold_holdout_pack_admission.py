from __future__ import annotations

from pathlib import Path

from koschei_sentinel.gold_holdout_pack_preflight import (
    preflight_gold_holdout_inference_pack,
)
from koschei_sentinel.gold_holdout_pack_signing import (
    GoldHoldoutPackSignatureProof,
    load_gold_holdout_pack_signature,
    verify_gold_holdout_inference_pack_signature,
)
from koschei_sentinel.gold_review_signing import load_reviewer_public_key


def admit_signed_gold_holdout_pack(
    *,
    inference_pack: str | Path,
    signature_path: str | Path,
    reviewer_public_key_path: str | Path,
) -> GoldHoldoutPackSignatureProof:
    """Fail closed before any HOLDOUT consumer trusts pack contents."""
    pack = Path(inference_pack)
    preflight_gold_holdout_inference_pack(pack)
    proof = load_gold_holdout_pack_signature(signature_path)
    reviewer_public_key = load_reviewer_public_key(reviewer_public_key_path)
    verify_gold_holdout_inference_pack_signature(
        proof,
        pack / "manifest.json",
        reviewer_public_key,
    )
    return proof
