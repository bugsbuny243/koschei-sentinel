from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from koschei_sentinel.defense_reflex_gold_release_audit import (
    GoldDefenseReleaseAudit,
    audit_gold_defense_release,
)
from koschei_sentinel.gold_review_signing import (
    GoldReviewSignatureAudit,
    audit_gold_release_review_signatures,
)


@dataclass(frozen=True)
class GoldReleaseSnapshotVerification:
    release_audit: GoldDefenseReleaseAudit
    review_signature_audit: GoldReviewSignatureAudit


def snapshot_verified_gold_release(
    release_dir: str | Path,
    destination: str | Path,
    *,
    reviewer_public_key: Ed25519PublicKey,
    expected_release_audit_sha256: str,
    expected_review_signature_audit_sha256: str,
) -> tuple[Path, GoldReleaseSnapshotVerification]:
    source = Path(release_dir)
    snapshot = Path(destination)
    if snapshot.exists():
        raise FileExistsError(f"Gold release snapshot already exists: {snapshot}")
    try:
        shutil.copytree(source, snapshot, symlinks=True)
        release_audit = audit_gold_defense_release(snapshot)
        if not release_audit.valid:
            detail = "; ".join(release_audit.violations[:5])
            raise ValueError(
                "Gold release snapshot failed structural audit"
                + (f": {detail}" if detail else "")
            )
        if release_audit.audit_sha256 != expected_release_audit_sha256:
            raise ValueError("Gold release snapshot structural audit SHA differs from signed pack")

        signature_audit = audit_gold_release_review_signatures(
            snapshot,
            reviewer_public_key,
        )
        if not signature_audit.valid:
            detail = "; ".join(signature_audit.violations[:5])
            raise ValueError(
                "Gold release snapshot failed signed-review audit"
                + (f": {detail}" if detail else "")
            )
        if signature_audit.audit_sha256 != expected_review_signature_audit_sha256:
            raise ValueError("Gold release snapshot review-signature audit SHA differs from signed pack")

        return snapshot, GoldReleaseSnapshotVerification(
            release_audit=release_audit,
            review_signature_audit=signature_audit,
        )
    except (OSError, TypeError, ValueError):
        shutil.rmtree(snapshot, ignore_errors=True)
        raise
