from __future__ import annotations

import shutil
from pathlib import Path

from koschei_sentinel.gold_holdout_inference_verify import (
    GoldHoldoutInferenceVerification,
    verify_gold_holdout_inference_output,
)


def copy_gold_holdout_inference_output_snapshot(
    output_dir: str | Path,
    destination: str | Path,
) -> Path:
    source = Path(output_dir)
    snapshot = Path(destination)
    if snapshot.exists():
        raise FileExistsError(f"Gold HOLDOUT output snapshot already exists: {snapshot}")
    try:
        shutil.copytree(source, snapshot, symlinks=True)
        return snapshot
    except (OSError, TypeError, ValueError):
        shutil.rmtree(snapshot, ignore_errors=True)
        raise


def snapshot_verified_gold_holdout_inference_output(
    output_dir: str | Path,
    destination: str | Path,
    *,
    inference_pack_dir: str | Path,
    candidate_export_dir: str | Path,
) -> tuple[Path, GoldHoldoutInferenceVerification]:
    snapshot = copy_gold_holdout_inference_output_snapshot(output_dir, destination)
    try:
        verification = verify_gold_holdout_inference_output(
            snapshot,
            inference_pack_dir,
            candidate_export_dir,
        )
        if not verification.valid:
            detail = "; ".join(verification.violations[:5])
            raise ValueError(
                "Gold HOLDOUT output snapshot failed offline verification"
                + (f": {detail}" if detail else "")
            )
        return snapshot, verification
    except (OSError, TypeError, ValueError):
        shutil.rmtree(snapshot, ignore_errors=True)
        raise
