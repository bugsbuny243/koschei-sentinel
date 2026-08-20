from __future__ import annotations

import shutil
from pathlib import Path

from koschei_sentinel.cyber_sft_export_verify import verify_cyber_sft_export


def snapshot_verified_cyber_sft_export(
    candidate_export: str | Path,
    destination: str | Path,
) -> Path:
    source = Path(candidate_export)
    snapshot = Path(destination)
    if snapshot.exists():
        raise FileExistsError(f"Cyber SFT candidate snapshot already exists: {snapshot}")
    try:
        shutil.copytree(source, snapshot, symlinks=True)
        report = verify_cyber_sft_export(snapshot)
        if not report.valid:
            detail = "; ".join(report.violations[:5])
            raise ValueError(
                "Cyber SFT candidate snapshot failed fresh verification"
                + (f": {detail}" if detail else "")
            )
        return snapshot
    except (OSError, TypeError, ValueError):
        shutil.rmtree(snapshot, ignore_errors=True)
        raise
