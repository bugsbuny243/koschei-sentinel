from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_global_checkpoint import GlobalCheckpointIndex

_DIGEST = r"^[a-f0-9]{64}$"


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class MaterializedShardReceipt(StrictModel):
    model_shard_rank: int = Field(ge=0)
    relative_path: str = Field(min_length=1)
    initialization_manifest_sha256: str = Field(pattern=_DIGEST)
    file_sha256: str = Field(pattern=_DIGEST)
    file_size_bytes: int = Field(gt=0)


class GlobalCheckpointMaterializationReceipt(StrictModel):
    schema_version: Literal["sentinel.global-checkpoint-materialization.v1"] = (
        "sentinel.global-checkpoint-materialization.v1"
    )
    status: Literal["materialized"] = "materialized"
    source_index_sha256: str = Field(pattern=_DIGEST)
    shard_count: int = Field(gt=0)
    shards: list[MaterializedShardReceipt] = Field(min_length=1)
    fully_materialized: Literal[True] = True
    megatron_load_verified: Literal[False] = False
    execution_authorized: Literal[False] = False
    receipt_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def verify_receipt(self) -> "GlobalCheckpointMaterializationReceipt":
        if self.shard_count != len(self.shards):
            raise ValueError("shard_count disagrees with materialized shard receipts")
        if len({s.model_shard_rank for s in self.shards}) != self.shard_count:
            raise ValueError("duplicate materialized model_shard_rank")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if _digest(payload) != self.receipt_sha256:
            raise ValueError("global materialization receipt digest mismatch")
        return self


def finalize_global_checkpoint_materialization(
    index: GlobalCheckpointIndex,
    *,
    checkpoint_root: str | Path,
) -> GlobalCheckpointMaterializationReceipt:
    root = Path(checkpoint_root)
    if not root.exists() or not root.is_dir():
        raise ValueError("checkpoint_root must be an existing directory")

    receipts: list[dict[str, object]] = []
    for shard in index.shards:
        path = root / shard.relative_path
        if not path.is_file():
            raise ValueError(f"missing checkpoint shard: {shard.relative_path}")
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"empty checkpoint shard: {shard.relative_path}")
        receipts.append(
            {
                "model_shard_rank": shard.model_shard_rank,
                "relative_path": shard.relative_path,
                "initialization_manifest_sha256": shard.initialization_manifest_sha256,
                "file_sha256": _sha256_file(path),
                "file_size_bytes": size,
            }
        )

    payload = {
        "schema_version": "sentinel.global-checkpoint-materialization.v1",
        "status": "materialized",
        "source_index_sha256": index.index_sha256,
        "shard_count": len(receipts),
        "shards": receipts,
        "fully_materialized": True,
        "megatron_load_verified": False,
        "execution_authorized": False,
    }
    return GlobalCheckpointMaterializationReceipt.model_validate(
        {**payload, "receipt_sha256": _digest(payload)}
    )
