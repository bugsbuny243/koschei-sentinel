from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EvidenceArtifact:
    kind: str
    digest: str
    source: str
    verified: bool


@dataclass(frozen=True)
class ProductionEvidenceManifest:
    commit_sha: str
    corpus_id: str
    corpus_version: str
    corpus_digest: str
    engine_id: str
    engine_version: str
    run_date: str
    conformance_report_digest: str
    artifacts: tuple[EvidenceArtifact, ...]
    schema_version: str = "sentinel.production-evidence.v1"

    def __post_init__(self) -> None:
        if not self.commit_sha.strip():
            raise ValueError("production evidence requires commit identity")
        if not self.artifacts:
            raise ValueError("production evidence requires artifacts")
        kinds = [artifact.kind for artifact in self.artifacts]
        if len(kinds) != len(set(kinds)):
            raise ValueError("production evidence artifact kinds must be unique")

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def require_verified(self, required_kinds: frozenset[str]) -> None:
        indexed = {artifact.kind: artifact for artifact in self.artifacts}
        missing = required_kinds - indexed.keys()
        if missing:
            raise ValueError(f"production evidence missing artifacts: {sorted(missing)!r}")
        failed = sorted(kind for kind in required_kinds if not indexed[kind].verified)
        if failed:
            raise ValueError(f"production evidence contains unverified artifacts: {failed!r}")


REQUIRED_PRODUCTION_ARTIFACTS = frozenset({
    "ci",
    "live-inference",
    "tenant-isolation",
    "recovery",
    "containment-effect",
    "conformance",
})
