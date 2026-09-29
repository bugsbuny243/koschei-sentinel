import pytest

from koschei_sentinel.production_evidence import (
    EvidenceArtifact,
    ProductionEvidenceManifest,
    REQUIRED_PRODUCTION_ARTIFACTS,
)


def manifest(*, failed=None):
    artifacts = tuple(
        EvidenceArtifact(kind, f"sha256:{kind}", "sentinel-lab", kind != failed)
        for kind in sorted(REQUIRED_PRODUCTION_ARTIFACTS)
    )
    return ProductionEvidenceManifest(
        commit_sha="abc123", corpus_id="sentinel-security", corpus_version="1",
        corpus_digest="sha256:corpus", engine_id="engine", engine_version="1",
        run_date="2026-09-29", conformance_report_digest="sha256:report",
        artifacts=artifacts,
    )


def test_complete_verified_manifest_passes():
    item = manifest()
    item.require_verified(REQUIRED_PRODUCTION_ARTIFACTS)
    assert item.digest().startswith("sha256:")


def test_unverified_required_artifact_fails_closed():
    item = manifest(failed="live-inference")
    with pytest.raises(ValueError, match="unverified artifacts"):
        item.require_verified(REQUIRED_PRODUCTION_ARTIFACTS)


def test_missing_required_artifact_fails_closed():
    item = manifest()
    reduced = ProductionEvidenceManifest(
        item.commit_sha, item.corpus_id, item.corpus_version, item.corpus_digest,
        item.engine_id, item.engine_version, item.run_date,
        item.conformance_report_digest,
        tuple(a for a in item.artifacts if a.kind != "ci"),
    )
    with pytest.raises(ValueError, match="missing artifacts"):
        reduced.require_verified(REQUIRED_PRODUCTION_ARTIFACTS)
