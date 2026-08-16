from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from koschei_sentinel.blockchain_security_corpus import (
    BlockchainSourceClass,
    ChainFamily,
    ThreatDomain,
)
from koschei_sentinel.blockchain_security_ingest import (
    BlockchainSourceSnapshotSpec,
    ingest_snapshot,
    inspect_snapshot,
    snapshot_digest,
    write_ingest_result,
    write_snapshot_spec,
)
from koschei_sentinel.pretraining_corpus import RightsBasis


def _spec(root: Path, snapshot: Path) -> BlockchainSourceSnapshotSpec:
    provisional = BlockchainSourceSnapshotSpec(
        source_id="fixture.protocol",
        source_class=BlockchainSourceClass.PROTOCOL_SOURCE,
        rights_basis=RightsBasis.APACHE_2_0,
        snapshot_path=snapshot.relative_to(root).as_posix(),
        expected_snapshot_digest="0" * 64,
        chain_families=[ChainFamily.EVM],
        threat_domains=[
            ThreatDomain.SMART_CONTRACT,
            ThreatDomain.PRIVILEGED_ACCESS,
        ],
        family_refs=[],
    )
    _, files = inspect_snapshot(provisional, root=root)
    return provisional.model_copy(
        update={"expected_snapshot_digest": snapshot_digest(files)}
    )


def _snapshot(root: Path) -> Path:
    snapshot = root / "snapshot"
    snapshot.mkdir()
    (snapshot / "README.md").write_text(
        "Protocol security architecture and privileged upgrade boundary.\n",
        encoding="utf-8",
    )
    contracts = snapshot / "contracts"
    contracts.mkdir()
    (contracts / "Vault.sol").write_text(
        "contract Vault { function version() public pure returns (uint) { return 1; } }\n",
        encoding="utf-8",
    )
    hidden = snapshot / ".git"
    hidden.mkdir()
    (hidden / "config").write_text("ignored", encoding="utf-8")
    return snapshot


def test_ingest_is_content_addressed_and_deterministic(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    spec = _spec(tmp_path, snapshot)

    first = ingest_snapshot(spec, root=tmp_path)
    second = ingest_snapshot(spec, root=tmp_path)

    assert first == second
    assert first.manifest.snapshot_digest == spec.expected_snapshot_digest
    assert first.manifest.documents == 2
    assert len(first.manifest.files) == 2
    assert all(
        item.source_snapshot_digest == spec.expected_snapshot_digest
        for item in first.documents
    )
    assert all(item.chain_families == [ChainFamily.EVM] for item in first.documents)


def test_snapshot_drift_after_pinning_is_rejected(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    spec = _spec(tmp_path, snapshot)
    (snapshot / "README.md").write_text(
        "Changed after trust ceremony.\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="trusted expected digest"):
        ingest_snapshot(spec, root=tmp_path)


def test_duplicate_content_cannot_inflate_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    text = "Same security material.\n"
    (snapshot / "one.md").write_text(text, encoding="utf-8")
    (snapshot / "two.md").write_text(text, encoding="utf-8")
    spec = _spec(tmp_path, snapshot)

    with pytest.raises(ValueError, match="duplicate content"):
        ingest_snapshot(spec, root=tmp_path)


def test_sensitive_material_is_rejected_during_ingest(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "incident.md").write_text(
        "Escalate this report to analyst@example.com.\n",
        encoding="utf-8",
    )
    spec = _spec(tmp_path, snapshot)

    with pytest.raises(ValueError, match="sensitive or raw identifier"):
        ingest_snapshot(spec, root=tmp_path)


def test_symlink_in_snapshot_is_rejected(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("outside\n", encoding="utf-8")
    link = snapshot / "linked.md"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")

    provisional = BlockchainSourceSnapshotSpec(
        source_id="fixture.protocol",
        source_class=BlockchainSourceClass.PROTOCOL_SOURCE,
        rights_basis=RightsBasis.APACHE_2_0,
        snapshot_path="snapshot",
        expected_snapshot_digest="0" * 64,
        chain_families=[ChainFamily.EVM],
        threat_domains=[ThreatDomain.SMART_CONTRACT],
    )
    with pytest.raises(ValueError, match="symbolic links"):
        inspect_snapshot(provisional, root=tmp_path)


def test_write_result_is_no_replace_and_digest_bound(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    result = ingest_snapshot(_spec(tmp_path, snapshot), root=tmp_path)
    corpus = tmp_path / "out" / "corpus.jsonl"
    manifest = tmp_path / "out" / "manifest.json"

    write_ingest_result(result, corpus_path=corpus, manifest_path=manifest)

    rows = [
        json.loads(line)
        for line in corpus.read_text(encoding="utf-8").splitlines()
    ]
    stored = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(rows) == result.manifest.documents
    assert stored["corpus_file_digest"] == result.manifest.corpus_file_digest
    with pytest.raises(FileExistsError):
        write_ingest_result(result, corpus_path=corpus, manifest_path=manifest)


def test_snapshot_spec_is_no_replace(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    spec = _spec(tmp_path, snapshot)
    destination = tmp_path / "spec.json"

    write_snapshot_spec(spec, destination)

    with pytest.raises(FileExistsError):
        write_snapshot_spec(spec, destination)


def test_snapshot_path_cannot_escape_repository_root() -> None:
    with pytest.raises(ValueError, match="snapshot_path"):
        BlockchainSourceSnapshotSpec(
            source_id="fixture.protocol",
            source_class=BlockchainSourceClass.PROTOCOL_SOURCE,
            rights_basis=RightsBasis.APACHE_2_0,
            snapshot_path="../outside",
            expected_snapshot_digest="0" * 64,
            chain_families=[ChainFamily.EVM],
            threat_domains=[ThreatDomain.SMART_CONTRACT],
        )
