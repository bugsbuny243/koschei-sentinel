from pathlib import Path

import pytest

from koschei_sentinel.pq_network_intelligence import (
    build_pq_research_snapshot_receipt,
    materialize_pq_network_research_record,
    write_pq_research_snapshot_receipt,
)

_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY = _ROOT / "configs/corpus/pq-network-watch.v1.json"
_SNAPSHOT = _ROOT / "fixtures/pq/research-snapshot/synthetic-source.txt"
_CLAIM = _ROOT / "fixtures/pq/research-snapshot/synthetic-claim.json"


def test_record_and_receipt_outputs_must_be_distinct(tmp_path: Path) -> None:
    snapshot_receipt_path = tmp_path / "snapshot-receipt.json"
    snapshot_receipt = build_pq_research_snapshot_receipt(
        source_id="ethereum-protocol-priorities-2026-02-18",
        snapshot_path=_SNAPSHOT,
        captured_at="2026-09-09T02:05:00+03:00",
        watch_registry_path=_REGISTRY,
    )
    write_pq_research_snapshot_receipt(snapshot_receipt, snapshot_receipt_path)
    shared_output = tmp_path / "shared.json"

    with pytest.raises(ValueError, match="outputs must be different files"):
        materialize_pq_network_research_record(
            claim_path=_CLAIM,
            snapshot_receipt_path=snapshot_receipt_path,
            snapshot_path=_SNAPSHOT,
            watch_registry_path=_REGISTRY,
            output_path=shared_output,
            materialization_receipt_path=shared_output,
        )

    assert not shared_output.exists()
