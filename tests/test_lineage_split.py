from koschei_sentinel.dataset import export_record
from koschei_sentinel.split import split_dataset

SALT = "unit-test-salt-with-minimum-length"


def source(case_id: str, target: str, lineages: list[str]) -> dict:
    return {
        "schema_version": "arvis.export.v1",
        "case_id": case_id,
        "network": "solana-mainnet",
        "target": target,
        "signed_verdict": {
            "grade": "D",
            "signature": f"fixture-signature-{case_id}",
            "triggered_rules": ["KS-LINEAGE-001"],
            "summary": "A deterministic lineage relation was observed.",
        },
        "evidence": [
            {
                "evidence_id": f"evidence-{case_id}",
                "kind": "intelligence_graph",
                "statement": "A bounded historical relation was observed.",
                "confidence": "VERIFIED",
                "rule_ids": ["KS-LINEAGE-001"],
            }
        ],
        "lineage_ids": lineages,
    }


def test_export_preserves_only_pseudonymized_lineage_refs() -> None:
    item = export_record(
        source("case-1", "target-1", ["actor-family-a", "funding-cluster-b"]),
        salt=SALT,
    )
    assert len(item.lineage_refs) == 2
    assert all(value.startswith("lineage_") for value in item.lineage_refs)
    payload = item.model_dump_json()
    assert "actor-family-a" not in payload
    assert "funding-cluster-b" not in payload


def test_transitive_lineage_overlap_is_one_split_component() -> None:
    records = [
        source("case-1", "target-1", ["actor-family-a"]),
        source("case-2", "target-2", ["actor-family-a", "funding-cluster-b"]),
        source("case-3", "target-3", ["funding-cluster-b"]),
    ]
    examples = [export_record(item, salt=SALT) for item in records]

    assert len({item.group_ref for item in examples}) == 3

    manifest = split_dataset(examples, output_dir=None, dry_run=True)
    assert manifest.total_examples == 3
    assert manifest.total_groups == 1
    assert sorted(report.examples for report in manifest.splits.values()) == [0, 0, 3]
    assert sorted(report.groups for report in manifest.splits.values()) == [0, 0, 1]
