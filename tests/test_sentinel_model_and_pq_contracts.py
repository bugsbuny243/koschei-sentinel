import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_sentinel_native_model_target_is_distinct_from_external_bootstrap() -> None:
    target = _load("configs/model/sentinel-moe-397b-a35b.target.json")

    assert target["schema_version"] == "sentinel.model-target.v1"
    assert target["product_identity"] == "KOSCHEI_SENTINEL"
    assert target["architecture_family"] == "SPARSE_MOE"
    assert target["parameter_targets"]["total_parameters_billion"] == 397
    assert target["parameter_targets"]["active_parameters_billion"] == 35

    bootstrap = target["external_bootstrap_models"]
    assert len(bootstrap) == 1
    assert bootstrap[0]["model"] == "Qwen/Qwen3.5-397B-A17B"
    assert bootstrap[0]["revision"] == "8472618112abcbd45acbcdc58436aff4233c23f7"
    assert bootstrap[0]["role"] == "EXTERNAL_BOOTSTRAP_AND_TRAINER_PROOF"
    assert bootstrap[0]["is_koschei_sentinel"] is False

    policies = target["policies"]
    assert policies["external_model_must_not_define_product_identity"] is True
    assert policies["paid_large_model_training_requires_explicit_approval"] is True
    assert policies["unverified_checkpoint_use"] == "DENY"
    assert policies["self_generated_training_data_default"] == "DENY"


def test_pq_watch_preserves_priority_discoveries_but_keeps_training_closed() -> None:
    watch = _load("configs/corpus/pq-network-watch.v1.json")

    assert watch["schema_version"] == "sentinel.pq-network-watch.v1"
    assert watch["event_schema"] == "schemas/pq-network-intelligence-v1.schema.json"
    assert watch["source_policy"]["training_authorization_default"] is False
    assert watch["source_policy"]["require_snapshot_sha256_before_verification"] is True
    assert watch["source_policy"]["require_provenance_receipt_before_dataset_admission"] is True

    seeds = {item["title"]: item for item in watch["seed_discoveries"]}
    priority_titles = {
        "EF Protocol: Current and Emerging Priorities",
        "Hegotá EIP Opinion Post and Tier List",
    }
    assert priority_titles.issubset(seeds)

    for seed in watch["seed_discoveries"]:
        assert seed["training_authorization"] is False
        assert seed["review_status"] == "PROPOSED"
        assert seed["snapshot_sha256"] is None

    for title in priority_titles:
        assert seeds[title]["canonical_locator"] is None


def test_pq_event_schema_requires_sentinel_graph_evidence_and_migration_surface() -> None:
    schema = _load("schemas/pq-network-intelligence-v1.schema.json")

    required = set(schema["required"])
    assert {"graph", "migration_surface", "evidence", "confidence"}.issubset(required)

    graph_required = set(schema["properties"]["graph"]["required"])
    assert graph_required == {
        "entity",
        "event",
        "intent",
        "capability",
        "vulnerability",
        "action",
        "consequence",
        "evidence_summary",
    }

    migration_required = set(schema["properties"]["migration_surface"]["required"])
    assert migration_required == {
        "accounts",
        "validators",
        "consensus",
        "wallets_hsm",
        "bridges_cross_chain",
    }

    evidence_item = schema["properties"]["evidence"]["items"]
    evidence_required = set(evidence_item["required"])
    assert {"canonical_locator", "snapshot_sha256", "verified"}.issubset(evidence_required)
