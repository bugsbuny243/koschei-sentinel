from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.research_snapshot import (
    hash_stable_regular_file,
    load_strict_json_object,
    parse_timezone_timestamp,
    read_regular_bytes,
    sha256_canonical_json,
)
from koschei_sentinel.training import atomic_write

_DIGEST = r"^[a-f0-9]{64}$"
_SOURCE_CLASSES = Literal[
    "PRIMARY_PROTOCOL",
    "AUTHORITATIVE",
    "REVIEWED_SECONDARY",
    "CONTEXT_ONLY",
]
_LIFECYCLES = Literal[
    "DISCOVERED",
    "PROPOSED",
    "ACCEPTED",
    "IMPLEMENTING",
    "TESTING",
    "DEPLOYED",
    "REJECTED",
    "SUPERSEDED",
    "UNKNOWN",
]
_EVENT_TYPES = Literal[
    "PROTOCOL_PRIORITY",
    "PROPOSAL",
    "FORK_SCOPE",
    "DEVNET",
    "TESTNET",
    "MAINNET",
    "ACCOUNT_MIGRATION",
    "VALIDATOR_MIGRATION",
    "CONSENSUS_MIGRATION",
    "WALLET_OR_HSM_SUPPORT",
    "BRIDGE_OR_CROSS_CHAIN_COMPATIBILITY",
    "CRYPTOGRAPHY_CHANGE",
]
_MIGRATION_STATES = Literal[
    "NONE",
    "UNKNOWN",
    "RESEARCH",
    "PROPOSED",
    "TESTING",
    "DEPLOYED",
]


class PQResearchSnapshotReceipt(StrictModel):
    schema_version: Literal["sentinel.pq-research-snapshot-receipt.v1"] = (
        "sentinel.pq-research-snapshot-receipt.v1"
    )
    source_id: str = Field(min_length=3, max_length=256)
    title: str = Field(min_length=3, max_length=1024)
    publisher: str = Field(min_length=2, max_length=512)
    source_class: _SOURCE_CLASSES
    canonical_locator: str = Field(min_length=1, max_length=4096)
    review_status: Literal["PROPOSED"] = "PROPOSED"
    watch_registry_sha256: str = Field(pattern=_DIGEST)
    source_row_sha256: str = Field(pattern=_DIGEST)
    captured_at: str = Field(min_length=10, max_length=64)
    snapshot_name: str = Field(min_length=1, max_length=1024)
    snapshot_sha256: str = Field(pattern=_DIGEST)
    snapshot_size_bytes: int = Field(ge=1)
    collection_method: Literal["OPERATOR_SUPPLIED_LOCAL_FILE"] = "OPERATOR_SUPPLIED_LOCAL_FILE"
    source_match_verified: Literal[False] = False
    provenance_review_status: Literal["REVIEW_REQUIRED"] = "REVIEW_REQUIRED"
    network_fetch_performed: Literal[False] = False
    training_authorization: Literal[False] = False
    evaluation_authorization: Literal[False] = False
    gold_eligible: Literal[False] = False
    receipt_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def receipt_contract_verifies(self) -> PQResearchSnapshotReceipt:
        parse_timezone_timestamp(self.captured_at, "PQ research snapshot captured_at")
        unsigned = self.model_dump(mode="json")
        observed = str(unsigned.pop("receipt_sha256"))
        if sha256_canonical_json(unsigned) != observed:
            raise ValueError("PQ research snapshot receipt self-hash does not verify")
        return self


class PQNetworkGraph(StrictModel):
    entity: list[str] = Field(min_length=1)
    event: str = Field(min_length=3)
    intent: str | None = None
    capability: str | None = None
    vulnerability: str | None = None
    action: str | None = None
    consequence: str | None = None
    evidence_summary: str = Field(min_length=3)


class PQCryptography(StrictModel):
    current_signature_or_key_scheme: str | None = None
    proposed_pq_scheme: str | None = None
    hybrid_mode: bool | None = None
    key_exposure_model: str | None = None


class PQMigrationSurface(StrictModel):
    accounts: _MIGRATION_STATES
    validators: _MIGRATION_STATES
    consensus: _MIGRATION_STATES
    wallets_hsm: _MIGRATION_STATES
    bridges_cross_chain: _MIGRATION_STATES


class PQNetworkClaim(StrictModel):
    schema_version: Literal["sentinel.pq-network-claim.v1"] = "sentinel.pq-network-claim.v1"
    source_id: str = Field(min_length=3, max_length=256)
    record_id: str = Field(min_length=8, max_length=256)
    network: str = Field(min_length=2, max_length=256)
    observed_at: str = Field(min_length=10, max_length=64)
    event_type: _EVENT_TYPES | None = None
    lifecycle: _LIFECYCLES
    graph: PQNetworkGraph
    cryptography: PQCryptography | None = None
    migration_surface: PQMigrationSurface
    target_date: date | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: str | None = None

    @model_validator(mode="after")
    def claim_contract_verifies(self) -> PQNetworkClaim:
        parse_timezone_timestamp(self.observed_at, "PQ claim observed_at")
        if len(self.graph.entity) != len(set(self.graph.entity)):
            raise ValueError("PQ claim graph entities must be unique")
        return self


class PQEvidence(StrictModel):
    title: str = Field(min_length=3)
    publisher: str = Field(min_length=2)
    source_class: _SOURCE_CLASSES
    canonical_locator: str
    published_at: None = None
    snapshot_sha256: str = Field(pattern=_DIGEST)
    verified: Literal[False] = False


class PQNetworkIntelligenceRecord(StrictModel):
    schema_version: Literal["sentinel.pq-network-intelligence.v1"] = (
        "sentinel.pq-network-intelligence.v1"
    )
    record_id: str = Field(min_length=8)
    network: str = Field(min_length=2)
    observed_at: str
    event_type: _EVENT_TYPES | None = None
    lifecycle: _LIFECYCLES
    graph: PQNetworkGraph
    cryptography: PQCryptography | None = None
    migration_surface: PQMigrationSurface
    target_date: date | None = None
    evidence: list[PQEvidence] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: str | None = None
    training_authorization: Literal[False] = False

    @model_validator(mode="after")
    def record_contract_verifies(self) -> PQNetworkIntelligenceRecord:
        parse_timezone_timestamp(self.observed_at, "PQ record observed_at")
        if any(row.verified for row in self.evidence):
            raise ValueError("PQ research materialization cannot mark evidence verified")
        return self


class PQNetworkMaterializationReceipt(StrictModel):
    schema_version: Literal["sentinel.pq-network-materialization-receipt.v1"] = (
        "sentinel.pq-network-materialization-receipt.v1"
    )
    source_id: str
    claim_sha256: str = Field(pattern=_DIGEST)
    snapshot_receipt_sha256: str = Field(pattern=_DIGEST)
    snapshot_sha256: str = Field(pattern=_DIGEST)
    record_id: str
    record_sha256: str = Field(pattern=_DIGEST)
    evidence_verified: Literal[False] = False
    provenance_review_status: Literal["REVIEW_REQUIRED"] = "REVIEW_REQUIRED"
    training_authorization: Literal[False] = False
    receipt_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def receipt_contract_verifies(self) -> PQNetworkMaterializationReceipt:
        unsigned = self.model_dump(mode="json")
        observed = str(unsigned.pop("receipt_sha256"))
        if sha256_canonical_json(unsigned) != observed:
            raise ValueError("PQ materialization receipt self-hash does not verify")
        return self


class _PQWatchRegistry:
    def __init__(self, payload: dict[str, object], raw_sha256: str):
        self.payload = payload
        self.raw_sha256 = raw_sha256
        self.sources = self._validate()

    def _validate(self) -> dict[str, dict[str, object]]:
        if self.payload.get("schema_version") != "sentinel.pq-network-watch.v1":
            raise ValueError("PQ watch registry schema mismatch")
        policy = self.payload.get("source_policy")
        if not isinstance(policy, dict):
            raise ValueError("PQ watch registry lacks source_policy")
        required_policy = {
            "require_canonical_locator_before_verification": True,
            "require_snapshot_sha256_before_verification": True,
            "require_provenance_receipt_before_dataset_admission": True,
            "require_human_review_before_gold": True,
            "training_authorization_default": False,
        }
        for key, expected in required_policy.items():
            if policy.get(key) is not expected:
                raise ValueError(f"PQ watch source policy is not fail-closed: {key}")

        networks = self.payload.get("watch_networks")
        if not isinstance(networks, list) or not networks or not all(
            isinstance(row, str) and row.strip() for row in networks
        ):
            raise ValueError("PQ watch registry has invalid watch_networks")
        if len(networks) != len(set(networks)):
            raise ValueError("PQ watch registry has duplicate watch networks")

        discoveries = self.payload.get("seed_discoveries")
        if not isinstance(discoveries, list) or not discoveries:
            raise ValueError("PQ watch registry has no seed discoveries")
        by_id: dict[str, dict[str, object]] = {}
        for row in discoveries:
            if not isinstance(row, dict):
                raise ValueError("PQ watch seed discovery must be an object")
            source_id = row.get("source_id")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError("PQ watch seed discovery lacks source_id")
            if source_id in by_id:
                raise ValueError(f"duplicate PQ source_id: {source_id}")
            if row.get("training_authorization") is not False:
                raise ValueError(f"PQ source training authorization is not closed: {source_id}")
            if row.get("review_status") != "PROPOSED":
                raise ValueError(f"PQ source review status is not PROPOSED: {source_id}")
            if row.get("source_class") not in {
                "PRIMARY_PROTOCOL",
                "AUTHORITATIVE",
                "REVIEWED_SECONDARY",
                "CONTEXT_ONLY",
            }:
                raise ValueError(f"PQ source class is invalid: {source_id}")
            for field in ("title", "publisher"):
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"PQ source lacks {field}: {source_id}")
            by_id[source_id] = row
        return by_id

    @property
    def watch_networks(self) -> set[str]:
        return set(self.payload["watch_networks"])


def _load_watch_registry(path: str | Path) -> _PQWatchRegistry:
    raw = read_regular_bytes(path, "PQ watch registry")
    payload = load_strict_json_object(path, "PQ watch registry")
    return _PQWatchRegistry(payload, hashlib.sha256(raw).hexdigest())


def _source_for_capture(registry: _PQWatchRegistry, source_id: str) -> dict[str, object]:
    source = registry.sources.get(source_id)
    if source is None:
        raise ValueError(f"unknown PQ research snapshot source_id: {source_id}")
    locator = source.get("canonical_locator")
    if not isinstance(locator, str) or not locator.strip():
        raise ValueError(f"PQ source canonical locator is unresolved: {source_id}")
    if source.get("snapshot_sha256") is not None:
        raise ValueError(f"PQ source registry snapshot SHA must remain discovery-only: {source_id}")
    return source


def _model_text(model: StrictModel) -> str:
    return json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def build_pq_research_snapshot_receipt(
    *,
    source_id: str,
    snapshot_path: str | Path,
    captured_at: str,
    watch_registry_path: str | Path,
) -> PQResearchSnapshotReceipt:
    parse_timezone_timestamp(captured_at, "PQ research snapshot captured_at")
    registry = _load_watch_registry(watch_registry_path)
    source = _source_for_capture(registry, source_id)
    snapshot_sha, snapshot_size = hash_stable_regular_file(
        snapshot_path, "PQ research snapshot"
    )
    unsigned: dict[str, object] = {
        "schema_version": "sentinel.pq-research-snapshot-receipt.v1",
        "source_id": source_id,
        "title": str(source["title"]),
        "publisher": str(source["publisher"]),
        "source_class": str(source["source_class"]),
        "canonical_locator": str(source["canonical_locator"]),
        "review_status": "PROPOSED",
        "watch_registry_sha256": registry.raw_sha256,
        "source_row_sha256": sha256_canonical_json(source),
        "captured_at": captured_at,
        "snapshot_name": Path(snapshot_path).name,
        "snapshot_sha256": snapshot_sha,
        "snapshot_size_bytes": snapshot_size,
        "collection_method": "OPERATOR_SUPPLIED_LOCAL_FILE",
        "source_match_verified": False,
        "provenance_review_status": "REVIEW_REQUIRED",
        "network_fetch_performed": False,
        "training_authorization": False,
        "evaluation_authorization": False,
        "gold_eligible": False,
    }
    return PQResearchSnapshotReceipt(
        **unsigned,
        receipt_sha256=sha256_canonical_json(unsigned),
    )


def verify_pq_research_snapshot_receipt(
    *,
    receipt_path: str | Path,
    snapshot_path: str | Path,
    watch_registry_path: str | Path,
) -> PQResearchSnapshotReceipt:
    payload = load_strict_json_object(receipt_path, "PQ research snapshot receipt")
    receipt = PQResearchSnapshotReceipt.model_validate(payload)
    rebuilt = build_pq_research_snapshot_receipt(
        source_id=receipt.source_id,
        snapshot_path=snapshot_path,
        captured_at=receipt.captured_at,
        watch_registry_path=watch_registry_path,
    )
    if rebuilt != receipt:
        raise ValueError("PQ research snapshot receipt differs from current source/snapshot binding")
    return receipt


def write_pq_research_snapshot_receipt(
    receipt: PQResearchSnapshotReceipt, path: str | Path
) -> None:
    atomic_write(Path(path), _model_text(receipt))


def _load_claim(path: str | Path) -> PQNetworkClaim:
    payload = load_strict_json_object(path, "PQ network claim")
    return PQNetworkClaim.model_validate(payload)


def _build_research_record(
    *, claim: PQNetworkClaim, receipt: PQResearchSnapshotReceipt, registry: _PQWatchRegistry
) -> PQNetworkIntelligenceRecord:
    if claim.source_id != receipt.source_id:
        raise ValueError("PQ claim source_id differs from snapshot receipt")
    if claim.network not in registry.watch_networks:
        raise ValueError(f"PQ claim network is not in the watch set: {claim.network}")
    if receipt.watch_registry_sha256 != registry.raw_sha256:
        raise ValueError("PQ snapshot receipt watch registry SHA drifted")
    source = _source_for_capture(registry, receipt.source_id)
    if receipt.source_row_sha256 != sha256_canonical_json(source):
        raise ValueError("PQ snapshot receipt source row SHA drifted")
    if not receipt.canonical_locator or not receipt.snapshot_sha256:
        raise ValueError("PQ research materialization requires locator and snapshot SHA")

    return PQNetworkIntelligenceRecord(
        record_id=claim.record_id,
        network=claim.network,
        observed_at=claim.observed_at,
        event_type=claim.event_type,
        lifecycle=claim.lifecycle,
        graph=claim.graph,
        cryptography=claim.cryptography,
        migration_surface=claim.migration_surface,
        target_date=claim.target_date,
        evidence=[
            PQEvidence(
                title=receipt.title,
                publisher=receipt.publisher,
                source_class=receipt.source_class,
                canonical_locator=receipt.canonical_locator,
                published_at=None,
                snapshot_sha256=receipt.snapshot_sha256,
                verified=False,
            )
        ],
        confidence=claim.confidence,
        uncertainty=claim.uncertainty,
        training_authorization=False,
    )


def materialize_pq_network_research_record(
    *,
    claim_path: str | Path,
    snapshot_receipt_path: str | Path,
    snapshot_path: str | Path,
    watch_registry_path: str | Path,
    output_path: str | Path,
    materialization_receipt_path: str | Path,
) -> tuple[PQNetworkIntelligenceRecord, PQNetworkMaterializationReceipt]:
    output = Path(output_path)
    materialization_output = Path(materialization_receipt_path)
    if output.resolve(strict=False) == materialization_output.resolve(strict=False):
        raise ValueError("PQ record and materialization receipt outputs must be different files")
    if output.exists():
        raise FileExistsError(f"PQ network intelligence output already exists: {output}")
    if materialization_output.exists():
        raise FileExistsError(
            f"PQ network materialization receipt already exists: {materialization_output}"
        )

    registry = _load_watch_registry(watch_registry_path)
    snapshot_receipt = verify_pq_research_snapshot_receipt(
        receipt_path=snapshot_receipt_path,
        snapshot_path=snapshot_path,
        watch_registry_path=watch_registry_path,
    )
    claim = _load_claim(claim_path)
    record = _build_research_record(claim=claim, receipt=snapshot_receipt, registry=registry)
    record_payload = record.model_dump(mode="json")
    record_sha = sha256_canonical_json(record_payload)
    claim_sha = sha256_canonical_json(claim.model_dump(mode="json"))
    unsigned: dict[str, object] = {
        "schema_version": "sentinel.pq-network-materialization-receipt.v1",
        "source_id": snapshot_receipt.source_id,
        "claim_sha256": claim_sha,
        "snapshot_receipt_sha256": snapshot_receipt.receipt_sha256,
        "snapshot_sha256": snapshot_receipt.snapshot_sha256,
        "record_id": record.record_id,
        "record_sha256": record_sha,
        "evidence_verified": False,
        "provenance_review_status": "REVIEW_REQUIRED",
        "training_authorization": False,
    }
    materialization = PQNetworkMaterializationReceipt(
        **unsigned,
        receipt_sha256=sha256_canonical_json(unsigned),
    )
    atomic_write(output, _model_text(record))
    atomic_write(materialization_output, _model_text(materialization))
    return record, materialization


def verify_pq_network_materialization(
    *,
    claim_path: str | Path,
    snapshot_receipt_path: str | Path,
    snapshot_path: str | Path,
    watch_registry_path: str | Path,
    record_path: str | Path,
    materialization_receipt_path: str | Path,
) -> PQNetworkMaterializationReceipt:
    registry = _load_watch_registry(watch_registry_path)
    snapshot_receipt = verify_pq_research_snapshot_receipt(
        receipt_path=snapshot_receipt_path,
        snapshot_path=snapshot_path,
        watch_registry_path=watch_registry_path,
    )
    claim = _load_claim(claim_path)
    expected_record = _build_research_record(
        claim=claim, receipt=snapshot_receipt, registry=registry
    )
    record_payload = load_strict_json_object(record_path, "PQ network intelligence record")
    record = PQNetworkIntelligenceRecord.model_validate(record_payload)
    if record != expected_record:
        raise ValueError("PQ network intelligence record differs from current claim/evidence binding")

    receipt_payload = load_strict_json_object(
        materialization_receipt_path, "PQ network materialization receipt"
    )
    materialization = PQNetworkMaterializationReceipt.model_validate(receipt_payload)
    expected_unsigned: dict[str, object] = {
        "schema_version": "sentinel.pq-network-materialization-receipt.v1",
        "source_id": snapshot_receipt.source_id,
        "claim_sha256": sha256_canonical_json(claim.model_dump(mode="json")),
        "snapshot_receipt_sha256": snapshot_receipt.receipt_sha256,
        "snapshot_sha256": snapshot_receipt.snapshot_sha256,
        "record_id": record.record_id,
        "record_sha256": sha256_canonical_json(record.model_dump(mode="json")),
        "evidence_verified": False,
        "provenance_review_status": "REVIEW_REQUIRED",
        "training_authorization": False,
    }
    expected = PQNetworkMaterializationReceipt(
        **expected_unsigned,
        receipt_sha256=sha256_canonical_json(expected_unsigned),
    )
    if materialization != expected:
        raise ValueError("PQ materialization receipt differs from current claim/snapshot/record binding")
    return materialization
