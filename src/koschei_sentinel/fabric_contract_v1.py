"""Koschei Fabric v1 contract provider for Sentinel.

This module is metadata-only: importing or reading the contract must never start
training, HOLDOUT inference, active defense execution, network collection, or
paid compute. Runtime adapters may consume this contract in observe mode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

FABRIC_SCHEMA_VERSION: Final = "1.0"
FABRIC_COMPONENT: Final = "koschei-sentinel"


@dataclass(frozen=True, slots=True)
class FabricCapabilityV1:
    id: str
    domain: str
    status: str
    backend: str
    frontend: str
    telemetry: str


@dataclass(frozen=True, slots=True)
class FabricComponentV1:
    schemaVersion: str
    component: str
    repository: str
    role: str
    preserveExisting: bool
    defaultMode: str
    breakingChangesAllowed: bool
    crossProjectAccess: str
    paidComputeTriggered: bool
    capabilities: tuple[FabricCapabilityV1, ...]


def fabric_component_v1() -> FabricComponentV1:
    """Return Sentinel's immutable, observe-first Fabric capability contract."""
    return FabricComponentV1(
        schemaVersion=FABRIC_SCHEMA_VERSION,
        component=FABRIC_COMPONENT,
        repository="bugsbuny243/koschei-sentinel",
        role="independent-cybersecurity-model",
        preserveExisting=True,
        defaultMode="observe",
        breakingChangesAllowed=False,
        crossProjectAccess="contract-only",
        paidComputeTriggered=False,
        capabilities=(
            FabricCapabilityV1(
                id="sentinel-evaluation-and-provenance",
                domain="core",
                status="stable",
                backend="existing",
                frontend="planned",
                telemetry="existing",
            ),
            FabricCapabilityV1(
                id="fabric-observation-plane",
                domain="web4",
                status="experimental",
                backend="adapter",
                frontend="planned",
                telemetry="planned",
            ),
            FabricCapabilityV1(
                id="pq-network-intelligence",
                domain="web6",
                status="experimental",
                backend="existing",
                frontend="planned",
                telemetry="existing",
            ),
        ),
    )


def fabric_component_payload_v1() -> dict[str, object]:
    """Return a JSON-serializable payload for a Fabric adapter."""
    return asdict(fabric_component_v1())


def require_safe_fabric_contract_v1(component: FabricComponentV1 | None = None) -> FabricComponentV1:
    """Fail closed if integration metadata weakens Sentinel safety boundaries."""
    value = component or fabric_component_v1()
    if value.schemaVersion != FABRIC_SCHEMA_VERSION:
        raise ValueError("unsupported Fabric schema version")
    if not value.preserveExisting:
        raise ValueError("Fabric contract must preserve existing Sentinel behavior")
    if value.breakingChangesAllowed:
        raise ValueError("Fabric contract must not permit breaking changes")
    if value.crossProjectAccess != "contract-only":
        raise ValueError("cross-project access must remain contract-only")
    if value.defaultMode not in {"observe", "shadow"}:
        raise ValueError("Sentinel Fabric integration must start in observe/shadow mode")
    if value.paidComputeTriggered:
        raise ValueError("Fabric metadata must never trigger paid compute")
    return value
