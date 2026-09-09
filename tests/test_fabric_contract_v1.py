from dataclasses import replace

import pytest

from koschei_sentinel.fabric_contract_v1 import (
    fabric_component_payload_v1,
    fabric_component_v1,
    require_safe_fabric_contract_v1,
)


def test_fabric_contract_is_observe_first_and_non_breaking() -> None:
    component = require_safe_fabric_contract_v1()
    assert component.preserveExisting is True
    assert component.breakingChangesAllowed is False
    assert component.defaultMode == "observe"
    assert component.crossProjectAccess == "contract-only"
    assert component.paidComputeTriggered is False


def test_fabric_contract_exposes_existing_pq_intelligence_without_execution() -> None:
    capabilities = {item.id: item for item in fabric_component_v1().capabilities}
    pq = capabilities["pq-network-intelligence"]
    assert pq.domain == "web6"
    assert pq.backend == "existing"
    assert pq.frontend == "planned"


def test_fabric_payload_is_adapter_friendly() -> None:
    payload = fabric_component_payload_v1()
    assert payload["schemaVersion"] == "1.0"
    assert payload["component"] == "koschei-sentinel"
    assert payload["paidComputeTriggered"] is False


def test_paid_compute_flag_fails_closed() -> None:
    unsafe = replace(fabric_component_v1(), paidComputeTriggered=True)
    with pytest.raises(ValueError, match="paid compute"):
        require_safe_fabric_contract_v1(unsafe)


def test_active_default_mode_fails_closed() -> None:
    unsafe = replace(fabric_component_v1(), defaultMode="active")
    with pytest.raises(ValueError, match="observe/shadow"):
        require_safe_fabric_contract_v1(unsafe)
