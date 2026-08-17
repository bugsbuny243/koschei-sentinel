import pytest

from koschei_sentinel.cyber_perception import TelemetrySourceType
from koschei_sentinel.perception_adapter_profiles import generic_endpoint_process_adapter
from koschei_sentinel.perception_adapter_sdk import (
    SanitizedTelemetryEvent,
    run_perception_adapter_batch,
)
from koschei_sentinel.perception_source_registry import (
    PerceptionSourceEnrollment,
    PerceptionSourceRegistry,
    admit_perception_adapter_result,
)


def _result():
    event = SanitizedTelemetryEvent(
        event_id="event:endpoint",
        source_type=TelemetrySourceType.ENDPOINT,
        source_instance="edr:prod",
        payload_sha256="a" * 64,
        fields={
            "device_id": "device:prod",
            "hostname": "prod",
            "process_id": "process:worker",
            "process_name": "worker",
        },
    )
    return run_perception_adapter_batch(
        generic_endpoint_process_adapter(),
        [event],
        batch_id="batch:endpoint",
    )


def _registry(*, enabled: bool = True, adapter_id: str = "generic.endpoint-process"):
    return PerceptionSourceRegistry(
        registry_id="registry:prod",
        enrollments=[
            PerceptionSourceEnrollment(
                source_type=TelemetrySourceType.ENDPOINT,
                source_instance="edr:prod",
                allowed_adapter_ids=[adapter_id],
                independence_domain="endpoint-control-plane",
                enabled=enabled,
            )
        ],
    )


def test_registered_adapter_result_is_admitted_with_bound_digests() -> None:
    result = _result()
    admitted = admit_perception_adapter_result(result, _registry())

    assert admitted.admission.admitted is True
    assert admitted.admission.adapter_id == "generic.endpoint-process"
    assert admitted.admission.source_principals == ["ENDPOINT:edr:prod"]
    assert admitted.admission.independence_domains == ["endpoint-control-plane"]
    assert admitted.admission.adapter_batch_sha256 == result.batch_sha256


def test_unregistered_sensor_is_rejected() -> None:
    registry = PerceptionSourceRegistry(
        registry_id="registry:other",
        enrollments=[
            PerceptionSourceEnrollment(
                source_type=TelemetrySourceType.ENDPOINT,
                source_instance="edr:other",
                allowed_adapter_ids=["generic.endpoint-process"],
                independence_domain="endpoint-control-plane",
            )
        ],
    )
    with pytest.raises(ValueError, match="not enrolled"):
        admit_perception_adapter_result(_result(), registry)


def test_adapter_not_allowed_for_sensor_is_rejected() -> None:
    with pytest.raises(ValueError, match="not permitted"):
        admit_perception_adapter_result(_result(), _registry(adapter_id="vendor.other"))


def test_disabled_sensor_enrollment_is_rejected() -> None:
    with pytest.raises(ValueError, match="disabled"):
        admit_perception_adapter_result(_result(), _registry(enabled=False))


def test_tampered_adapter_batch_digest_is_rejected() -> None:
    result = _result().model_copy(update={"batch_sha256": "f" * 64})
    with pytest.raises(ValueError, match="digest does not match"):
        admit_perception_adapter_result(result, _registry())


def test_tampered_receipt_output_digest_is_rejected() -> None:
    result = _result()
    bad_receipt = result.receipts[0].model_copy(update={"output_sha256": "f" * 64})
    tampered = result.model_copy(update={"receipts": [bad_receipt]})
    # Rebind the outer batch digest so the inner receipt integrity check is what fails.
    import hashlib
    import json

    payload = json.dumps(
        {
            "perception_batch": tampered.perception_batch.model_dump(mode="json"),
            "receipts": [row.model_dump(mode="json") for row in tampered.receipts],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    tampered = tampered.model_copy(
        update={"batch_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}
    )
    with pytest.raises(ValueError, match="output digest"):
        admit_perception_adapter_result(tampered, _registry())
