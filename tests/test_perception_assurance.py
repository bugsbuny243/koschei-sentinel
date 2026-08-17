import pytest

from koschei_sentinel.cyber_perception import TelemetrySourceType
from koschei_sentinel.perception_adapter_profiles import (
    generic_cloud_iam_adapter,
    generic_endpoint_process_adapter,
)
from koschei_sentinel.perception_adapter_sdk import (
    SanitizedTelemetryEvent,
    run_perception_adapter_batch,
)
from koschei_sentinel.perception_assurance import fuse_admitted_perception_batches
from koschei_sentinel.perception_source_registry import (
    PerceptionSourceEnrollment,
    PerceptionSourceRegistry,
    admit_perception_adapter_result,
)


def _endpoint_result():
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


def _iam_result():
    event = SanitizedTelemetryEvent(
        event_id="event:iam",
        source_type=TelemetrySourceType.CLOUD_IAM,
        source_instance="iam:prod",
        payload_sha256="b" * 64,
        fields={
            "identity_id": "identity:alice",
            "principal": "alice",
            "resource_id": "cloud:prod",
            "resource_type": "cluster",
        },
    )
    return run_perception_adapter_batch(
        generic_cloud_iam_adapter(),
        [event],
        batch_id="batch:iam",
    )


def _registry(*, same_domain: bool = False):
    return PerceptionSourceRegistry(
        registry_id="registry:prod",
        enrollments=[
            PerceptionSourceEnrollment(
                source_type=TelemetrySourceType.ENDPOINT,
                source_instance="edr:prod",
                allowed_adapter_ids=["generic.endpoint-process"],
                independence_domain="shared-domain" if same_domain else "endpoint-control-plane",
            ),
            PerceptionSourceEnrollment(
                source_type=TelemetrySourceType.CLOUD_IAM,
                source_instance="iam:prod",
                allowed_adapter_ids=["generic.cloud-iam"],
                independence_domain="shared-domain" if same_domain else "cloud-control-plane",
            ),
        ],
    )


def test_assured_fusion_counts_independent_control_planes() -> None:
    registry = _registry()
    result = fuse_admitted_perception_batches(
        [
            admit_perception_adapter_result(_endpoint_result(), registry),
            admit_perception_adapter_result(_iam_result(), registry),
        ],
        registry=registry,
        fused_batch_id="fusion:assured",
    )

    assert result.assurance.independent_domain_count == 2
    assert result.assurance.independence_domains == [
        "cloud-control-plane",
        "endpoint-control-plane",
    ]
    assert result.assurance.largest_domain_fraction == 0.5
    assert result.assurance.production_admitted is True


def test_two_sources_in_same_domain_do_not_count_as_independent_corroboration() -> None:
    registry = _registry(same_domain=True)
    result = fuse_admitted_perception_batches(
        [
            admit_perception_adapter_result(_endpoint_result(), registry),
            admit_perception_adapter_result(_iam_result(), registry),
        ],
        registry=registry,
        fused_batch_id="fusion:same-domain",
    )

    assert result.assurance.independent_domain_count == 1
    assert result.assurance.largest_domain_fraction == 1.0


def test_assured_fusion_rejects_replayed_admission() -> None:
    registry = _registry()
    admitted = admit_perception_adapter_result(_endpoint_result(), registry)
    with pytest.raises(ValueError, match="admission_id values must be unique"):
        fuse_admitted_perception_batches(
            [admitted, admitted.model_copy(deep=True)],
            registry=registry,
            fused_batch_id="fusion:replay",
        )


def test_assured_fusion_reverifies_forged_admission_receipt() -> None:
    registry = _registry()
    admitted = admit_perception_adapter_result(_endpoint_result(), registry)
    forged = admitted.model_copy(
        deep=True,
        update={
            "admission": admitted.admission.model_copy(
                update={"registry_sha256": "f" * 64}
            )
        },
    )
    with pytest.raises(ValueError, match="registry re-verification"):
        fuse_admitted_perception_batches(
            [forged],
            registry=registry,
            fused_batch_id="fusion:forged",
        )
