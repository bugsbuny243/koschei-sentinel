from __future__ import annotations

from koschei_sentinel.cyber_perception import TelemetrySourceType
from koschei_sentinel.cyber_state_graph import CyberEntityType
from koschei_sentinel.perception_adapter_sdk import (
    DeclarativeAdapterSpec,
    DeclarativePerceptionAdapter,
    EntityMappingSpec,
    PerceptionAdapterDescriptor,
)


def generic_endpoint_process_adapter() -> DeclarativePerceptionAdapter:
    return DeclarativePerceptionAdapter(
        DeclarativeAdapterSpec(
            descriptor=PerceptionAdapterDescriptor(
                adapter_id="generic.endpoint-process",
                vendor="generic",
                product="endpoint-process",
                adapter_version="1",
                supported_source_types=[TelemetrySourceType.ENDPOINT],
            ),
            source=EntityMappingSpec(
                id_field="device_id",
                entity_type=CyberEntityType.DEVICE,
                label_fields={"hostname": "hostname", "platform": "platform"},
            ),
            target=EntityMappingSpec(
                id_field="process_id",
                entity_type=CyberEntityType.PROCESS,
                label_fields={"process_name": "process_name", "image_sha256": "image_sha256"},
            ),
            relation_type="executes",
            default_confidence=1.0,
        )
    )


def generic_cloud_iam_adapter() -> DeclarativePerceptionAdapter:
    return DeclarativePerceptionAdapter(
        DeclarativeAdapterSpec(
            descriptor=PerceptionAdapterDescriptor(
                adapter_id="generic.cloud-iam",
                vendor="generic",
                product="cloud-iam",
                adapter_version="1",
                supported_source_types=[TelemetrySourceType.CLOUD_IAM],
            ),
            source=EntityMappingSpec(
                id_field="identity_id",
                entity_type=CyberEntityType.IDENTITY,
                label_fields={"principal": "principal", "tenant": "tenant"},
            ),
            target=EntityMappingSpec(
                id_field="resource_id",
                entity_type=CyberEntityType.CLOUD_RESOURCE,
                label_fields={"resource_type": "resource_type", "region": "region"},
            ),
            relation_type="authenticates_to",
            default_confidence=1.0,
        )
    )


def generic_cicd_adapter() -> DeclarativePerceptionAdapter:
    return DeclarativePerceptionAdapter(
        DeclarativeAdapterSpec(
            descriptor=PerceptionAdapterDescriptor(
                adapter_id="generic.cicd",
                vendor="generic",
                product="cicd",
                adapter_version="1",
                supported_source_types=[TelemetrySourceType.CICD],
            ),
            source=EntityMappingSpec(
                id_field="repository_id",
                entity_type=CyberEntityType.REPOSITORY,
                label_fields={"repository": "repository", "revision": "revision"},
            ),
            target=EntityMappingSpec(
                id_field="pipeline_id",
                entity_type=CyberEntityType.PIPELINE,
                label_fields={"pipeline": "pipeline", "run_id": "run_id"},
            ),
            relation_type="modifies_pipeline",
            default_confidence=1.0,
        )
    )


def generic_signer_wallet_adapter() -> DeclarativePerceptionAdapter:
    return DeclarativePerceptionAdapter(
        DeclarativeAdapterSpec(
            descriptor=PerceptionAdapterDescriptor(
                adapter_id="generic.signer-wallet",
                vendor="generic",
                product="signer-wallet",
                adapter_version="1",
                supported_source_types=[TelemetrySourceType.SIGNER_WALLET],
            ),
            source=EntityMappingSpec(
                id_field="wallet_id",
                entity_type=CyberEntityType.WALLET,
                label_fields={"network": "network", "wallet_class": "wallet_class"},
            ),
            target=EntityMappingSpec(
                id_field="transaction_id",
                entity_type=CyberEntityType.TRANSACTION,
                label_fields={"network": "network", "tx_kind": "tx_kind"},
            ),
            relation_type="prepares_transaction",
            default_confidence=1.0,
        )
    )
