from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel


class CloudRuntimeStore(StrictModel):
    kind: Literal["OBJECT_STORAGE", "CLICKHOUSE"]
    required: Literal[True] = True
    uri_env: str = Field(pattern=r"^KOSCHEI_[A-Z0-9_]+$")


class CloudRuntimeStores(StrictModel):
    checkpoint_artifacts: CloudRuntimeStore
    security_analytics: CloudRuntimeStore

    @model_validator(mode="after")
    def stores_have_distinct_bindings(self) -> CloudRuntimeStores:
        if self.checkpoint_artifacts.kind != "OBJECT_STORAGE":
            raise ValueError("checkpoint_artifacts must use OBJECT_STORAGE")
        if self.security_analytics.kind != "CLICKHOUSE":
            raise ValueError("security_analytics must use CLICKHOUSE")
        if self.checkpoint_artifacts.uri_env == self.security_analytics.uri_env:
            raise ValueError("persistent stores must use distinct environment bindings")
        return self


class CloudRuntimeNetwork(StrictModel):
    public_inbound_default: Literal["DENY"] = "DENY"
    cluster_rendezvous: Literal["PRIVATE_OR_PROVIDER_INTERNAL"] = (
        "PRIVATE_OR_PROVIDER_INTERNAL"
    )


class CloudRuntimeBudget(StrictModel):
    paid_compute_default: Literal["DENY"] = "DENY"
    approval_env: Literal["KOSCHEI_CLOUD_RUNTIME_APPROVED"] = (
        "KOSCHEI_CLOUD_RUNTIME_APPROVED"
    )
    session_env: Literal["KOSCHEI_CLOUD_RUNTIME_SESSION"] = (
        "KOSCHEI_CLOUD_RUNTIME_SESSION"
    )


class CloudRuntimeTermination(StrictModel):
    require_checkpoint_sync: Literal[True] = True
    require_manifest_sha256: Literal[True] = True
    require_remote_persistence_before_release: Literal[True] = True


class CloudRuntimeProfile(StrictModel):
    schema_version: Literal["sentinel.cloud-runtime-profile.v1"] = (
        "sentinel.cloud-runtime-profile.v1"
    )
    profile_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    ownership: Literal["EXTERNAL_EPHEMERAL_COMPUTE"] = "EXTERNAL_EPHEMERAL_COMPUTE"
    provider_binding: Literal["UNBOUND"] = "UNBOUND"
    own_server_required: Literal[False] = False
    compute_lifecycle: Literal["EPHEMERAL"] = "EPHEMERAL"
    persistent_local_state: Literal[False] = False
    training_backend: Literal["megatron-swift"] = "megatron-swift"
    stores: CloudRuntimeStores
    model_sources: list[
        Literal["OBJECT_STORAGE", "LOCAL_CACHE", "EXTERNAL_REGISTRY_OPTIONAL"]
    ] = Field(min_length=1, max_length=3)
    network: CloudRuntimeNetwork = Field(default_factory=CloudRuntimeNetwork)
    budget: CloudRuntimeBudget = Field(default_factory=CloudRuntimeBudget)
    termination: CloudRuntimeTermination = Field(default_factory=CloudRuntimeTermination)

    @model_validator(mode="after")
    def runtime_is_serverless_and_fail_closed(self) -> CloudRuntimeProfile:
        if "OBJECT_STORAGE" not in self.model_sources:
            raise ValueError("OBJECT_STORAGE must be an allowed model source")
        if self.budget.paid_compute_default != "DENY":
            raise ValueError("paid compute must fail closed")
        return self


class CloudRuntimePreflight(StrictModel):
    schema_version: Literal["sentinel.cloud-runtime-preflight.v1"] = (
        "sentinel.cloud-runtime-preflight.v1"
    )
    profile_id: str
    own_server_required: Literal[False]
    provider_bound: Literal[False]
    persistent_store_bindings_present: bool
    paid_compute_authorized: bool
    launch_session_present: bool
    launch_ready: bool
    blockers: list[str]


class CloudRuntimeDriveArchive(StrictModel):
    kind: Literal["GOOGLE_DRIVE_ARCHIVE"] = "GOOGLE_DRIVE_ARCHIVE"
    required: Literal[True] = True
    folder_id_env: Literal["KOSCHEI_DRIVE_ARCHIVE_FOLDER_ID"] = (
        "KOSCHEI_DRIVE_ARCHIVE_FOLDER_ID"
    )
    role: Literal["COLD_PERSISTENT_ARCHIVE"] = "COLD_PERSISTENT_ARCHIVE"


class CloudRuntimeHotAnalytics(StrictModel):
    kind: Literal["CLICKHOUSE"] = "CLICKHOUSE"
    required: Literal[False] = False
    uri_env: Literal["KOSCHEI_CLICKHOUSE_DSN"] = "KOSCHEI_CLICKHOUSE_DSN"
    role: Literal["HOT_ANALYTICS_ONLY"] = "HOT_ANALYTICS_ONLY"
    retention_days: int = Field(default=7, ge=1, le=30)


class CloudRuntimeLargeRunStore(StrictModel):
    kind: Literal["OBJECT_STORAGE"] = "OBJECT_STORAGE"
    required: Literal[False] = False
    uri_env: Literal["KOSCHEI_OBJECT_STORE_URI"] = "KOSCHEI_OBJECT_STORE_URI"
    role: Literal["LARGE_RUN_CHECKPOINT_STAGING"] = "LARGE_RUN_CHECKPOINT_STAGING"
    activation: Literal["LARGE_RUN_ONLY"] = "LARGE_RUN_ONLY"


class CloudRuntimeStoresV2(StrictModel):
    persistent_archive: CloudRuntimeDriveArchive
    hot_analytics: CloudRuntimeHotAnalytics
    large_run_checkpoint_staging: CloudRuntimeLargeRunStore


class CloudRuntimeLargeRunBudget(StrictModel):
    default: Literal["DENY"] = "DENY"
    approval_env: Literal["KOSCHEI_LARGE_RUN_APPROVED"] = "KOSCHEI_LARGE_RUN_APPROVED"
    require_object_storage: Literal[True] = True


class CloudRuntimeBudgetV2(StrictModel):
    paid_compute_default: Literal["DENY"] = "DENY"
    approval_env: Literal["KOSCHEI_CLOUD_RUNTIME_APPROVED"] = (
        "KOSCHEI_CLOUD_RUNTIME_APPROVED"
    )
    session_env: Literal["KOSCHEI_CLOUD_RUNTIME_SESSION"] = (
        "KOSCHEI_CLOUD_RUNTIME_SESSION"
    )
    large_run: CloudRuntimeLargeRunBudget = Field(default_factory=CloudRuntimeLargeRunBudget)


class CloudRuntimeTerminationV2(StrictModel):
    require_archive_sync: Literal[True] = True
    require_manifest_sha256: Literal[True] = True
    require_remote_persistence_before_release: Literal[True] = True
    large_run_require_object_storage_sync: Literal[True] = True
    clickhouse_flush_required: Literal[False] = False


class CloudRuntimeProfileV2(StrictModel):
    schema_version: Literal["sentinel.cloud-runtime-profile.v2"] = (
        "sentinel.cloud-runtime-profile.v2"
    )
    profile_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    ownership: Literal["EXTERNAL_EPHEMERAL_COMPUTE"] = "EXTERNAL_EPHEMERAL_COMPUTE"
    provider_binding: Literal["UNBOUND"] = "UNBOUND"
    own_server_required: Literal[False] = False
    compute_lifecycle: Literal["EPHEMERAL"] = "EPHEMERAL"
    persistent_local_state: Literal[False] = False
    training_backend: Literal["megatron-swift"] = "megatron-swift"
    stores: CloudRuntimeStoresV2
    model_sources: list[
        Literal[
            "GOOGLE_DRIVE_ARCHIVE",
            "LOCAL_CACHE",
            "OBJECT_STORAGE_LARGE_RUN_ONLY",
            "EXTERNAL_REGISTRY_OPTIONAL",
        ]
    ] = Field(min_length=2, max_length=4)
    network: CloudRuntimeNetwork = Field(default_factory=CloudRuntimeNetwork)
    budget: CloudRuntimeBudgetV2 = Field(default_factory=CloudRuntimeBudgetV2)
    termination: CloudRuntimeTerminationV2 = Field(default_factory=CloudRuntimeTerminationV2)

    @model_validator(mode="after")
    def runtime_is_cost_aware_and_fail_closed(self) -> CloudRuntimeProfileV2:
        required_sources = {"GOOGLE_DRIVE_ARCHIVE", "LOCAL_CACHE"}
        if not required_sources.issubset(self.model_sources):
            raise ValueError("GOOGLE_DRIVE_ARCHIVE and LOCAL_CACHE must be allowed model sources")
        if self.stores.hot_analytics.required is not False:
            raise ValueError("ClickHouse hot analytics must remain optional")
        if self.stores.large_run_checkpoint_staging.required is not False:
            raise ValueError("object storage must remain large-run-only")
        if self.budget.paid_compute_default != "DENY" or self.budget.large_run.default != "DENY":
            raise ValueError("paid compute and large runs must fail closed")
        return self


class CloudRuntimePreflightV2(StrictModel):
    schema_version: Literal["sentinel.cloud-runtime-preflight.v2"] = (
        "sentinel.cloud-runtime-preflight.v2"
    )
    profile_id: str
    own_server_required: Literal[False]
    provider_bound: Literal[False]
    persistent_archive_binding_present: bool
    clickhouse_binding_present: bool
    object_storage_binding_present: bool
    paid_compute_authorized: bool
    launch_session_present: bool
    large_run_requested: bool
    large_run_authorized: bool
    launch_ready: bool
    blockers: list[str]
    warnings: list[str]


def load_cloud_runtime_profile(path: str | Path) -> CloudRuntimeProfile | CloudRuntimeProfileV2:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version")
    if schema_version == "sentinel.cloud-runtime-profile.v1":
        return CloudRuntimeProfile.model_validate(payload)
    if schema_version == "sentinel.cloud-runtime-profile.v2":
        return CloudRuntimeProfileV2.model_validate(payload)
    raise ValueError(f"unsupported cloud runtime schema_version: {schema_version!r}")


def _preflight_v1(
    profile: CloudRuntimeProfile,
    environ: Mapping[str, str],
) -> CloudRuntimePreflight:
    required_store_envs = (
        profile.stores.checkpoint_artifacts.uri_env,
        profile.stores.security_analytics.uri_env,
    )
    missing_store_envs = [name for name in required_store_envs if not environ.get(name)]
    paid_compute_authorized = environ.get(profile.budget.approval_env) == "YES"
    launch_session_present = bool(environ.get(profile.budget.session_env))

    blockers: list[str] = []
    if missing_store_envs:
        blockers.append("missing persistent store bindings: " + ", ".join(missing_store_envs))
    if not paid_compute_authorized:
        blockers.append(f"paid compute not authorized via {profile.budget.approval_env}=YES")
    if not launch_session_present:
        blockers.append(f"launch session missing in {profile.budget.session_env}")

    return CloudRuntimePreflight(
        profile_id=profile.profile_id,
        own_server_required=False,
        provider_bound=False,
        persistent_store_bindings_present=not missing_store_envs,
        paid_compute_authorized=paid_compute_authorized,
        launch_session_present=launch_session_present,
        launch_ready=not blockers,
        blockers=blockers,
    )


def _preflight_v2(
    profile: CloudRuntimeProfileV2,
    environ: Mapping[str, str],
    *,
    large_run: bool,
) -> CloudRuntimePreflightV2:
    archive_env = profile.stores.persistent_archive.folder_id_env
    clickhouse_env = profile.stores.hot_analytics.uri_env
    object_storage_env = profile.stores.large_run_checkpoint_staging.uri_env

    archive_present = bool(environ.get(archive_env))
    clickhouse_present = bool(environ.get(clickhouse_env))
    object_storage_present = bool(environ.get(object_storage_env))
    paid_compute_authorized = environ.get(profile.budget.approval_env) == "YES"
    launch_session_present = bool(environ.get(profile.budget.session_env))
    large_run_authorized = environ.get(profile.budget.large_run.approval_env) == "YES"

    blockers: list[str] = []
    warnings: list[str] = []

    if not archive_present:
        blockers.append(f"persistent Google Drive archive missing in {archive_env}")
    if not paid_compute_authorized:
        blockers.append(f"paid compute not authorized via {profile.budget.approval_env}=YES")
    if not launch_session_present:
        blockers.append(f"launch session missing in {profile.budget.session_env}")

    if large_run:
        if not object_storage_present:
            blockers.append(f"large run requires object storage binding in {object_storage_env}")
        if not large_run_authorized:
            blockers.append(
                f"large run not authorized via {profile.budget.large_run.approval_env}=YES"
            )
    elif not object_storage_present:
        warnings.append("object storage is unbound; large-run checkpoint staging is unavailable")

    if not clickhouse_present:
        warnings.append(
            "ClickHouse hot analytics is disabled; telemetry must remain in local batch files "
            "until archived or explicitly ingested"
        )

    return CloudRuntimePreflightV2(
        profile_id=profile.profile_id,
        own_server_required=False,
        provider_bound=False,
        persistent_archive_binding_present=archive_present,
        clickhouse_binding_present=clickhouse_present,
        object_storage_binding_present=object_storage_present,
        paid_compute_authorized=paid_compute_authorized,
        launch_session_present=launch_session_present,
        large_run_requested=large_run,
        large_run_authorized=large_run_authorized,
        launch_ready=not blockers,
        blockers=blockers,
        warnings=warnings,
    )


def preflight_cloud_runtime(
    profile: CloudRuntimeProfile | CloudRuntimeProfileV2,
    env: Mapping[str, str] | None = None,
    *,
    large_run: bool = False,
) -> CloudRuntimePreflight | CloudRuntimePreflightV2:
    environ = os.environ if env is None else env
    if isinstance(profile, CloudRuntimeProfileV2):
        return _preflight_v2(profile, environ, large_run=large_run)
    if large_run:
        raise ValueError("--large-run requires a sentinel.cloud-runtime-profile.v2 profile")
    return _preflight_v1(profile, environ)
