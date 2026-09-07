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


def load_cloud_runtime_profile(path: str | Path) -> CloudRuntimeProfile:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CloudRuntimeProfile.model_validate(payload)


def preflight_cloud_runtime(
    profile: CloudRuntimeProfile,
    env: Mapping[str, str] | None = None,
) -> CloudRuntimePreflight:
    environ = os.environ if env is None else env
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
