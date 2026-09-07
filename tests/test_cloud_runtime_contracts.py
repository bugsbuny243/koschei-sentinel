from __future__ import annotations

from pathlib import Path

from koschei_sentinel.cloud_runtime import (
    CloudRuntimeProfile,
    load_cloud_runtime_profile,
    preflight_cloud_runtime,
)


PROFILE = Path("configs/runtime/ephemeral-cloud-gpu.v1.json")


def test_default_profile_requires_no_owned_server() -> None:
    profile = load_cloud_runtime_profile(PROFILE)

    assert profile.own_server_required is False
    assert profile.provider_binding == "UNBOUND"
    assert profile.compute_lifecycle == "EPHEMERAL"
    assert profile.persistent_local_state is False
    assert profile.stores.checkpoint_artifacts.kind == "OBJECT_STORAGE"
    assert profile.stores.security_analytics.kind == "CLICKHOUSE"
    assert "EXTERNAL_REGISTRY_OPTIONAL" in profile.model_sources


def test_preflight_fails_closed_without_persistent_bindings_and_paid_approval() -> None:
    profile = load_cloud_runtime_profile(PROFILE)

    result = preflight_cloud_runtime(profile, env={})

    assert result.launch_ready is False
    assert result.persistent_store_bindings_present is False
    assert result.paid_compute_authorized is False
    assert result.launch_session_present is False
    assert any("KOSCHEI_OBJECT_STORE_URI" in blocker for blocker in result.blockers)
    assert any("KOSCHEI_CLICKHOUSE_DSN" in blocker for blocker in result.blockers)
    assert any("KOSCHEI_CLOUD_RUNTIME_APPROVED=YES" in blocker for blocker in result.blockers)


def test_preflight_becomes_ready_without_provider_or_hugging_face_dependency() -> None:
    profile = load_cloud_runtime_profile(PROFILE)
    env = {
        "KOSCHEI_OBJECT_STORE_URI": "s3://sentinel-private/checkpoints",
        "KOSCHEI_CLICKHOUSE_DSN": "clickhouse://sentinel.invalid/default",
        "KOSCHEI_CLOUD_RUNTIME_APPROVED": "YES",
        "KOSCHEI_CLOUD_RUNTIME_SESSION": "fixture-session",
    }

    result = preflight_cloud_runtime(profile, env=env)

    assert result.launch_ready is True
    assert result.provider_bound is False
    assert result.own_server_required is False
    assert result.blockers == []


def test_profile_rejects_non_object_checkpoint_store() -> None:
    payload = load_cloud_runtime_profile(PROFILE).model_dump(mode="json")
    payload["stores"]["checkpoint_artifacts"]["kind"] = "CLICKHOUSE"

    try:
        CloudRuntimeProfile.model_validate(payload)
    except ValueError as exc:
        assert "checkpoint_artifacts must use OBJECT_STORAGE" in str(exc)
    else:
        raise AssertionError("profile accepted ClickHouse as checkpoint artifact storage")


def test_profile_rejects_shared_store_secret_binding() -> None:
    payload = load_cloud_runtime_profile(PROFILE).model_dump(mode="json")
    payload["stores"]["security_analytics"]["uri_env"] = "KOSCHEI_OBJECT_STORE_URI"

    try:
        CloudRuntimeProfile.model_validate(payload)
    except ValueError as exc:
        assert "distinct environment bindings" in str(exc)
    else:
        raise AssertionError("profile accepted one environment binding for both stores")
