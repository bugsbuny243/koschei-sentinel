from __future__ import annotations

from pathlib import Path

from koschei_sentinel import cloud_runtime

PROFILE = Path("configs/runtime/ephemeral-cloud-gpu-cost-aware.v2.json")


def test_v2_profile_uses_drive_as_required_archive_and_clickhouse_as_optional_hot_store() -> None:
    profile = cloud_runtime.load_cloud_runtime_profile(PROFILE)

    assert isinstance(profile, cloud_runtime.CloudRuntimeProfileV2)
    assert profile.stores.persistent_archive.kind == "GOOGLE_DRIVE_ARCHIVE"
    assert profile.stores.persistent_archive.required is True
    assert profile.stores.hot_analytics.kind == "CLICKHOUSE"
    assert profile.stores.hot_analytics.required is False
    assert profile.stores.hot_analytics.retention_days == 7
    assert profile.stores.large_run_checkpoint_staging.required is False
    assert "GOOGLE_DRIVE_ARCHIVE" in profile.model_sources
    assert "LOCAL_CACHE" in profile.model_sources


def test_normal_v2_preflight_does_not_require_clickhouse_or_object_storage() -> None:
    profile = cloud_runtime.load_cloud_runtime_profile(PROFILE)
    env = {
        "KOSCHEI_DRIVE_ARCHIVE_FOLDER_ID": "fixture-drive-folder",
        "KOSCHEI_CLOUD_RUNTIME_APPROVED": "YES",
        "KOSCHEI_CLOUD_RUNTIME_SESSION": "fixture-session",
    }

    result = cloud_runtime.preflight_cloud_runtime(profile, env=env)

    assert isinstance(result, cloud_runtime.CloudRuntimePreflightV2)
    assert result.launch_ready is True
    assert result.persistent_archive_binding_present is True
    assert result.clickhouse_binding_present is False
    assert result.object_storage_binding_present is False
    assert result.large_run_requested is False
    assert result.blockers == []
    assert any("ClickHouse hot analytics is disabled" in item for item in result.warnings)
    assert any("large-run checkpoint staging" in item for item in result.warnings)


def test_v2_preflight_still_fails_closed_without_drive_archive() -> None:
    profile = cloud_runtime.load_cloud_runtime_profile(PROFILE)
    env = {
        "KOSCHEI_CLOUD_RUNTIME_APPROVED": "YES",
        "KOSCHEI_CLOUD_RUNTIME_SESSION": "fixture-session",
    }

    result = cloud_runtime.preflight_cloud_runtime(profile, env=env)

    assert result.launch_ready is False
    assert any("KOSCHEI_DRIVE_ARCHIVE_FOLDER_ID" in item for item in result.blockers)


def test_large_run_requires_separate_approval_and_object_storage() -> None:
    profile = cloud_runtime.load_cloud_runtime_profile(PROFILE)
    env = {
        "KOSCHEI_DRIVE_ARCHIVE_FOLDER_ID": "fixture-drive-folder",
        "KOSCHEI_CLOUD_RUNTIME_APPROVED": "YES",
        "KOSCHEI_CLOUD_RUNTIME_SESSION": "fixture-session",
    }

    result = cloud_runtime.preflight_cloud_runtime(profile, env=env, large_run=True)

    assert result.launch_ready is False
    assert result.large_run_requested is True
    assert result.large_run_authorized is False
    assert any("KOSCHEI_OBJECT_STORE_URI" in item for item in result.blockers)
    assert any("KOSCHEI_LARGE_RUN_APPROVED=YES" in item for item in result.blockers)


def test_large_run_can_be_ready_without_clickhouse() -> None:
    profile = cloud_runtime.load_cloud_runtime_profile(PROFILE)
    env = {
        "KOSCHEI_DRIVE_ARCHIVE_FOLDER_ID": "fixture-drive-folder",
        "KOSCHEI_OBJECT_STORE_URI": "s3://fixture-large-run",
        "KOSCHEI_CLOUD_RUNTIME_APPROVED": "YES",
        "KOSCHEI_CLOUD_RUNTIME_SESSION": "fixture-session",
        "KOSCHEI_LARGE_RUN_APPROVED": "YES",
    }

    result = cloud_runtime.preflight_cloud_runtime(profile, env=env, large_run=True)

    assert result.launch_ready is True
    assert result.object_storage_binding_present is True
    assert result.clickhouse_binding_present is False
    assert result.large_run_authorized is True
    assert result.blockers == []


def test_large_run_flag_is_rejected_by_legacy_v1_profile() -> None:
    legacy = cloud_runtime.load_cloud_runtime_profile(
        "configs/runtime/ephemeral-cloud-gpu.v1.json"
    )

    try:
        cloud_runtime.preflight_cloud_runtime(legacy, env={}, large_run=True)
    except ValueError as exc:
        assert "requires a sentinel.cloud-runtime-profile.v2" in str(exc)
    else:
        raise AssertionError("legacy v1 profile accepted the v2 large-run gate")
