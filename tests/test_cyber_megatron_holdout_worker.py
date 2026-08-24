import json
from types import SimpleNamespace

import pytest

import koschei_sentinel.cyber_megatron_holdout_worker as worker_module
from koschei_sentinel.cyber_megatron_holdout import build_cyber_megatron_holdout_plan
from koschei_sentinel.cyber_megatron_holdout_worker import (
    EXECUTION_STATE_FILENAME,
    FAILURES_FILENAME,
    HOLDOUT_LAUNCH_APPROVAL_ENV,
    HOLDOUT_LAUNCH_APPROVAL_VALUE,
    HOLDOUT_LAUNCH_SESSION_ENV,
    PREDICTIONS_FILENAME,
    RAW_RESULTS_FILENAME,
    RECEIPT_FILENAME,
    REQUEST_BINDINGS_FILENAME,
    REQUESTS_FILENAME,
    WORKER_PLAN_FILENAME,
    CyberMegatronHoldoutWorkerProfile,
    execute_cyber_megatron_holdout_worker,
    finalize_cyber_megatron_holdout_worker,
    prepare_cyber_megatron_holdout_worker,
    verify_cyber_megatron_holdout_output,
)
from tests.test_cyber_megatron_candidate import _relative
from tests.test_cyber_megatron_holdout import _plan_kwargs


def _worker_fixture(tmp_path):
    candidate, source_kwargs = _plan_kwargs(tmp_path)
    holdout_plan_path = tmp_path / "build" / "holdout" / "397b-plan.json"
    holdout_plan = build_cyber_megatron_holdout_plan(
        **source_kwargs,
        output_path=_relative(tmp_path, holdout_plan_path),
    )
    worker_dir = tmp_path / "build" / "holdout" / "397b-worker"
    profile = CyberMegatronHoldoutWorkerProfile(
        nodes=2,
        gpus_per_node=2,
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        vllm_version="0.17.0",
        max_num_seqs=2,
    )
    worker_plan = prepare_cyber_megatron_holdout_worker(
        **source_kwargs,
        holdout_plan_path=_relative(tmp_path, holdout_plan_path),
        output_dir=_relative(tmp_path, worker_dir),
        profile=profile,
    )
    return candidate, source_kwargs, holdout_plan, holdout_plan_path, worker_dir, worker_plan


def _valid_response() -> str:
    return json.dumps(
        {
            "interpretation": "Visible evidence remains below the containment threshold.",
            "defense_sequence": [
                {
                    "sequence": 1,
                    "expected_mode": "GUARD",
                    "action": "OBSERVE",
                    "target_entity_id": "asset-001",
                    "rationale": "Continue observation while preserving evidence.",
                    "supporting_evidence_ids": ["fixture-evidence-001"],
                    "outcome_verification_required": True,
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _fake_swift_run(tmp_path):
    def fake_run(command, *, cwd, check, env):
        assert check is False
        assert cwd == tmp_path
        assert env[HOLDOUT_LAUNCH_APPROVAL_ENV] == HOLDOUT_LAUNCH_APPROVAL_VALUE
        requests_path = cwd / command[command.index("--val_dataset") + 1]
        result_path = cwd / command[command.index("--result_path") + 1]
        rows = [
            json.loads(line)
            for line in requests_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        response = _valid_response()
        results = []
        for row in rows:
            messages = list(row["messages"])
            messages.append({"role": "assistant", "content": response})
            results.append(
                {
                    "response": response,
                    "labels": None,
                    "logprobs": None,
                    "messages": messages,
                }
            )
        result_path.write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in results
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    return fake_run


def _approve_mocked_execution(monkeypatch, tmp_path, worker_plan) -> None:
    monkeypatch.setenv(HOLDOUT_LAUNCH_APPROVAL_ENV, HOLDOUT_LAUNCH_APPROVAL_VALUE)
    monkeypatch.setenv(HOLDOUT_LAUNCH_SESSION_ENV, worker_plan.plan_sha256)
    monkeypatch.setenv("RAY_ADDRESS", "auto")
    monkeypatch.setattr(worker_module.shutil, "which", lambda name: f"/mock/{name}")
    versions = {"ms-swift": "4.5.2", "vllm": "0.17.0", "ray": "2.54.0"}
    monkeypatch.setattr(worker_module, "_package_version", versions.__getitem__)
    monkeypatch.setattr(worker_module, "_ray_cluster_capacity", lambda address: (2, 4))
    monkeypatch.setattr(worker_module.subprocess, "run", _fake_swift_run(tmp_path))


def test_prepare_worker_binds_merged_checkpoint_prompts_and_ray_command(tmp_path) -> None:
    candidate, _source, holdout, _holdout_path, worker_dir, worker_plan = _worker_fixture(
        tmp_path
    )

    assert worker_plan.candidate_sha256 == candidate.candidate_sha256
    assert worker_plan.checkpoint_tree_sha256 == candidate.checkpoint_tree_sha256
    assert worker_plan.holdout_plan_sha256 == holdout.plan_sha256
    assert worker_plan.checkpoint_format == "swift-merged-hf"
    assert worker_plan.profile.distributed_executor_backend == "ray"
    assert worker_plan.profile.tensor_parallel_size == 2
    assert worker_plan.profile.pipeline_parallel_size == 2
    assert worker_plan.profile.total_gpus == 4
    assert worker_plan.command[:2] == ["swift", "infer"]
    backend_index = worker_plan.command.index("--infer_backend")
    assert worker_plan.command[backend_index + 1] == "vllm"
    engine_kwargs = worker_plan.command[worker_plan.command.index("--vllm_engine_kwargs") + 1]
    assert json.loads(engine_kwargs) == {"distributed_executor_backend": "ray"}
    assert sorted(path.name for path in worker_dir.iterdir()) == sorted(
        [WORKER_PLAN_FILENAME, REQUESTS_FILENAME, REQUEST_BINDINGS_FILENAME]
    )
    requests = (worker_dir / REQUESTS_FILENAME).read_text(encoding="utf-8")
    assert "expected_sequence" not in requests
    assert "expected_interpretation" not in requests


def test_execute_requires_exact_approved_worker_plan_sha(tmp_path, monkeypatch) -> None:
    _candidate, _source, _holdout, _holdout_path, worker_dir, worker_plan = _worker_fixture(
        tmp_path
    )
    monkeypatch.setenv(HOLDOUT_LAUNCH_APPROVAL_ENV, HOLDOUT_LAUNCH_APPROVAL_VALUE)
    monkeypatch.setenv(HOLDOUT_LAUNCH_SESSION_ENV, "0" * 64)

    with pytest.raises(PermissionError, match="exact worker plan SHA256"):
        execute_cyber_megatron_holdout_worker(
            worker_dir=_relative(tmp_path, worker_dir),
            root=tmp_path,
        )

    assert worker_plan.plan_sha256 != "0" * 64
    assert not (worker_dir / RAW_RESULTS_FILENAME).exists()


def test_execute_rehashes_checkpoint_before_any_paid_subprocess(tmp_path, monkeypatch) -> None:
    _candidate, _source, _holdout, _holdout_path, worker_dir, worker_plan = _worker_fixture(
        tmp_path
    )
    checkpoint = tmp_path / worker_plan.checkpoint_path
    (checkpoint / "model.safetensors").write_bytes(b"changed-after-prepare")
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("subprocess must not start")

    monkeypatch.setattr(worker_module.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="tree SHA differs"):
        execute_cyber_megatron_holdout_worker(
            worker_dir=_relative(tmp_path, worker_dir),
            root=tmp_path,
        )
    assert called is False


def test_execute_rejects_ray_cluster_shape_before_swift(tmp_path, monkeypatch) -> None:
    _candidate, _source, _holdout, _holdout_path, worker_dir, worker_plan = _worker_fixture(
        tmp_path
    )
    monkeypatch.setenv(HOLDOUT_LAUNCH_APPROVAL_ENV, HOLDOUT_LAUNCH_APPROVAL_VALUE)
    monkeypatch.setenv(HOLDOUT_LAUNCH_SESSION_ENV, worker_plan.plan_sha256)
    monkeypatch.setenv("RAY_ADDRESS", "auto")
    monkeypatch.setattr(worker_module.shutil, "which", lambda name: f"/mock/{name}")
    versions = {"ms-swift": "4.5.2", "vllm": "0.17.0", "ray": "2.54.0"}
    monkeypatch.setattr(worker_module, "_package_version", versions.__getitem__)
    monkeypatch.setattr(worker_module, "_ray_cluster_capacity", lambda address: (1, 2))
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("subprocess must not start")

    monkeypatch.setattr(worker_module.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="Ray cluster capacity differs"):
        execute_cyber_megatron_holdout_worker(
            worker_dir=_relative(tmp_path, worker_dir),
            root=tmp_path,
        )
    assert called is False


def test_mocked_execute_finalize_and_offline_verify_replay_raw_swift_output(
    tmp_path,
    monkeypatch,
) -> None:
    _candidate, source, _holdout, holdout_path, worker_dir, worker_plan = _worker_fixture(
        tmp_path
    )
    _approve_mocked_execution(monkeypatch, tmp_path, worker_plan)

    return_code = execute_cyber_megatron_holdout_worker(
        worker_dir=_relative(tmp_path, worker_dir),
        root=tmp_path,
    )
    assert return_code == 0
    assert (worker_dir / RAW_RESULTS_FILENAME).is_file()
    assert (worker_dir / EXECUTION_STATE_FILENAME).is_file()

    receipt = finalize_cyber_megatron_holdout_worker(
        worker_dir=_relative(tmp_path, worker_dir),
        inference_pack=source["inference_pack"],
        root=tmp_path,
    )
    assert receipt.prediction_count == 1
    assert receipt.failure_count == 0
    assert receipt.failed_case_ids == []
    assert receipt.runtime_versions == {
        "ms-swift": "4.5.2",
        "vllm": "0.17.0",
        "ray": "2.54.0",
    }
    assert (worker_dir / PREDICTIONS_FILENAME).is_file()
    assert (worker_dir / FAILURES_FILENAME).is_file()
    assert (worker_dir / RECEIPT_FILENAME).is_file()

    verification = verify_cyber_megatron_holdout_output(
        **source,
        worker_dir=_relative(tmp_path, worker_dir),
        holdout_plan_path=_relative(tmp_path, holdout_path),
    )
    assert verification.valid is True
    assert verification.source_plan_verified is True
    assert verification.worker_plan_verified is True
    assert verification.request_binding_verified is True
    assert verification.execution_verified is True
    assert verification.raw_results_replayed is True
    assert verification.receipt_verified is True
    assert verification.complete_case_accounting is True
    assert verification.violations == []


def test_offline_verifier_rejects_posthoc_normalized_prediction_edit(
    tmp_path,
    monkeypatch,
) -> None:
    _candidate, source, _holdout, holdout_path, worker_dir, worker_plan = _worker_fixture(
        tmp_path
    )
    _approve_mocked_execution(monkeypatch, tmp_path, worker_plan)
    assert execute_cyber_megatron_holdout_worker(
        worker_dir=_relative(tmp_path, worker_dir),
        root=tmp_path,
    ) == 0
    finalize_cyber_megatron_holdout_worker(
        worker_dir=_relative(tmp_path, worker_dir),
        inference_pack=source["inference_pack"],
        root=tmp_path,
    )

    prediction_path = worker_dir / PREDICTIONS_FILENAME
    prediction_path.write_text(
        prediction_path.read_text(encoding="utf-8").replace(
            "Visible evidence remains below the containment threshold.",
            "posthoc edit",
        ),
        encoding="utf-8",
    )
    verification = verify_cyber_megatron_holdout_output(
        **source,
        worker_dir=_relative(tmp_path, worker_dir),
        holdout_plan_path=_relative(tmp_path, holdout_path),
    )
    assert verification.valid is False
    assert any("raw SWIFT replay" in row for row in verification.violations)


def test_finalize_rejects_raw_result_not_bound_to_planned_messages(
    tmp_path,
    monkeypatch,
) -> None:
    _candidate, source, _holdout, _holdout_path, worker_dir, worker_plan = _worker_fixture(
        tmp_path
    )
    _approve_mocked_execution(monkeypatch, tmp_path, worker_plan)
    assert execute_cyber_megatron_holdout_worker(
        worker_dir=_relative(tmp_path, worker_dir),
        root=tmp_path,
    ) == 0

    raw_path = worker_dir / RAW_RESULTS_FILENAME
    raw = json.loads(raw_path.read_text(encoding="utf-8").splitlines()[0])
    raw["messages"][0]["content"] = "attacker changed the planned prompt"
    rewritten = json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n"
    raw_path.write_text(rewritten, encoding="utf-8")

    with pytest.raises(ValueError, match="raw results differ from execution state"):
        finalize_cyber_megatron_holdout_worker(
            worker_dir=_relative(tmp_path, worker_dir),
            inference_pack=source["inference_pack"],
            root=tmp_path,
        )
