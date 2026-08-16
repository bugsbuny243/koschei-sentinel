from __future__ import annotations

import json
from pathlib import Path

import pytest

from koschei_sentinel.blockchain_model_tournament import (
    BlockchainModelTournamentPolicy,
    BlockchainModelTournamentSpec,
    BlockchainTournamentCandidateSpec,
    build_runtime_profile,
    run_blockchain_model_tournament,
)
from koschei_sentinel.blockchain_security_corpus import ChainFamily, ThreatDomain
from koschei_sentinel.blockchain_security_eval import (
    BlockchainEvalCaseResult,
    BlockchainEvalTask,
    BlockchainSecurityEvalPolicy,
    audit_blockchain_security_eval,
    build_eval_receipt,
)
from koschei_sentinel.blockchain_training import BlockchainAdapterManifest


def _digest(index: int, *, width: int = 64) -> str:
    return format(index, f"0{width}x")[-width:]


def _adapter(candidate_id: str, index: int) -> BlockchainAdapterManifest:
    return BlockchainAdapterManifest(
        run_id=candidate_id,
        base_model=f"fixture/model-{index}",
        base_revision=_digest(index, width=40),
        release_manifest_file_digest=_digest(100),
        release_manifest_digest=_digest(101),
        source_corpus_digest=_digest(102),
        benchmark_suite_digest=_digest(103),
        train_split_digest=_digest(104),
        validation_split_digest=_digest(105),
        held_out_test_split_digest=_digest(106),
        training_config_digest=_digest(200 + index),
        train_documents=100,
        validation_documents=20,
        held_out_test_documents=20,
        held_out_test_sources=5,
        train_chunks=100,
        validation_chunks=20,
        adapter_digest=_digest(300 + index),
        adapter_files=["adapter/adapter_model.safetensors"],
        output_dir=f"build/{candidate_id}",
        train_loss=0.2,
        eval_loss=0.1,
        held_out_test_consumed=False,
    )


def _eval_policy() -> BlockchainSecurityEvalPolicy:
    return BlockchainSecurityEvalPolicy(
        policy_id="tournament-test",
        min_cases=4,
        required_chain_families=[ChainFamily.EVM],
        required_threat_domains=[ThreatDomain.SMART_CONTRACT],
        required_tasks=[
            BlockchainEvalTask.VULNERABILITY_DETECTION,
            BlockchainEvalTask.DEFENSIVE_PATCH_REVIEW,
            BlockchainEvalTask.ABSTENTION_CALIBRATION,
        ],
        min_cases_per_required_chain=1,
        min_cases_per_required_threat_domain=1,
        min_cases_per_required_task=1,
        min_overall_pass_bps=5000,
        min_chain_pass_bps=0,
        min_threat_pass_bps=0,
        min_task_pass_bps=0,
        min_grounding_bps=0,
        min_task_correct_bps=0,
        min_patch_safe_bps=0,
        min_abstention_correct_bps=0,
    )


def _cases(
    *, weak: bool = False, authority_failure: bool = False
) -> list[BlockchainEvalCaseResult]:
    tasks = [
        BlockchainEvalTask.VULNERABILITY_DETECTION,
        BlockchainEvalTask.VULNERABILITY_DETECTION,
        BlockchainEvalTask.DEFENSIVE_PATCH_REVIEW,
        BlockchainEvalTask.ABSTENTION_CALIBRATION,
    ]
    output: list[BlockchainEvalCaseResult] = []
    for index, task in enumerate(tasks, 1):
        output.append(
            BlockchainEvalCaseResult(
                case_ref=f"eval_{index:024x}",
                case_digest=_digest(400 + index),
                oracle_digest=_digest(500 + index),
                chain_families=[ChainFamily.EVM],
                threat_domains=[ThreatDomain.SMART_CONTRACT],
                task=task,
                authority_valid=not (authority_failure and index == 1),
                privacy_clean=True,
                grounding_valid=True,
                verdict_identity_valid=True,
                confidence_valid=True,
                task_correct=not (weak and index == 1),
                family_isolated=True,
                patch_safe=True if task is BlockchainEvalTask.DEFENSIVE_PATCH_REVIEW else None,
                abstention_correct=(
                    True if task is BlockchainEvalTask.ABSTENTION_CALIBRATION else None
                ),
                raw_model_output_stored=False,
            )
        )
    return output


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _materialize_candidate(
    root: Path,
    *,
    candidate_id: str,
    index: int,
    wall_time_ms: int,
    weak: bool = False,
    authority_failure: bool = False,
    environment_digest: str | None = None,
) -> BlockchainTournamentCandidateSpec:
    directory = root / "candidates" / candidate_id
    directory.mkdir(parents=True)
    adapter = _adapter(candidate_id, index)
    receipt = build_eval_receipt(
        candidate_id=candidate_id,
        adapter=adapter,
        evaluator_digest=_digest(600),
        cases=_cases(weak=weak, authority_failure=authority_failure),
    )
    audit = audit_blockchain_security_eval(receipt, adapter, _eval_policy())
    runtime = build_runtime_profile(
        candidate_id=candidate_id,
        adapter_digest=adapter.adapter_digest,
        runtime_suite_digest=_digest(700),
        environment_digest=environment_digest or _digest(701),
        measured_cases=20,
        prompt_tokens=20_000,
        generated_tokens=10_000,
        wall_time_ms=wall_time_ms,
        p95_case_latency_ms=max(1, wall_time_ms // 10),
        peak_gpu_memory_mb=16_000 + index,
    )
    adapter_path = directory / "adapter.json"
    receipt_path = directory / "receipt.json"
    audit_path = directory / "audit.json"
    runtime_path = directory / "runtime.json"
    _write_json(adapter_path, adapter)
    _write_json(receipt_path, receipt)
    _write_json(audit_path, audit)
    _write_json(runtime_path, runtime)
    return BlockchainTournamentCandidateSpec(
        candidate_id=candidate_id,
        adapter_manifest_path=adapter_path.relative_to(root).as_posix(),
        eval_receipt_path=receipt_path.relative_to(root).as_posix(),
        eval_audit_path=audit_path.relative_to(root).as_posix(),
        runtime_profile_path=runtime_path.relative_to(root).as_posix(),
    )


def _tournament_policy() -> BlockchainModelTournamentPolicy:
    return BlockchainModelTournamentPolicy(
        policy_id="tournament-test",
        min_candidates=3,
        min_eligible_candidates=3,
        max_peak_gpu_memory_mb=64_000,
    )


def test_security_quality_beats_raw_runtime_speed(tmp_path: Path) -> None:
    candidates = [
        _materialize_candidate(
            tmp_path,
            candidate_id="candidate-a",
            index=1,
            wall_time_ms=10_000,
        ),
        _materialize_candidate(
            tmp_path,
            candidate_id="candidate-b",
            index=2,
            wall_time_ms=5_000,
        ),
        _materialize_candidate(
            tmp_path,
            candidate_id="candidate-c",
            index=3,
            wall_time_ms=1_000,
            weak=True,
        ),
    ]
    tournament = BlockchainModelTournamentSpec(
        tournament_id="security-first",
        candidates=candidates,
    )

    result = run_blockchain_model_tournament(
        tournament,
        _tournament_policy(),
        _eval_policy(),
        root=tmp_path,
    )

    assert result.ready is True
    assert result.winner_candidate_id == "candidate-b"
    assert result.ranking == ["candidate-b", "candidate-a", "candidate-c"]
    weak = next(item for item in result.candidates if item.candidate_id == "candidate-c")
    assert weak.throughput_milli_tokens_per_second > next(
        item.throughput_milli_tokens_per_second
        for item in result.candidates
        if item.candidate_id == "candidate-b"
    )
    assert weak.security_floor_bps < 10_000


def test_mismatched_runtime_environment_blocks_winner(tmp_path: Path) -> None:
    candidates = [
        _materialize_candidate(
            tmp_path,
            candidate_id="candidate-a",
            index=1,
            wall_time_ms=10_000,
        ),
        _materialize_candidate(
            tmp_path,
            candidate_id="candidate-b",
            index=2,
            wall_time_ms=9_000,
        ),
        _materialize_candidate(
            tmp_path,
            candidate_id="candidate-c",
            index=3,
            wall_time_ms=8_000,
            environment_digest=_digest(999),
        ),
    ]
    result = run_blockchain_model_tournament(
        BlockchainModelTournamentSpec(tournament_id="mismatch", candidates=candidates),
        _tournament_policy(),
        _eval_policy(),
        root=tmp_path,
    )

    assert result.ready is False
    assert result.winner_candidate_id is None
    assert "runtime environment digest" in " ".join(result.violations)


def test_candidate_that_fails_eval_gate_is_not_eligible(tmp_path: Path) -> None:
    candidates = [
        _materialize_candidate(
            tmp_path,
            candidate_id="candidate-a",
            index=1,
            wall_time_ms=10_000,
        ),
        _materialize_candidate(
            tmp_path,
            candidate_id="candidate-b",
            index=2,
            wall_time_ms=9_000,
        ),
        _materialize_candidate(
            tmp_path,
            candidate_id="candidate-c",
            index=3,
            wall_time_ms=8_000,
            authority_failure=True,
        ),
    ]
    result = run_blockchain_model_tournament(
        BlockchainModelTournamentSpec(tournament_id="hard-gate", candidates=candidates),
        _tournament_policy(),
        _eval_policy(),
        root=tmp_path,
    )

    failed = next(item for item in result.candidates if item.candidate_id == "candidate-c")
    assert failed.eligible is False
    assert result.ready is False
    assert result.eligible_candidates == 2
    assert result.winner_candidate_id is None


def test_runtime_profile_tampering_is_rejected(tmp_path: Path) -> None:
    candidate = _materialize_candidate(
        tmp_path,
        candidate_id="candidate-a",
        index=1,
        wall_time_ms=10_000,
    )
    runtime_path = tmp_path / candidate.runtime_profile_path
    payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    payload["p95_case_latency_ms"] += 1
    runtime_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="runtime profile"):
        run_blockchain_model_tournament(
            BlockchainModelTournamentSpec(tournament_id="tamper", candidates=[candidate]),
            BlockchainModelTournamentPolicy(
                policy_id="tamper",
                min_candidates=2,
                min_eligible_candidates=2,
            ),
            _eval_policy(),
            root=tmp_path,
        )


def test_repository_tournament_policy_is_fail_closed() -> None:
    root = Path(__file__).parents[1]
    payload = json.loads(
        (root / "configs/eval/blockchain-model-tournament.v1.json").read_text(
            encoding="utf-8"
        )
    )
    policy = BlockchainModelTournamentPolicy.model_validate(payload)
    assert policy.min_candidates >= 3
    assert policy.min_eligible_candidates >= 3
    assert policy.require_same_source_corpus_digest is True
    assert policy.require_same_held_out_test_split_digest is True
    assert policy.require_same_benchmark_suite_digest is True
    assert policy.require_same_evaluator_digest is True
    assert policy.require_same_runtime_suite_digest is True
    assert policy.require_same_runtime_environment_digest is True
    assert policy.require_unique_adapter_digests is True
    assert policy.require_eval_gate_pass is True
