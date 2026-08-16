from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.blockchain_security_eval import (
    BlockchainSecurityEvalAudit,
    BlockchainSecurityEvalPolicy,
    audit_blockchain_security_eval,
    load_eval_policy,
    load_eval_receipt,
)
from koschei_sentinel.blockchain_training import BlockchainAdapterManifest
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import atomic_write, model_digest, resolve_under_root

_DIGEST = r"^[a-f0-9]{64}$"
_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"


class BlockchainTournamentRuntimeProfile(StrictModel):
    schema_version: Literal["sentinel.blockchain-runtime-profile.v1"] = (
        "sentinel.blockchain-runtime-profile.v1"
    )
    candidate_id: str = Field(pattern=_ID)
    adapter_digest: str = Field(pattern=_DIGEST)
    runtime_suite_digest: str = Field(pattern=_DIGEST)
    environment_digest: str = Field(pattern=_DIGEST)
    measured_cases: int = Field(ge=1, le=1_000_000)
    prompt_tokens: int = Field(ge=1)
    generated_tokens: int = Field(ge=1)
    wall_time_ms: int = Field(ge=1)
    p95_case_latency_ms: int = Field(ge=1)
    peak_gpu_memory_mb: int = Field(ge=1)
    raw_model_outputs_stored: Literal[False] = False
    profile_digest: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def digest_is_valid(self) -> BlockchainTournamentRuntimeProfile:
        payload = self.model_dump(mode="json")
        expected = payload.pop("profile_digest")
        if _digest(payload) != expected:
            raise ValueError("blockchain runtime profile digest mismatch")
        return self


class BlockchainTournamentCandidateSpec(StrictModel):
    candidate_id: str = Field(pattern=_ID)
    adapter_manifest_path: str = Field(min_length=1, max_length=1024)
    eval_receipt_path: str = Field(min_length=1, max_length=1024)
    eval_audit_path: str = Field(min_length=1, max_length=1024)
    runtime_profile_path: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def paths_are_local(self) -> BlockchainTournamentCandidateSpec:
        for field in (
            "adapter_manifest_path",
            "eval_receipt_path",
            "eval_audit_path",
            "runtime_profile_path",
        ):
            _validate_relative_path(getattr(self, field), field)
        return self


class BlockchainModelTournamentSpec(StrictModel):
    schema_version: Literal["sentinel.blockchain-model-tournament.v1"] = (
        "sentinel.blockchain-model-tournament.v1"
    )
    tournament_id: str = Field(pattern=_ID)
    candidates: list[BlockchainTournamentCandidateSpec] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def candidates_are_unique(self) -> BlockchainModelTournamentSpec:
        identifiers = [item.candidate_id for item in self.candidates]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("tournament candidate_id values must be unique")
        for field in (
            "adapter_manifest_path",
            "eval_receipt_path",
            "eval_audit_path",
            "runtime_profile_path",
        ):
            paths = [getattr(item, field) for item in self.candidates]
            if len(paths) != len(set(paths)):
                raise ValueError(f"tournament {field} values must be unique")
        return self


class BlockchainModelTournamentPolicy(StrictModel):
    schema_version: Literal["sentinel.blockchain-model-tournament-policy.v1"] = (
        "sentinel.blockchain-model-tournament-policy.v1"
    )
    policy_id: str = Field(pattern=_ID)
    min_candidates: int = Field(default=3, ge=2, le=128)
    min_eligible_candidates: int = Field(default=3, ge=2, le=128)
    require_same_source_corpus_digest: Literal[True] = True
    require_same_held_out_test_split_digest: Literal[True] = True
    require_same_benchmark_suite_digest: Literal[True] = True
    require_same_evaluator_digest: Literal[True] = True
    require_same_runtime_suite_digest: Literal[True] = True
    require_same_runtime_environment_digest: Literal[True] = True
    require_unique_adapter_digests: Literal[True] = True
    require_eval_gate_pass: Literal[True] = True
    max_peak_gpu_memory_mb: int = Field(default=131_072, ge=1)

    @model_validator(mode="after")
    def eligible_count_is_possible(self) -> BlockchainModelTournamentPolicy:
        if self.min_eligible_candidates > self.min_candidates:
            raise ValueError("min_eligible_candidates may not exceed min_candidates")
        return self


class BlockchainTournamentCandidateResult(StrictModel):
    candidate_id: str
    base_model: str
    base_revision: str
    adapter_digest: str = Field(pattern=_DIGEST)
    eligible: bool
    rank: int | None = Field(default=None, ge=1)
    exclusion_reasons: list[str]
    overall_pass_bps: int = Field(ge=0, le=10_000)
    security_floor_bps: int = Field(ge=0, le=10_000)
    security_sum_bps: int = Field(ge=0)
    worst_chain_pass_bps: int = Field(ge=0, le=10_000)
    worst_threat_pass_bps: int = Field(ge=0, le=10_000)
    worst_task_pass_bps: int = Field(ge=0, le=10_000)
    grounding_bps: int = Field(ge=0, le=10_000)
    task_correct_bps: int = Field(ge=0, le=10_000)
    patch_safe_bps: int = Field(ge=0, le=10_000)
    abstention_correct_bps: int = Field(ge=0, le=10_000)
    throughput_milli_tokens_per_second: int = Field(ge=1)
    p95_case_latency_ms: int = Field(ge=1)
    peak_gpu_memory_mb: int = Field(ge=1)


class BlockchainModelTournamentResult(StrictModel):
    schema_version: Literal["sentinel.blockchain-model-tournament-result.v1"] = (
        "sentinel.blockchain-model-tournament-result.v1"
    )
    ready: bool
    tournament_id: str
    policy_id: str
    tournament_digest: str = Field(pattern=_DIGEST)
    policy_digest: str = Field(pattern=_DIGEST)
    eval_policy_digest: str = Field(pattern=_DIGEST)
    source_corpus_digest: str | None = Field(default=None, pattern=_DIGEST)
    held_out_test_split_digest: str | None = Field(default=None, pattern=_DIGEST)
    benchmark_suite_digest: str | None = Field(default=None, pattern=_DIGEST)
    evaluator_digest: str | None = Field(default=None, pattern=_DIGEST)
    runtime_suite_digest: str | None = Field(default=None, pattern=_DIGEST)
    runtime_environment_digest: str | None = Field(default=None, pattern=_DIGEST)
    total_candidates: int = Field(ge=1)
    eligible_candidates: int = Field(ge=0)
    winner_candidate_id: str | None = None
    ranking: list[str]
    candidates: list[BlockchainTournamentCandidateResult]
    violations: list[str]
    automatic_production_promotion_allowed: Literal[False] = False


class _LoadedCandidate:
    def __init__(
        self,
        *,
        spec: BlockchainTournamentCandidateSpec,
        adapter: BlockchainAdapterManifest,
        audit: BlockchainSecurityEvalAudit,
        evaluator_digest: str,
        runtime: BlockchainTournamentRuntimeProfile,
    ) -> None:
        self.spec = spec
        self.adapter = adapter
        self.audit = audit
        self.evaluator_digest = evaluator_digest
        self.runtime = runtime


def build_runtime_profile(
    *,
    candidate_id: str,
    adapter_digest: str,
    runtime_suite_digest: str,
    environment_digest: str,
    measured_cases: int,
    prompt_tokens: int,
    generated_tokens: int,
    wall_time_ms: int,
    p95_case_latency_ms: int,
    peak_gpu_memory_mb: int,
) -> BlockchainTournamentRuntimeProfile:
    payload = {
        "schema_version": "sentinel.blockchain-runtime-profile.v1",
        "candidate_id": candidate_id,
        "adapter_digest": adapter_digest,
        "runtime_suite_digest": runtime_suite_digest,
        "environment_digest": environment_digest,
        "measured_cases": measured_cases,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "wall_time_ms": wall_time_ms,
        "p95_case_latency_ms": p95_case_latency_ms,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "raw_model_outputs_stored": False,
    }
    return BlockchainTournamentRuntimeProfile.model_validate(
        {**payload, "profile_digest": _digest(payload)}
    )


def load_runtime_profile(path: str | Path) -> BlockchainTournamentRuntimeProfile:
    try:
        return BlockchainTournamentRuntimeProfile.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid blockchain tournament runtime profile") from exc


def load_tournament_spec(path: str | Path) -> BlockchainModelTournamentSpec:
    try:
        return BlockchainModelTournamentSpec.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid blockchain model tournament spec") from exc


def load_tournament_policy(path: str | Path) -> BlockchainModelTournamentPolicy:
    try:
        return BlockchainModelTournamentPolicy.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid blockchain model tournament policy") from exc


def run_blockchain_model_tournament(
    tournament: BlockchainModelTournamentSpec,
    policy: BlockchainModelTournamentPolicy,
    eval_policy: BlockchainSecurityEvalPolicy,
    *,
    root: str | Path = ".",
) -> BlockchainModelTournamentResult:
    root_path = Path(root).resolve()
    loaded = [
        _load_candidate(item, eval_policy, root=root_path)
        for item in sorted(tournament.candidates, key=lambda value: value.candidate_id)
    ]
    violations: list[str] = []
    if len(loaded) < policy.min_candidates:
        violations.append(
            f"candidates {len(loaded)} below minimum {policy.min_candidates}"
        )

    adapter_digests = [item.adapter.adapter_digest for item in loaded]
    if len(adapter_digests) != len(set(adapter_digests)):
        violations.append("tournament contains duplicate adapter digests")

    common = {
        "source_corpus_digest": _common_value(
            [item.adapter.source_corpus_digest for item in loaded],
            "source corpus digest",
            violations,
        ),
        "held_out_test_split_digest": _common_value(
            [item.adapter.held_out_test_split_digest for item in loaded],
            "held-out test split digest",
            violations,
        ),
        "benchmark_suite_digest": _common_value(
            [item.adapter.benchmark_suite_digest for item in loaded],
            "benchmark suite digest",
            violations,
        ),
        "evaluator_digest": _common_value(
            [item.evaluator_digest for item in loaded],
            "evaluator digest",
            violations,
        ),
        "runtime_suite_digest": _common_value(
            [item.runtime.runtime_suite_digest for item in loaded],
            "runtime suite digest",
            violations,
        ),
        "runtime_environment_digest": _common_value(
            [item.runtime.environment_digest for item in loaded],
            "runtime environment digest",
            violations,
        ),
    }

    candidate_results: list[BlockchainTournamentCandidateResult] = []
    for item in loaded:
        exclusion_reasons: list[str] = []
        if not item.audit.ready:
            exclusion_reasons.append("blockchain security evaluation gate did not pass")
        if item.runtime.peak_gpu_memory_mb > policy.max_peak_gpu_memory_mb:
            exclusion_reasons.append(
                "runtime peak GPU memory exceeds tournament policy"
            )
        candidate_results.append(
            _candidate_result(item, exclusion_reasons=exclusion_reasons)
        )

    eligible = [item for item in candidate_results if item.eligible]
    if len(eligible) < policy.min_eligible_candidates:
        violations.append(
            f"eligible candidates {len(eligible)} below minimum "
            f"{policy.min_eligible_candidates}"
        )

    ranked = sorted(eligible, key=_ranking_key)
    rank_by_id = {item.candidate_id: index for index, item in enumerate(ranked, 1)}
    final_candidates = [
        item.model_copy(update={"rank": rank_by_id.get(item.candidate_id)})
        for item in candidate_results
    ]
    ranking = [item.candidate_id for item in ranked]
    ready = not violations
    return BlockchainModelTournamentResult(
        ready=ready,
        tournament_id=tournament.tournament_id,
        policy_id=policy.policy_id,
        tournament_digest=model_digest(tournament),
        policy_digest=model_digest(policy),
        eval_policy_digest=model_digest(eval_policy),
        source_corpus_digest=common["source_corpus_digest"],
        held_out_test_split_digest=common["held_out_test_split_digest"],
        benchmark_suite_digest=common["benchmark_suite_digest"],
        evaluator_digest=common["evaluator_digest"],
        runtime_suite_digest=common["runtime_suite_digest"],
        runtime_environment_digest=common["runtime_environment_digest"],
        total_candidates=len(final_candidates),
        eligible_candidates=len(eligible),
        winner_candidate_id=ranking[0] if ready and ranking else None,
        ranking=ranking,
        candidates=final_candidates,
        violations=violations,
    )


def write_tournament_result(result: BlockchainModelTournamentResult, path: str | Path) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"blockchain model tournament result already exists: {destination}")
    atomic_write(
        destination,
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )


def _load_candidate(
    spec: BlockchainTournamentCandidateSpec,
    eval_policy: BlockchainSecurityEvalPolicy,
    *,
    root: Path,
) -> _LoadedCandidate:
    adapter_path = resolve_under_root(root, spec.adapter_manifest_path)
    receipt_path = resolve_under_root(root, spec.eval_receipt_path)
    audit_path = resolve_under_root(root, spec.eval_audit_path)
    runtime_path = resolve_under_root(root, spec.runtime_profile_path)
    for path, label in (
        (adapter_path, "adapter manifest"),
        (receipt_path, "evaluation receipt"),
        (audit_path, "evaluation audit"),
        (runtime_path, "runtime profile"),
    ):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"candidate {label} is missing or unsafe: {spec.candidate_id}")

    try:
        adapter = BlockchainAdapterManifest.model_validate_json(
            adapter_path.read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError(f"invalid adapter manifest: {spec.candidate_id}") from exc
    if adapter.run_id != spec.candidate_id:
        raise ValueError("tournament candidate_id does not match adapter run_id")
    if adapter.held_out_test_consumed is not False:
        raise ValueError("tournament candidate consumed held-out test data")

    receipt = load_eval_receipt(receipt_path)
    if receipt.candidate_id != spec.candidate_id:
        raise ValueError("tournament candidate_id does not match evaluation receipt")
    recomputed = audit_blockchain_security_eval(receipt, adapter, eval_policy)
    try:
        stored_audit = BlockchainSecurityEvalAudit.model_validate_json(
            audit_path.read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError(f"invalid evaluation audit: {spec.candidate_id}") from exc
    if stored_audit != recomputed:
        raise ValueError("stored blockchain evaluation audit is stale or inconsistent")

    runtime = load_runtime_profile(runtime_path)
    if runtime.candidate_id != spec.candidate_id:
        raise ValueError("tournament candidate_id does not match runtime profile")
    if runtime.adapter_digest != adapter.adapter_digest:
        raise ValueError("runtime profile adapter digest does not match candidate")
    return _LoadedCandidate(
        spec=spec,
        adapter=adapter,
        audit=recomputed,
        evaluator_digest=receipt.evaluator_digest,
        runtime=runtime,
    )


def _candidate_result(
    candidate: _LoadedCandidate,
    *,
    exclusion_reasons: list[str],
) -> BlockchainTournamentCandidateResult:
    audit = candidate.audit
    worst_chain = min(audit.chain_pass_bps.values(), default=0)
    worst_threat = min(audit.threat_pass_bps.values(), default=0)
    worst_task = min(audit.task_pass_bps.values(), default=0)
    security_metrics = (
        audit.overall_pass_bps,
        worst_chain,
        worst_threat,
        worst_task,
        audit.grounding_bps,
        audit.task_correct_bps,
        audit.patch_safe_bps,
        audit.abstention_correct_bps,
    )
    throughput = max(
        1,
        candidate.runtime.generated_tokens * 1_000_000 // candidate.runtime.wall_time_ms,
    )
    return BlockchainTournamentCandidateResult(
        candidate_id=candidate.spec.candidate_id,
        base_model=candidate.adapter.base_model,
        base_revision=candidate.adapter.base_revision,
        adapter_digest=candidate.adapter.adapter_digest,
        eligible=not exclusion_reasons,
        exclusion_reasons=exclusion_reasons,
        overall_pass_bps=audit.overall_pass_bps,
        security_floor_bps=min(security_metrics),
        security_sum_bps=sum(security_metrics),
        worst_chain_pass_bps=worst_chain,
        worst_threat_pass_bps=worst_threat,
        worst_task_pass_bps=worst_task,
        grounding_bps=audit.grounding_bps,
        task_correct_bps=audit.task_correct_bps,
        patch_safe_bps=audit.patch_safe_bps,
        abstention_correct_bps=audit.abstention_correct_bps,
        throughput_milli_tokens_per_second=throughput,
        p95_case_latency_ms=candidate.runtime.p95_case_latency_ms,
        peak_gpu_memory_mb=candidate.runtime.peak_gpu_memory_mb,
    )


def _ranking_key(item: BlockchainTournamentCandidateResult) -> tuple[int | str, ...]:
    return (
        -item.security_floor_bps,
        -item.security_sum_bps,
        -item.overall_pass_bps,
        -item.throughput_milli_tokens_per_second,
        item.p95_case_latency_ms,
        item.peak_gpu_memory_mb,
        item.candidate_id,
    )


def _common_value(values: list[str], label: str, violations: list[str]) -> str | None:
    unique = sorted(set(values))
    if len(unique) != 1:
        violations.append(f"candidates do not share one {label}")
        return None
    return unique[0]


def _validate_relative_path(value: str, field: str) -> None:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or value.startswith("~")
        or "\\" in value
    ):
        raise ValueError(f"{field} must stay within the repository root")


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
