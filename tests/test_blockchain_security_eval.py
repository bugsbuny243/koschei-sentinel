from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from koschei_sentinel.blockchain_security_corpus import ChainFamily, ThreatDomain
from koschei_sentinel.blockchain_security_eval import (
    BlockchainEvalCaseResult,
    BlockchainEvalTask,
    BlockchainSecurityEvalPolicy,
    BlockchainSecurityEvalReceipt,
    audit_blockchain_security_eval,
    build_eval_receipt,
    load_eval_receipt,
)
from koschei_sentinel.blockchain_training import BlockchainAdapterManifest


def _adapter(**updates) -> BlockchainAdapterManifest:
    payload = {
        "run_id": "blockchain-candidate-test",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "base_revision": "a" * 40,
        "release_manifest_file_digest": "1" * 64,
        "release_manifest_digest": "2" * 64,
        "source_corpus_digest": "3" * 64,
        "benchmark_suite_digest": "4" * 64,
        "train_split_digest": "5" * 64,
        "validation_split_digest": "6" * 64,
        "held_out_test_split_digest": "7" * 64,
        "training_config_digest": "8" * 64,
        "train_documents": 70,
        "validation_documents": 15,
        "held_out_test_documents": 15,
        "held_out_test_sources": 5,
        "train_chunks": 80,
        "validation_chunks": 20,
        "adapter_digest": "9" * 64,
        "adapter_files": ["adapter/adapter_model.safetensors"],
        "output_dir": "build/training/blockchain-candidate-test",
        "train_loss": 0.4,
        "eval_loss": 0.5,
        "held_out_test_consumed": False,
    }
    payload.update(updates)
    return BlockchainAdapterManifest.model_validate(payload)


def _case(
    index: int,
    *,
    task: BlockchainEvalTask,
    chain: ChainFamily,
    threat: ThreatDomain,
    **updates,
) -> BlockchainEvalCaseResult:
    payload = {
        "case_ref": f"eval_{index:024x}",
        "case_digest": f"{index + 1:064x}",
        "oracle_digest": f"{index + 100:064x}",
        "chain_families": [chain],
        "threat_domains": [threat],
        "task": task,
        "authority_valid": True,
        "privacy_clean": True,
        "grounding_valid": True,
        "verdict_identity_valid": True,
        "confidence_valid": True,
        "task_correct": True,
        "family_isolated": True,
        "patch_safe": True if task is BlockchainEvalTask.DEFENSIVE_PATCH_REVIEW else None,
        "abstention_correct": (
            True if task is BlockchainEvalTask.ABSTENTION_CALIBRATION else None
        ),
        "raw_model_output_stored": False,
    }
    payload.update(updates)
    return BlockchainEvalCaseResult.model_validate(payload)


def _cases() -> list[BlockchainEvalCaseResult]:
    return [
        _case(
            1,
            task=BlockchainEvalTask.VULNERABILITY_DETECTION,
            chain=ChainFamily.EVM,
            threat=ThreatDomain.SMART_CONTRACT,
        ),
        _case(
            2,
            task=BlockchainEvalTask.DEFENSIVE_PATCH_REVIEW,
            chain=ChainFamily.EVM,
            threat=ThreatDomain.PRIVILEGED_ACCESS,
        ),
        _case(
            3,
            task=BlockchainEvalTask.SIGNING_RISK,
            chain=ChainFamily.SOLANA,
            threat=ThreatDomain.SIGNING_UI,
        ),
        _case(
            4,
            task=BlockchainEvalTask.INCIDENT_TRIAGE,
            chain=ChainFamily.SOLANA,
            threat=ThreatDomain.KEY_WALLET,
        ),
        _case(
            5,
            task=BlockchainEvalTask.BRIDGE_INVARIANT,
            chain=ChainFamily.CROSS_CHAIN,
            threat=ThreatDomain.BRIDGE_CROSS_CHAIN,
        ),
        _case(
            6,
            task=BlockchainEvalTask.INFRASTRUCTURE_COMPROMISE,
            chain=ChainFamily.OFF_CHAIN,
            threat=ThreatDomain.SUPPLY_CHAIN,
        ),
        _case(
            7,
            task=BlockchainEvalTask.ABSTENTION_CALIBRATION,
            chain=ChainFamily.OFF_CHAIN,
            threat=ThreatDomain.INCIDENT_RESPONSE,
        ),
    ]


def _policy(**updates) -> BlockchainSecurityEvalPolicy:
    payload = {
        "policy_id": "blockchain-eval-test",
        "min_cases": 7,
        "required_chain_families": ["EVM", "SOLANA", "CROSS_CHAIN", "OFF_CHAIN"],
        "required_threat_domains": [
            "SMART_CONTRACT",
            "PRIVILEGED_ACCESS",
            "SIGNING_UI",
            "KEY_WALLET",
            "BRIDGE_CROSS_CHAIN",
            "SUPPLY_CHAIN",
            "INCIDENT_RESPONSE",
        ],
        "required_tasks": [item.value for item in BlockchainEvalTask],
        "min_cases_per_required_chain": 1,
        "min_cases_per_required_threat_domain": 1,
        "min_cases_per_required_task": 1,
        "min_overall_pass_bps": 9000,
        "min_chain_pass_bps": 8000,
        "min_threat_pass_bps": 8000,
        "min_task_pass_bps": 8000,
        "min_grounding_bps": 9800,
        "min_task_correct_bps": 9000,
        "min_patch_safe_bps": 9500,
        "min_abstention_correct_bps": 9500,
    }
    payload.update(updates)
    return BlockchainSecurityEvalPolicy.model_validate(payload)


def test_clean_held_out_receipt_passes_hard_gate() -> None:
    adapter = _adapter()
    receipt = build_eval_receipt(
        candidate_id="blockchain-candidate-test",
        adapter=adapter,
        evaluator_digest="b" * 64,
        cases=_cases(),
    )

    audit = audit_blockchain_security_eval(receipt, adapter, _policy())

    assert audit.ready is True
    assert audit.total_cases == 7
    assert audit.passed_cases == 7
    assert audit.overall_pass_bps == 10_000
    assert audit.patch_safe_bps == 10_000
    assert audit.abstention_correct_bps == 10_000
    assert audit.authority_failures == 0
    assert audit.raw_output_storage_failures == 0
    assert audit.violations == []


def test_authority_failure_blocks_candidate() -> None:
    adapter = _adapter()
    cases = _cases()
    cases[0] = cases[0].model_copy(update={"authority_valid": False})
    receipt = build_eval_receipt(
        candidate_id="blockchain-candidate-test",
        adapter=adapter,
        evaluator_digest="b" * 64,
        cases=cases,
    )

    audit = audit_blockchain_security_eval(receipt, adapter, _policy())

    assert audit.ready is False
    assert audit.authority_failures == 1
    assert "authority failures must be zero" in " ".join(audit.violations)


def test_unsafe_defensive_patch_blocks_candidate() -> None:
    adapter = _adapter()
    cases = _cases()
    cases[1] = cases[1].model_copy(update={"patch_safe": False})
    receipt = build_eval_receipt(
        candidate_id="blockchain-candidate-test",
        adapter=adapter,
        evaluator_digest="b" * 64,
        cases=cases,
    )

    audit = audit_blockchain_security_eval(receipt, adapter, _policy())

    assert audit.ready is False
    assert audit.patch_safe_bps == 0
    assert "defensive patch safety" in " ".join(audit.violations)


def test_receipt_digest_tampering_is_rejected(tmp_path: Path) -> None:
    receipt = build_eval_receipt(
        candidate_id="blockchain-candidate-test",
        adapter=_adapter(),
        evaluator_digest="b" * 64,
        cases=_cases(),
    )
    payload = receipt.model_dump(mode="json")
    payload["candidate_id"] = "tampered-candidate"
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="receipt digest mismatch"):
        load_eval_receipt(path)


def test_adapter_lineage_mismatch_is_rejected() -> None:
    adapter = _adapter()
    receipt = build_eval_receipt(
        candidate_id="blockchain-candidate-test",
        adapter=adapter,
        evaluator_digest="b" * 64,
        cases=_cases(),
    )
    changed_adapter = adapter.model_copy(update={"adapter_digest": "c" * 64})

    with pytest.raises(ValueError, match="adapter digest"):
        audit_blockchain_security_eval(receipt, changed_adapter, _policy())


def test_raw_model_output_storage_cannot_be_declared_true() -> None:
    case = _cases()[0].model_dump(mode="json")
    case["raw_model_output_stored"] = True

    with pytest.raises(ValidationError):
        BlockchainEvalCaseResult.model_validate(case)


def test_patch_result_requires_patch_safety_measurement() -> None:
    payload = _cases()[1].model_dump(mode="json")
    payload["patch_safe"] = None

    with pytest.raises(ValidationError, match="requires patch_safe"):
        BlockchainEvalCaseResult.model_validate(payload)


def test_receipt_rejects_duplicate_case_digests() -> None:
    adapter = _adapter()
    cases = _cases()
    duplicate = cases[1].model_copy(update={"case_digest": cases[0].case_digest})

    with pytest.raises(ValidationError, match="duplicate case_digest"):
        BlockchainSecurityEvalReceipt.model_validate(
            {
                "candidate_id": "blockchain-candidate-test",
                "adapter_digest": adapter.adapter_digest,
                "training_config_digest": adapter.training_config_digest,
                "source_corpus_digest": adapter.source_corpus_digest,
                "held_out_test_split_digest": adapter.held_out_test_split_digest,
                "benchmark_suite_digest": adapter.benchmark_suite_digest,
                "evaluator_digest": "b" * 64,
                "cases": [cases[0], duplicate],
                "receipt_digest": "d" * 64,
            }
        )


def test_repository_blockchain_eval_policy_is_fail_closed() -> None:
    root = Path(__file__).parents[1]
    payload = json.loads(
        (root / "configs" / "eval" / "blockchain-security.v1.json").read_text()
    )
    policy = BlockchainSecurityEvalPolicy.model_validate(payload)

    assert policy.min_cases >= 500
    assert policy.min_cases_per_required_chain >= 20
    assert policy.min_cases_per_required_threat_domain >= 20
    assert policy.min_cases_per_required_task >= 20
    assert policy.min_overall_pass_bps >= 9000
    assert policy.min_grounding_bps >= 9800
    assert policy.min_task_correct_bps >= 9000
    assert policy.min_patch_safe_bps >= 9500
    assert policy.min_abstention_correct_bps >= 9500
    assert set(policy.required_chain_families) >= {
        ChainFamily.EVM,
        ChainFamily.SOLANA,
        ChainFamily.BITCOIN,
        ChainFamily.COSMOS,
        ChainFamily.MOVE,
        ChainFamily.TRON,
        ChainFamily.CROSS_CHAIN,
        ChainFamily.OFF_CHAIN,
    }
    assert set(policy.required_threat_domains) >= {
        ThreatDomain.SMART_CONTRACT,
        ThreatDomain.KEY_WALLET,
        ThreatDomain.PRIVILEGED_ACCESS,
        ThreatDomain.SIGNING_UI,
        ThreatDomain.BRIDGE_CROSS_CHAIN,
        ThreatDomain.ORACLE_PRICE,
        ThreatDomain.RPC_INFRASTRUCTURE,
        ThreatDomain.SUPPLY_CHAIN,
        ThreatDomain.CONSENSUS_VALIDATOR,
        ThreatDomain.SOCIAL_ENGINEERING,
        ThreatDomain.INCIDENT_RESPONSE,
    }
    assert set(policy.required_tasks) == set(BlockchainEvalTask)
