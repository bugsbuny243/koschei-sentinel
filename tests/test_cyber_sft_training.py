import json

import pytest

from koschei_sentinel.cyber_sft_trainer import _language_lora_targets, execute_cyber_sft
from koschei_sentinel.cyber_sft_training import (
    CyberExecutionProfile,
    CyberSFTConfig,
    CyberSFTPlan,
    cyber_sft_messages,
    plan_cyber_sft,
    split_cyber_sft_examples,
)
from koschei_sentinel.defense_reflex_corpus_v3 import (
    DefenseReflexTrainingExampleV3,
    build_defense_reflex_v3_manifest,
    serialize_defense_reflex_v3,
)


def _example(index: int) -> DefenseReflexTrainingExampleV3:
    digit = format(index % 16, "x")
    evidence_id = f"evidence:scenario:{index}:observed"
    return DefenseReflexTrainingExampleV3(
        example_id=f"defense-reflex-v3:test:{index}",
        scenario_id=f"scenario:{index}",
        source_report_sha256=digit * 64,
        review_sha256=format((index + 1) % 16, "x") * 64,
        graph_snapshots=[
            {
                "schema_version": "sentinel.cyber-state-graph.v1",
                "graph_id": f"graph:{index}",
                "entities": [
                    {"entity_id": f"device:{index}", "entity_type": "DEVICE", "labels": {}},
                    {"entity_id": f"wallet:{index}", "entity_type": "WALLET", "labels": {}},
                ],
                "relations": [
                    {
                        "relation_id": f"relation:{index}",
                        "source_entity_id": f"device:{index}",
                        "target_entity_id": f"wallet:{index}",
                        "relation_type": "reaches_signer",
                        "status": "OBSERVED",
                        "confidence": 0.95,
                        "evidence": [
                            {
                                "evidence_id": evidence_id,
                                "source": "test-sensor",
                                "content_sha256": "a" * 64,
                                "timestamp": None,
                            }
                        ],
                        "rationale": None,
                    }
                ],
            }
        ],
        critical_entity_ids=[f"wallet:{index}"],
        observed_ticks=[
            {
                "tick": 0,
                "defense_mode": "COMBAT",
                "attack_confidence": 0.95,
            }
        ],
        expected_interpretation=(
            "Observed signer reach on the protected path requires evidence-grounded containment."
        ),
        expected_sequence=[
            {
                "sequence": 1,
                "expected_mode": "COMBAT",
                "action": "FREEZE_SIGNER",
                "target_entity_id": f"wallet:{index}",
                "rationale": "Stop hostile signing on the protected path.",
                "supporting_evidence_ids": [evidence_id],
                "outcome_verification_required": True,
            }
        ],
        provenance={
            "review_method": "SYNTHETIC_POLICY",
            "reviewer_id": "policy:test",
        },
        promotion_eligible=False,
    )


def _config() -> CyberSFTConfig:
    return CyberSFTConfig(
        run_id="cyber-sft-test",
        stage="DEFENSE_REFLEX",
        execution_profile=CyberExecutionProfile.DENSE_SINGLE_GPU_QLORA,
        base_model="Qwen/Qwen3.5-9B-Base",
        base_revision="a" * 40,
        corpus_dir="build/reflex-v3",
        output_dir="build/run-test",
        validation_ratio=0.25,
        minimum_cuda_memory_gb=0.0,
    )


def test_cyber_sft_plan_verifies_v3_corpus_digest(tmp_path) -> None:
    examples = [_example(index) for index in range(4)]
    payload = serialize_defense_reflex_v3(examples)
    manifest = build_defense_reflex_v3_manifest(examples)
    corpus = tmp_path / "build" / "reflex-v3"
    corpus.mkdir(parents=True)
    (corpus / "examples.jsonl").write_text(payload, encoding="utf-8")
    (corpus / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )

    plan = plan_cyber_sft(_config(), root=tmp_path)
    assert plan.example_count == 4
    assert plan.training_examples == 3
    assert plan.validation_examples == 1
    assert plan.corpus_examples_sha256 == manifest.examples_sha256
    assert plan.corpus_promotion_eligible is False
    assert any("not promotion-eligible" in warning for warning in plan.warnings)


def test_cyber_sft_split_is_deterministic() -> None:
    examples = [_example(index) for index in range(20)]
    first = split_cyber_sft_examples(examples, validation_ratio=0.2, seed=1701)
    second = split_cyber_sft_examples(list(reversed(examples)), validation_ratio=0.2, seed=1701)
    assert [row.example_id for row in first[0]] == [row.example_id for row in second[0]]
    assert [row.example_id for row in first[1]] == [row.example_id for row in second[1]]


def test_defense_reflex_prompt_contains_no_hidden_review_or_outcome_labels() -> None:
    example = _example(1)
    messages = cyber_sft_messages(example)
    assert len(messages) == 3
    user = messages[1]["content"]
    answer = messages[2]["content"]

    assert example.expected_interpretation not in user
    assert example.expected_interpretation in answer
    assert "wallet:1" in user
    assert "failure_type" not in user
    assert "review_method" not in user
    assert "reviewer_id" not in user
    assert "outcome" not in user.lower()
    assert "outcome_verification_ids" not in answer
    assert "outcome_verification_required" in answer


def test_target_supporting_evidence_exists_in_input_graph() -> None:
    example = _example(2)
    present = {
        evidence["evidence_id"]
        for graph in example.graph_snapshots
        for relation in graph["relations"]
        for evidence in relation["evidence"]
    }
    requested = {
        evidence_id
        for step in example.expected_sequence
        for evidence_id in step["supporting_evidence_ids"]
    }
    assert requested
    assert requested.issubset(present)


def test_lora_target_resolution_covers_hybrid_language_modules_and_excludes_vision() -> None:
    class FakeModel:
        def named_modules(self):
            return [
                ("model.language_model.layers.0.self_attn.q_proj", object()),
                ("model.language_model.layers.0.mlp.up_proj", object()),
                ("model.language_model.layers.1.linear_attn.in_proj_qkv", object()),
                ("model.language_model.layers.1.linear_attn.out_proj", object()),
                ("model.visual.blocks.0.attn.q_proj", object()),
            ]

    targets = _language_lora_targets(
        FakeModel(),
        ["q_proj", "up_proj", "in_proj_qkv", "out_proj"],
    )
    assert targets == [
        "model.language_model.layers.0.mlp.up_proj",
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.1.linear_attn.in_proj_qkv",
        "model.language_model.layers.1.linear_attn.out_proj",
    ]
    assert all("visual" not in row for row in targets)


def test_current_trainer_refuses_moe_profile_before_loading_gpu_dependencies() -> None:
    config = CyberSFTConfig(
        run_id="cyber-sft-moe-test",
        stage="DEFENSE_REFLEX",
        execution_profile=CyberExecutionProfile.MOE_DISTRIBUTED_REQUIRED,
        base_model="Qwen/Qwen3.5-35B-A3B-Base",
        base_revision="b" * 40,
        corpus_dir="build/reflex-v3",
        output_dir="build/moe-test",
        gradient_checkpointing=False,
        enable_router_aux_loss=True,
    )
    plan = CyberSFTPlan(
        run_id=config.run_id,
        stage=config.stage,
        execution_profile=config.execution_profile,
        executable_with_current_trainer=False,
        corpus_promotion_eligible=False,
        base_model=config.base_model,
        base_revision=config.base_revision,
        corpus_examples_sha256="a" * 64,
        corpus_manifest_sha256="b" * 64,
        example_count=32,
        training_examples=29,
        validation_examples=3,
        effective_batch_size=16,
        estimated_optimizer_steps=2,
        input_adapter_dir=None,
        output_dir=config.output_dir,
        warnings=[],
    )
    with pytest.raises(RuntimeError, match="distributed MoE executor"):
        execute_cyber_sft(config, plan)
