import json

from koschei_sentinel.cyber_sft_trainer import _language_lora_targets
from koschei_sentinel.cyber_sft_training import (
    CyberSFTConfig,
    cyber_sft_messages,
    plan_cyber_sft,
    split_cyber_sft_examples,
)
from koschei_sentinel.defense_reflex_corpus_v2 import (
    DefenseReflexTrainingExampleV2,
    build_defense_reflex_v2_manifest,
    serialize_defense_reflex_v2,
)


def _example(index: int) -> DefenseReflexTrainingExampleV2:
    digit = format(index % 16, "x")
    return DefenseReflexTrainingExampleV2(
        example_id=f"defense-reflex-v2:test:{index}",
        scenario_id=f"scenario:{index}",
        failure_type="NO_CONTAINMENT",
        source_report_sha256=digit * 64,
        correction_sha256=format((index + 1) % 16, "x") * 64,
        graph_snapshots=[
            {
                "graph_id": f"graph:{index}",
                "entities": [{"entity_id": f"wallet:{index}"}],
                "relations": [],
            }
        ],
        critical_entity_ids=[f"wallet:{index}"],
        observed_ticks=[
            {
                "tick": 0,
                "defense_mode": "GUARD",
                "attack_confidence": 0.4,
            }
        ],
        corrected_interpretation="Independent evidence confirms the protected path needs containment.",
        expected_sequence=[
            {
                "sequence": 1,
                "expected_mode": "COMBAT",
                "action": "FREEZE_SIGNER",
                "target_entity_id": f"wallet:{index}",
                "rationale": "Stop hostile signing on the protected path.",
                "supporting_evidence_ids": [f"evidence:{index}"],
                "outcome_verification_ids": [f"outcome:{index}"],
            }
        ],
        review_evidence_ids=[f"review:{index}"],
        provenance={"reviewer_id": "reviewer:test"},
    )


def _config() -> CyberSFTConfig:
    return CyberSFTConfig(
        run_id="cyber-sft-test",
        stage="DEFENSE_REFLEX",
        base_model="Qwen/Qwen3.5-9B-Base",
        base_revision="a" * 40,
        corpus_dir="build/reflex-v2",
        output_dir="build/run-test",
        validation_ratio=0.25,
        minimum_cuda_memory_gb=0.0,
    )


def test_cyber_sft_plan_verifies_v2_corpus_digest(tmp_path) -> None:
    examples = [_example(index) for index in range(4)]
    payload = serialize_defense_reflex_v2(examples)
    manifest = build_defense_reflex_v2_manifest(examples)
    corpus = tmp_path / "build" / "reflex-v2"
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


def test_cyber_sft_split_is_deterministic() -> None:
    examples = [_example(index) for index in range(20)]
    first = split_cyber_sft_examples(examples, validation_ratio=0.2, seed=1701)
    second = split_cyber_sft_examples(list(reversed(examples)), validation_ratio=0.2, seed=1701)
    assert [row.example_id for row in first[0]] == [row.example_id for row in second[0]]
    assert [row.example_id for row in first[1]] == [row.example_id for row in second[1]]


def test_defense_reflex_prompt_does_not_leak_reviewed_answer() -> None:
    example = _example(1)
    messages = cyber_sft_messages(example)
    assert len(messages) == 3
    assert example.corrected_interpretation not in messages[1]["content"]
    assert example.corrected_interpretation in messages[2]["content"]
    assert "wallet:1" in messages[1]["content"]


def test_lora_target_resolution_excludes_vision_modules() -> None:
    class FakeModel:
        def named_modules(self):
            return [
                ("model.language_model.layers.0.self_attn.q_proj", object()),
                ("model.language_model.layers.0.mlp.up_proj", object()),
                ("model.visual.blocks.0.attn.q_proj", object()),
            ]

    targets = _language_lora_targets(FakeModel(), ["q_proj", "up_proj"])
    assert targets == [
        "model.language_model.layers.0.mlp.up_proj",
        "model.language_model.layers.0.self_attn.q_proj",
    ]
    assert all("visual" not in row for row in targets)
