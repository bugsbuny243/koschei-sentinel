from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from koschei_sentinel.dataset import DatasetExample
from koschei_sentinel.trainer import _fit_supervised_messages, _tokenize
from koschei_sentinel.training import (
    TrainingConfig,
    load_release_examples,
    plan_training,
    supervised_messages,
)

_ROW = {
    "schema_version": "sentinel.dataset.v1",
    "example_id": "example_" + "a" * 24,
    "group_ref": "group_" + "b" * 24,
    "source_digest": "c" * 64,
    "case": {
        "schema_version": "sentinel.case.v1",
        "case_id": "case_" + "d" * 24,
        "target_ref": "target_" + "e" * 24,
        "network": "solana-mainnet",
        "signed_verdict": {
            "grade": "D",
            "signature": "signature_" + "f" * 24,
            "triggered_rules": ["KS-HOLDER-001"],
            "summary": "A deterministic holder concentration condition was found.",
        },
        "evidence": [
            {
                "evidence_id": "evidence_" + "1" * 24,
                "kind": "holder_intelligence",
                "statement": "The largest holder controls a material supply share.",
                "confidence": "VERIFIED",
                "rule_ids": ["KS-HOLDER-001"],
                "attributes": {"share_bps": 4200},
            }
        ],
        "limitations": ["No live market data was supplied."],
    },
}


class _CharacterTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def __call__(self, text: str, **_: Any) -> dict[str, list[int]]:
        values = list(range(len(text)))
        return {"input_ids": values, "attention_mask": [1] * len(values)}


def _config() -> TrainingConfig:
    return TrainingConfig(
        run_id="fixture-v0.6",
        base_model="sentinel-fixture/model",
        base_revision="0" * 40,
        dataset_release="fixtures/training/release",
        output_dir="build/training/fixture-v0.6",
    )


def _write_release(root: Path) -> None:
    release = root / "fixtures/training/release"
    release.mkdir(parents=True)
    payloads = {
        "train": json.dumps(_ROW, sort_keys=True, separators=(",", ":")) + "\n",
        "validation": "",
        "test": "",
    }
    reports = {}
    for split_name, payload in payloads.items():
        (release / f"{split_name}.jsonl").write_text(payload, encoding="utf-8")
        populated = split_name == "train"
        reports[split_name] = {
            "examples": 1 if populated else 0,
            "groups": 1 if populated else 0,
            "digest": hashlib.sha256(payload.encode()).hexdigest(),
        }
    manifest = {
        "schema_version": "sentinel.quality-manifest.v1",
        "dry_run": False,
        "seed": "fixture",
        "train_bps": 8000,
        "validation_bps": 1000,
        "test_bps": 1000,
        "total_examples": 1,
        "total_groups": 1,
        "splits": reports,
        "grade_counts": {"D": 1},
        "confidence_counts": {"VERIFIED": 1},
        "evidence_kind_counts": {"holder_intelligence": 1},
        "warnings": ["validation split is empty", "test split is empty"],
    }
    (release / "quality-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _long_example() -> DatasetExample:
    rules = [f"KS-LONG-{index:03d}" for index in range(4)]
    evidence = []
    for rule_index, rule_id in enumerate(rules):
        for item_index in range(4):
            evidence.append(
                {
                    "evidence_id": f"evidence_{rule_index}_{item_index}_" + "x" * 16,
                    "kind": "bounded_module_evidence",
                    "statement": (
                        f"Evidence for {rule_id} item {item_index}. " + "detail " * 140
                    ),
                    "confidence": "VERIFIED" if item_index == 0 else "INFERRED",
                    "rule_ids": [rule_id],
                    "attributes": {
                        f"attribute_{index}": "value " * 40 for index in range(10)
                    },
                }
            )
    payload = {
        **_ROW,
        "example_id": "example_long_" + "a" * 24,
        "group_ref": "group_long_" + "b" * 24,
        "case": {
            **_ROW["case"],
            "case_id": "case_long_" + "d" * 24,
            "signed_verdict": {
                **_ROW["case"]["signed_verdict"],
                "triggered_rules": rules,
                "summary": "Long deterministic summary. " + "summary " * 300,
            },
            "evidence": evidence,
            "limitations": ["limitation " * 120 for _ in range(12)],
        },
    }
    return DatasetExample.model_validate(payload)


def test_training_plan_validates_release_and_estimates_steps(tmp_path: Path) -> None:
    _write_release(tmp_path)

    plan = plan_training(_config(), root=tmp_path)

    assert plan.splits["train"].examples == 1
    assert plan.estimated_optimizer_steps == 1
    assert plan.warnings == ["validation split is empty", "test split is empty"]


def test_training_plan_rejects_release_digest_drift(tmp_path: Path) -> None:
    _write_release(tmp_path)
    train_path = tmp_path / "fixtures/training/release/train.jsonl"
    train_path.write_text(train_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        plan_training(_config(), root=tmp_path)


def test_training_config_rejects_path_escape_and_mutable_revision() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(
            run_id="bad-path",
            base_model="owner/model",
            base_revision="0" * 40,
            dataset_release="../private",
            output_dir="build/training/bad",
        )
    with pytest.raises(ValidationError):
        TrainingConfig(
            run_id="bad-revision",
            base_model="owner/model",
            base_revision="main",
            dataset_release="build/releases/safe",
            output_dir="build/training/bad",
        )


def test_supervision_is_policy_grounded_and_timestamp_free(tmp_path: Path) -> None:
    _write_release(tmp_path)
    rows, _ = load_release_examples(
        tmp_path.resolve(), "fixtures/training/release/train.jsonl"
    )

    messages = supervised_messages(rows[0])
    answer = json.loads(messages[-1]["content"])

    assert [item["role"] for item in messages] == ["system", "user", "assistant"]
    assert answer["assessment"] == "EXPLANATION_ONLY"
    assert answer["claims"][0]["evidence_ids"] == ["evidence_" + "1" * 24]
    assert "generated_at" not in answer


def test_sequence_budget_compacts_without_losing_citation_grounding() -> None:
    tokenizer = _CharacterTokenizer()
    example = _long_example()

    original = supervised_messages(example)
    original_full = tokenizer.apply_chat_template(
        original,
        tokenize=False,
        add_generation_prompt=False,
    )
    assert len(original_full) > 4000

    messages = _fit_supervised_messages(example, tokenizer, 4000)
    compact_case = json.loads(messages[1]["content"])
    compact_answer = json.loads(messages[-1]["content"])
    compact_full = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    known_ids = {item["evidence_id"] for item in compact_case["evidence"]}
    cited_ids = {
        evidence_id
        for claim in compact_answer["claims"]
        for evidence_id in claim["evidence_ids"]
    }

    assert len(compact_full) <= 4000
    assert len(compact_case["evidence"]) < len(example.case.evidence)
    assert cited_ids
    assert cited_ids <= known_ids
    assert len(compact_answer["claims"]) == 4

    encoded = _tokenize(example, tokenizer, 4000)
    assert len(encoded["input_ids"]) <= 4000
    first_answer_token = next(
        index for index, value in enumerate(encoded["labels"]) if value != -100
    )
    assert first_answer_token > 0
    assert all(value == -100 for value in encoded["labels"][:first_answer_token])
