from __future__ import annotations

import json
from pathlib import Path

import pytest

from koschei_sentinel.language_foundation import build_language_foundation_release
from koschei_sentinel.language_training import (
    LanguageTrainingConfig,
    _tokenize_documents,
    plan_language_training,
    write_language_training_plan,
)
from koschei_sentinel.training import LoraConfig, QuantizationConfig

FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures/language_foundation/language-foundation-contract-vector.v1.json"
)


def _release(root: Path) -> tuple[Path, dict[str, object]]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    release = root / "language-release"
    build_language_foundation_release(
        FIXTURE,
        output_dir=release,
        expected_source_commit=str(payload["source_commit"]),
        expected_source_corpus_sha256=str(payload["corpus_sha256"]),
    )
    return release, payload


def _config(root: Path, release: Path, payload: dict[str, object]) -> LanguageTrainingConfig:
    return LanguageTrainingConfig(
        run_id="koschei-language-test",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        base_revision="b" * 40,
        language_release=release.relative_to(root).as_posix(),
        expected_source_commit=str(payload["source_commit"]),
        expected_source_corpus_sha256=str(payload["corpus_sha256"]),
        output_dir="build/language-training/test",
        max_sequence_length=512,
        quantization=QuantizationConfig(compute_dtype="float16"),
        lora=LoraConfig(),
    )


def test_plan_binds_exact_language_source_and_holds_out_test(tmp_path: Path) -> None:
    release, payload = _release(tmp_path)
    config = _config(tmp_path, release, payload)

    plan = plan_language_training(config, root=tmp_path)

    assert plan.lineage_stage == "stage1_language_foundation"
    assert plan.authority == "offline_language_research_only"
    assert plan.source_repository == "bugsbuny243/koschei-lang"
    assert plan.source_commit == payload["source_commit"]
    assert plan.source_corpus_sha256 == payload["corpus_sha256"]
    assert plan.train_documents > 0
    assert plan.validation_documents > 0
    assert plan.test_documents > 0
    assert plan.test_split_digest not in {
        plan.train_split_digest,
        plan.validation_split_digest,
    }


def test_plan_rejects_wrong_language_commit_pin(tmp_path: Path) -> None:
    release, payload = _release(tmp_path)
    config = _config(tmp_path, release, payload).model_copy(
        update={"expected_source_commit": "c" * 40}
    )

    with pytest.raises(ValueError, match="source commit"):
        plan_language_training(config, root=tmp_path)


def test_plan_refuses_existing_training_output(tmp_path: Path) -> None:
    release, payload = _release(tmp_path)
    config = _config(tmp_path, release, payload)
    output = tmp_path / config.output_dir
    output.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="output already exists"):
        plan_language_training(config, root=tmp_path)


def test_plan_file_is_no_replace(tmp_path: Path) -> None:
    release, payload = _release(tmp_path)
    config = _config(tmp_path, release, payload)
    plan = plan_language_training(config, root=tmp_path)
    destination = tmp_path / "plan.json"

    write_language_training_plan(plan, destination)
    with pytest.raises(FileExistsError, match="plan already exists"):
        write_language_training_plan(plan, destination)


class _Tokenizer:
    eos_token_id = 99

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert not add_special_tokens
        return {"input_ids": list(range(1, len(text) + 1))}


def test_document_tokenization_keeps_documents_bounded() -> None:
    from koschei_sentinel.language_foundation import LanguageFoundationDocument

    document = LanguageFoundationDocument(
        schema_version="sentinel.language-foundation-document.v1",
        document_id="a" * 64,
        family="example:demo",
        kind="koschei_source",
        path="examples/demo/main.ks",
        source_sha256="b" * 64,
        text="abcdef",
    )

    rows = _tokenize_documents([document], _Tokenizer(), 4)

    assert rows == [
        {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]},
        {"input_ids": [5, 6, 99], "attention_mask": [1, 1, 1]},
    ]
