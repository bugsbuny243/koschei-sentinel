from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import koschei_sentinel.blockchain_training as training_module
from koschei_sentinel.blockchain_security_corpus import (
    BlockchainSecurityDocument,
    BlockchainSourceClass,
    ChainFamily,
    ThreatDomain,
    content_digest,
)
from koschei_sentinel.blockchain_security_release import (
    BlockchainSecurityReleaseManifest,
    BlockchainSecurityReleaseSplit,
)
from koschei_sentinel.blockchain_training import (
    BlockchainTrainingConfig,
    _render_training_text,
    _tokenize_documents,
    _verify_execution_lineage,
    plan_blockchain_training,
)
from koschei_sentinel.pretraining_corpus import RightsBasis


def _document(index: int, split: str) -> BlockchainSecurityDocument:
    character = "0123456789abcdef"[index]
    text = f"{split} blockchain security material {index}.\n"
    return BlockchainSecurityDocument(
        document_ref=f"doc_{character * 24}",
        source_class=BlockchainSourceClass.PROTOCOL_SOURCE,
        rights_basis=RightsBasis.KOSCHEI_OWNED,
        source_snapshot_digest=character * 64,
        content_digest=content_digest(text),
        family_refs=[f"family_{character * 24}"],
        chain_families=[ChainFamily.EVM],
        threat_domains=[ThreatDomain.PRIVILEGED_ACCESS],
        text=text,
    )


def _payload(documents: list[BlockchainSecurityDocument]) -> str:
    return "".join(
        json.dumps(
            item.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
        for item in documents
    )


def _materialize_release(root: Path) -> tuple[Path, BlockchainSecurityReleaseManifest]:
    release = root / "release"
    release.mkdir()
    split_documents = {
        "train": [_document(0, "train"), _document(1, "train")],
        "validation": [_document(2, "validation"), _document(3, "validation")],
        "test": [_document(4, "test"), _document(5, "test")],
    }
    reports: dict[str, BlockchainSecurityReleaseSplit] = {}
    all_documents: list[BlockchainSecurityDocument] = []
    for split_name, documents in split_documents.items():
        ordered = sorted(documents, key=lambda item: item.document_ref)
        payload = _payload(ordered)
        (release / f"{split_name}.jsonl").write_text(payload, encoding="utf-8")
        snapshots = sorted(item.source_snapshot_digest for item in ordered)
        reports[split_name] = BlockchainSecurityReleaseSplit(
            path=f"{split_name}.jsonl",
            documents=len(ordered),
            sources=len(snapshots),
            components=len(snapshots),
            families=len(ordered),
            digest=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            source_snapshot_digests=snapshots,
            chain_source_counts={"EVM": len(snapshots)},
            threat_source_counts={"PRIVILEGED_ACCESS": len(snapshots)},
        )
        all_documents.extend(ordered)

    ordered_all = sorted(all_documents, key=lambda item: item.document_ref)
    source_digest = hashlib.sha256(_payload(ordered_all).encode("utf-8")).hexdigest()
    manifest = BlockchainSecurityReleaseManifest(
        release_policy_id="training-test",
        split_seed="training-test-seed",
        catalog_file_digest="1" * 64,
        catalog_policy_file_digest="2" * 64,
        source_catalog_manifest_file_digest="3" * 64,
        source_catalog_digest="4" * 64,
        source_corpus_file_digest=source_digest,
        source_corpus_digest=source_digest,
        corpus_audit_file_digest="5" * 64,
        corpus_audit_digest="6" * 64,
        holdout_file_digest="7" * 64,
        holdout_digest="8" * 64,
        benchmark_suite_digest="9" * 64,
        blockchain_policy_file_digest="a" * 64,
        blockchain_policy_digest="b" * 64,
        release_policy_file_digest="c" * 64,
        release_policy_digest="d" * 64,
        documents=6,
        sources=6,
        components=6,
        family_leakage_detected=False,
        source_leakage_detected=False,
        splits=reports,
    )
    (release / "blockchain-security-release-manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return release, manifest


def _config(manifest: BlockchainSecurityReleaseManifest) -> BlockchainTrainingConfig:
    return BlockchainTrainingConfig(
        run_id="blockchain-training-test",
        base_model="owner/model",
        base_revision="e" * 40,
        blockchain_release="release",
        expected_source_corpus_digest=manifest.source_corpus_digest,
        expected_benchmark_suite_digest=manifest.benchmark_suite_digest,
        output_dir="training-output",
        max_sequence_length=512,
        per_device_batch_size=1,
        gradient_accumulation_steps=2,
        epochs=1.0,
    )


def test_plan_binds_exact_release_and_held_out_test(tmp_path: Path) -> None:
    _, manifest = _materialize_release(tmp_path)
    config = _config(manifest)

    plan = plan_blockchain_training(config, root=tmp_path)

    assert plan.lineage_stage == "stage2_blockchain_continued_pretraining"
    assert plan.authority == "offline_blockchain_research_only"
    assert plan.train_documents == 2
    assert plan.validation_documents == 2
    assert plan.held_out_test_documents == 2
    assert plan.held_out_test_sources == 2
    assert plan.train_split_digest == manifest.splits["train"].digest
    assert plan.validation_split_digest == manifest.splits["validation"].digest
    assert plan.held_out_test_split_digest == manifest.splits["test"].digest
    assert plan.benchmark_suite_digest == manifest.benchmark_suite_digest


def test_execution_lineage_does_not_open_or_hash_test_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, manifest = _materialize_release(tmp_path)
    config = _config(manifest)
    plan = plan_blockchain_training(config, root=tmp_path)
    seen: list[str] = []
    original = training_module._hash_file

    def recording_hash(path: Path) -> str:
        seen.append(path.name)
        return original(path)

    monkeypatch.setattr(training_module, "_hash_file", recording_hash)
    release, stored = _verify_execution_lineage(config, plan, root=tmp_path)

    assert release == tmp_path / "release"
    assert stored == manifest
    assert "train.jsonl" in seen
    assert "validation.jsonl" in seen
    assert "test.jsonl" not in seen


def test_training_text_omits_provenance_identifiers() -> None:
    document = _document(0, "train")
    rendered = _render_training_text(document)

    assert "EVM" in rendered
    assert "PRIVILEGED_ACCESS" in rendered
    assert document.text in rendered
    assert document.document_ref not in rendered
    assert document.source_snapshot_digest not in rendered
    assert document.family_refs[0] not in rendered


def test_tokenization_chunks_without_touching_test_contract() -> None:
    class Tokenizer:
        eos_token_id = 999

        def __call__(self, text: str, *, add_special_tokens: bool):
            assert add_special_tokens is False
            return {"input_ids": list(range(1100))}

    rows = _tokenize_documents([_document(0, "train")], Tokenizer(), 512)

    assert [len(row["input_ids"]) for row in rows] == [512, 512, 77]
    assert rows[-1]["input_ids"][-1] == 999


def test_wrong_release_pin_blocks_training_plan(tmp_path: Path) -> None:
    _, manifest = _materialize_release(tmp_path)
    config = _config(manifest).model_copy(
        update={"expected_source_corpus_digest": "f" * 64}
    )

    with pytest.raises(ValueError, match="source corpus digest"):
        plan_blockchain_training(config, root=tmp_path)


def test_existing_output_blocks_training_plan(tmp_path: Path) -> None:
    _, manifest = _materialize_release(tmp_path)
    config = _config(manifest)
    (tmp_path / "training-output").mkdir()

    with pytest.raises(FileExistsError, match="output already exists"):
        plan_blockchain_training(config, root=tmp_path)
