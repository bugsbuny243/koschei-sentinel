from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from koschei_sentinel.continued_pretraining import (
    ContinuedPretrainingConfig,
    plan_continued_pretraining,
)
from koschei_sentinel.pretraining_corpus import (
    PretrainingCorpusPolicy,
    PretrainingDocument,
    PretrainingHoldoutSet,
    PretrainingSourceClass,
    RightsBasis,
    audit_pretraining_corpus,
    content_digest,
    write_pretraining_audit,
)


def pseudonym(prefix: str, character: str) -> str:
    return f"{prefix}_{character * 24}"


def documents() -> list[PretrainingDocument]:
    rows: list[PretrainingDocument] = []
    source_classes = [
        PretrainingSourceClass.KOSCHEI_SECURITY_CASE,
        PretrainingSourceClass.SOLANA_PROTOCOL,
        PretrainingSourceClass.PUBLIC_SECURITY_REPORT,
    ]
    rights = [
        RightsBasis.KOSCHEI_OWNED,
        RightsBasis.APACHE_2_0,
        RightsBasis.CC_BY_4_0,
    ]
    for index in range(3):
        text = f"Sanitized Stage 2 Solana security corpus document {index}."
        rows.append(
            PretrainingDocument(
                document_ref=pseudonym("doc", "abc"[index]),
                source_class=source_classes[index],
                rights_basis=rights[index],
                source_snapshot_digest=format(index + 1, "064x"),
                content_digest=content_digest(text),
                family_refs=[pseudonym("family", "def"[index])],
                text=text,
            )
        )
    return rows


def holdout() -> PretrainingHoldoutSet:
    return PretrainingHoldoutSet(benchmark_suite_digest="a" * 64)


def policy(*, min_documents: int = 3) -> PretrainingCorpusPolicy:
    return PretrainingCorpusPolicy(
        policy_id="stage2-plan-test",
        min_documents=min_documents,
        min_source_classes=3,
        min_unique_families=3,
        max_single_family_bps=4000,
    )


def materialize_inputs(
    root: Path,
    *,
    corpus_policy: PretrainingCorpusPolicy | None = None,
) -> ContinuedPretrainingConfig:
    rows = documents()
    active_policy = corpus_policy or policy()
    active_holdout = holdout()

    corpus_path = root / "corpus.jsonl"
    corpus_path.write_text(
        "".join(item.model_dump_json() + "\n" for item in rows),
        encoding="utf-8",
    )
    (root / "holdout.json").write_text(
        active_holdout.model_dump_json(),
        encoding="utf-8",
    )
    (root / "policy.json").write_text(
        active_policy.model_dump_json(),
        encoding="utf-8",
    )
    audit = audit_pretraining_corpus(rows, active_holdout, active_policy)
    write_pretraining_audit(audit, root / "audit.json")

    return ContinuedPretrainingConfig(
        run_id="stage2-test-run",
        base_model="Qwen/Qwen2.5-1.5B",
        base_revision="b" * 40,
        corpus_path="corpus.jsonl",
        corpus_audit_path="audit.json",
        holdout_path="holdout.json",
        policy_path="policy.json",
        output_dir="stage2-output",
        per_device_batch_size=1,
        gradient_accumulation_steps=2,
        epochs=2.0,
    )


def test_plan_binds_exact_passing_corpus_lineage(tmp_path: Path) -> None:
    config = materialize_inputs(tmp_path)
    plan = plan_continued_pretraining(config, root=tmp_path)

    assert plan.lineage_stage == "stage2_continued_pretraining"
    assert plan.dry_run is True
    assert plan.documents == 3
    assert plan.unique_families == 3
    assert plan.benchmark_suite_digest == "a" * 64
    assert plan.effective_batch_size == 2
    assert plan.estimated_optimizer_steps == 4
    assert plan.corpus_file_digest
    assert plan.corpus_digest
    assert plan.corpus_audit_file_digest
    assert plan.training_config_digest
    assert not (tmp_path / "stage2-output").exists()


def test_stale_audit_after_policy_change_blocks_plan(tmp_path: Path) -> None:
    config = materialize_inputs(tmp_path)
    changed = policy()
    changed = changed.model_copy(update={"policy_id": "stage2-plan-test-v2"})
    (tmp_path / "policy.json").write_text(changed.model_dump_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="stale or does not match"):
        plan_continued_pretraining(config, root=tmp_path)


def test_corpus_drift_after_audit_blocks_plan(tmp_path: Path) -> None:
    config = materialize_inputs(tmp_path)
    rows = documents()
    payload = rows[0].model_dump(mode="json")
    payload["text"] = "Changed after audit."
    (tmp_path / "corpus.jsonl").write_text(
        json.dumps(payload) + "\n"
        + "".join(item.model_dump_json() + "\n" for item in rows[1:]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid pretraining corpus row"):
        plan_continued_pretraining(config, root=tmp_path)


def test_failed_corpus_audit_cannot_create_stage2_plan(tmp_path: Path) -> None:
    config = materialize_inputs(tmp_path, corpus_policy=policy(min_documents=10))
    with pytest.raises(ValueError, match="passing corpus audit"):
        plan_continued_pretraining(config, root=tmp_path)


def test_existing_output_directory_blocks_plan(tmp_path: Path) -> None:
    config = materialize_inputs(tmp_path)
    (tmp_path / "stage2-output").mkdir()
    with pytest.raises(FileExistsError, match="output already exists"):
        plan_continued_pretraining(config, root=tmp_path)


def test_stage2_config_requires_immutable_base_revision() -> None:
    with pytest.raises(ValidationError):
        ContinuedPretrainingConfig(
            run_id="bad-revision",
            base_model="Qwen/Qwen2.5-1.5B",
            base_revision="main",
            corpus_path="corpus.jsonl",
            corpus_audit_path="audit.json",
            holdout_path="holdout.json",
            policy_path="policy.json",
            output_dir="out",
        )
