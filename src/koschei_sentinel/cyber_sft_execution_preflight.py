from __future__ import annotations

from pathlib import Path

from koschei_sentinel.cyber_sft_training import (
    CyberSFTConfig,
    CyberSFTPlan,
    _assert_explicit_split_disjoint,
    _combined_promotion_eligibility,
    load_cyber_sft_examples,
    load_cyber_sft_validation_examples,
    split_cyber_sft_examples,
)
from koschei_sentinel.models import StrictModel


def resolve_planned_cyber_sft_corpora(
    config: CyberSFTConfig,
    plan: CyberSFTPlan,
    *,
    root: str | Path = ".",
) -> tuple[list[StrictModel], list[StrictModel], str, str, bool | None]:
    """Revalidate planned TRAIN/VALIDATION sources without model or CUDA access."""

    rows, examples_sha, manifest_sha, training_promotion = load_cyber_sft_examples(
        config,
        root=root,
    )
    if examples_sha != plan.corpus_examples_sha256:
        raise ValueError("Cyber SFT TRAIN corpus changed after plan creation")
    if manifest_sha != plan.corpus_manifest_sha256:
        raise ValueError("Cyber SFT TRAIN manifest changed after plan creation")

    explicit_validation = config.validation_corpus_dir is not None
    if plan.explicit_validation != explicit_validation:
        raise ValueError("Cyber SFT plan explicit-validation mode differs from config")

    promotion_eligible = training_promotion
    if explicit_validation:
        loaded = load_cyber_sft_validation_examples(config, root=root)
        if loaded is None:
            raise ValueError("explicit Cyber SFT VALIDATION corpus could not be loaded")
        validation_rows, validation_examples_sha, validation_manifest_sha, validation_promotion = (
            loaded
        )
        _assert_explicit_split_disjoint(rows, validation_rows)
        training_rows = rows
        promotion_eligible = _combined_promotion_eligibility(
            training_promotion,
            validation_promotion,
        )
        if validation_examples_sha != plan.validation_corpus_examples_sha256:
            raise ValueError("Cyber SFT VALIDATION corpus changed after plan creation")
        if validation_manifest_sha != plan.validation_corpus_manifest_sha256:
            raise ValueError("Cyber SFT VALIDATION manifest changed after plan creation")
    else:
        if (
            plan.validation_corpus_examples_sha256 is not None
            or plan.validation_corpus_manifest_sha256 is not None
        ):
            raise ValueError("Cyber SFT plan carries explicit VALIDATION hashes without a corpus")
        training_rows, validation_rows = split_cyber_sft_examples(
            rows,
            validation_ratio=config.validation_ratio,
            seed=config.seed,
        )

    if promotion_eligible != plan.corpus_promotion_eligible:
        raise ValueError("Cyber SFT corpus promotion eligibility changed after plan creation")
    if len(training_rows) != plan.training_examples:
        raise ValueError("Cyber SFT TRAIN example count differs from plan")
    if len(validation_rows) != plan.validation_examples:
        raise ValueError("Cyber SFT VALIDATION example count differs from plan")
    if len(training_rows) + len(validation_rows) != plan.example_count:
        raise ValueError("Cyber SFT total example count differs from plan")

    return (
        training_rows,
        validation_rows,
        examples_sha,
        manifest_sha,
        promotion_eligible,
    )
