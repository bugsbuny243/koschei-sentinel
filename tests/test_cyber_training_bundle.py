import pytest

from koschei_sentinel.causal_defense_corpus import CausalDefenseCorpusManifest
from koschei_sentinel.cyber_collection_batch import CollectionBatchSeal
from koschei_sentinel.cyber_training_bundle import build_cyber_training_bundle
from koschei_sentinel.defense_reflex_corpus import DefenseReflexCorpusManifest


def _seal(ready: bool = True) -> CollectionBatchSeal:
    return CollectionBatchSeal(
        batch_id="cyber-v3-batch-0001",
        ready_for_training_pipeline=ready,
        sources=4,
        artifacts=10,
        training_artifacts=10,
        rejected_artifacts=0,
        artifact_manifest_sha256="a" * 64,
        training_corpus_sha256="b" * 64,
        source_ids=["attack", "kubernetes", "rustsec", "yara"],
        violations=[] if ready else ["fixture violation"],
    )


def _reflex() -> DefenseReflexCorpusManifest:
    return DefenseReflexCorpusManifest(
        example_count=8,
        source_candidate_count=8,
        examples_sha256="c" * 64,
        correction_sha256s=["d" * 64],
        ready_for_training_pipeline=True,
    )


def _causal() -> CausalDefenseCorpusManifest:
    return CausalDefenseCorpusManifest(
        example_count=6,
        examples_sha256="f" * 64,
        episode_sha256s=["1" * 64],
        correction_sha256s=["2" * 64],
        ready_for_training_pipeline=True,
    )


def test_training_bundle_requires_a_ready_knowledge_seal() -> None:
    with pytest.raises(ValueError, match="not ready"):
        build_cyber_training_bundle(
            bundle_id="bundle:test",
            foundation_model_ref="foundation:test",
            foundation_model_revision="revision:test",
            knowledge_seal=_seal(ready=False),
            reflex_manifest=_reflex(),
            causal_manifest=_causal(),
            eval_holdout_sha256="e" * 64,
        )


def test_training_bundle_rejects_holdout_leakage_by_digest() -> None:
    with pytest.raises(ValueError, match="eval holdout"):
        build_cyber_training_bundle(
            bundle_id="bundle:test",
            foundation_model_ref="foundation:test",
            foundation_model_revision="revision:test",
            knowledge_seal=_seal(),
            reflex_manifest=_reflex(),
            causal_manifest=_causal(),
            eval_holdout_sha256="f" * 64,
        )


def test_training_bundle_binds_all_training_planes_and_holdout_provenance() -> None:
    bundle = build_cyber_training_bundle(
        bundle_id="bundle:test",
        foundation_model_ref="foundation:test",
        foundation_model_revision="revision:test",
        knowledge_seal=_seal(),
        reflex_manifest=_reflex(),
        causal_manifest=_causal(),
        eval_holdout_sha256="e" * 64,
    )

    assert bundle.ready_for_training is True
    assert bundle.knowledge_training_corpus_sha256 == "b" * 64
    assert bundle.defense_reflex_examples_sha256 == "c" * 64
    assert bundle.causal_defense_examples_sha256 == "f" * 64
    assert bundle.causal_defense_example_count == 6
    assert len(bundle.required_stages) == 4
