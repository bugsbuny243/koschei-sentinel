from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.causal_defense_corpus import CausalDefenseCorpusManifest
from koschei_sentinel.cyber_collection_batch import CollectionBatchSeal
from koschei_sentinel.defense_reflex_corpus import DefenseReflexCorpusManifest
from koschei_sentinel.models import StrictModel


class CyberTrainingStage(StrEnum):
    KNOWLEDGE_CONTINUED_PRETRAINING = "KNOWLEDGE_CONTINUED_PRETRAINING"
    DEFENSE_REFLEX_SFT = "DEFENSE_REFLEX_SFT"
    ADVERSARIAL_REASONING = "ADVERSARIAL_REASONING"
    CYBER_RANGE_REGRESSION = "CYBER_RANGE_REGRESSION"


class CyberTrainingBundle(StrictModel):
    schema_version: Literal["sentinel.cyber-training-bundle.v2"] = (
        "sentinel.cyber-training-bundle.v2"
    )
    bundle_id: str = Field(min_length=3, max_length=256)
    foundation_model_ref: str = Field(min_length=3, max_length=512)
    foundation_model_revision: str = Field(min_length=3, max_length=256)
    knowledge_batch_id: str
    knowledge_training_corpus_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    knowledge_artifact_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    defense_reflex_examples_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    defense_reflex_example_count: int = Field(gt=0)
    causal_defense_examples_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    causal_defense_example_count: int = Field(gt=0)
    eval_holdout_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    required_stages: list[CyberTrainingStage]
    ready_for_training: bool
    bundle_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def holdout_is_separate(self) -> CyberTrainingBundle:
        training_hashes = {
            self.knowledge_training_corpus_sha256,
            self.knowledge_artifact_manifest_sha256,
            self.defense_reflex_examples_sha256,
            self.causal_defense_examples_sha256,
        }
        if self.eval_holdout_sha256 in training_hashes:
            raise ValueError("eval holdout digest must be distinct from every training-plane digest")
        required = [
            CyberTrainingStage.KNOWLEDGE_CONTINUED_PRETRAINING,
            CyberTrainingStage.DEFENSE_REFLEX_SFT,
            CyberTrainingStage.ADVERSARIAL_REASONING,
            CyberTrainingStage.CYBER_RANGE_REGRESSION,
        ]
        if self.required_stages != required:
            raise ValueError("cyber training stages must use the canonical ordered pipeline")
        return self


def _bundle_digest(
    *,
    bundle_id: str,
    foundation_model_ref: str,
    foundation_model_revision: str,
    knowledge_seal: CollectionBatchSeal,
    reflex_manifest: DefenseReflexCorpusManifest,
    causal_manifest: CausalDefenseCorpusManifest,
    eval_holdout_sha256: str,
) -> str:
    payload = "|".join(
        [
            bundle_id,
            foundation_model_ref,
            foundation_model_revision,
            knowledge_seal.batch_id,
            knowledge_seal.training_corpus_sha256,
            knowledge_seal.artifact_manifest_sha256,
            reflex_manifest.examples_sha256,
            str(reflex_manifest.example_count),
            causal_manifest.examples_sha256,
            str(causal_manifest.example_count),
            eval_holdout_sha256,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_cyber_training_bundle(
    *,
    bundle_id: str,
    foundation_model_ref: str,
    foundation_model_revision: str,
    knowledge_seal: CollectionBatchSeal,
    reflex_manifest: DefenseReflexCorpusManifest,
    causal_manifest: CausalDefenseCorpusManifest,
    eval_holdout_sha256: str,
) -> CyberTrainingBundle:
    if not knowledge_seal.ready_for_training_pipeline:
        raise ValueError("Cyber Corpus batch seal is not ready for the training pipeline")
    if knowledge_seal.violations:
        raise ValueError("Cyber Corpus batch seal contains violations")
    if knowledge_seal.training_artifacts <= 0:
        raise ValueError("Cyber Corpus batch contains no training-authorized artifacts")
    if not reflex_manifest.ready_for_training_pipeline or reflex_manifest.example_count <= 0:
        raise ValueError("Defense Reflex Corpus is not ready for the training pipeline")
    if not causal_manifest.ready_for_training_pipeline or causal_manifest.example_count <= 0:
        raise ValueError("Causal Defense Corpus is not ready for the training pipeline")

    stages = [
        CyberTrainingStage.KNOWLEDGE_CONTINUED_PRETRAINING,
        CyberTrainingStage.DEFENSE_REFLEX_SFT,
        CyberTrainingStage.ADVERSARIAL_REASONING,
        CyberTrainingStage.CYBER_RANGE_REGRESSION,
    ]
    digest = _bundle_digest(
        bundle_id=bundle_id,
        foundation_model_ref=foundation_model_ref,
        foundation_model_revision=foundation_model_revision,
        knowledge_seal=knowledge_seal,
        reflex_manifest=reflex_manifest,
        causal_manifest=causal_manifest,
        eval_holdout_sha256=eval_holdout_sha256,
    )
    return CyberTrainingBundle(
        bundle_id=bundle_id,
        foundation_model_ref=foundation_model_ref,
        foundation_model_revision=foundation_model_revision,
        knowledge_batch_id=knowledge_seal.batch_id,
        knowledge_training_corpus_sha256=knowledge_seal.training_corpus_sha256,
        knowledge_artifact_manifest_sha256=knowledge_seal.artifact_manifest_sha256,
        defense_reflex_examples_sha256=reflex_manifest.examples_sha256,
        defense_reflex_example_count=reflex_manifest.example_count,
        causal_defense_examples_sha256=causal_manifest.examples_sha256,
        causal_defense_example_count=causal_manifest.example_count,
        eval_holdout_sha256=eval_holdout_sha256,
        required_stages=stages,
        ready_for_training=True,
        bundle_sha256=digest,
    )
