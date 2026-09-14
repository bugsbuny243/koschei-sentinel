from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_foundation_catalog import FoundationCatalogCursor
from koschei_sentinel.production_mcore_recovery import RecoveryLedger, RecoveryLedgerState
from koschei_sentinel.production_mcore_training_checkpoint import MCoreTrainingState
from koschei_sentinel.production_mcore_training_trends import (
    TrainingTrendState,
    TrainingTrendThresholds,
    TrainingTrendTracker,
)


class CatalogRecoveryResumeState(StrictModel):
    schema_version: Literal["sentinel.catalog-recovery-resume-state.v1"] = (
        "sentinel.catalog-recovery-resume-state.v1"
    )
    catalog_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    current_learning_rate: float = Field(gt=0.0)
    ledger: RecoveryLedgerState


@dataclass(frozen=True)
class CatalogResumeBundle:
    cursor: FoundationCatalogCursor | None
    ledger: RecoveryLedger
    trend_tracker: TrainingTrendTracker
    current_learning_rate: float


def build_catalog_recovery_resume_state(
    *,
    catalog_sha256: str,
    current_learning_rate: float,
    ledger: RecoveryLedger,
) -> CatalogRecoveryResumeState:
    return CatalogRecoveryResumeState(
        catalog_sha256=catalog_sha256,
        current_learning_rate=current_learning_rate,
        ledger=ledger.to_state(),
    )


def attach_catalog_resume_state(
    training_state: MCoreTrainingState,
    *,
    catalog_sha256: str,
    current_learning_rate: float,
    ledger: RecoveryLedger,
    trend_tracker: TrainingTrendTracker,
) -> MCoreTrainingState:
    return training_state.model_copy(
        update={
            "schema_version": "sentinel.mcore-training-state.v2",
            "recovery_state": build_catalog_recovery_resume_state(
                catalog_sha256=catalog_sha256,
                current_learning_rate=current_learning_rate,
                ledger=ledger,
            ).model_dump(mode="json"),
            "trend_state": trend_tracker.to_state().model_dump(mode="json"),
        }
    )


def restore_catalog_resume_bundle(
    training_state: MCoreTrainingState,
    *,
    expected_catalog_sha256: str,
    trend_thresholds: TrainingTrendThresholds,
) -> CatalogResumeBundle:
    cursor: FoundationCatalogCursor | None = None
    if training_state.data_state is not None:
        if training_state.data_state.get("schema_version") != "sentinel.foundation-catalog-cursor.v1":
            raise ValueError("training checkpoint contains incompatible catalog cursor")
        cursor = FoundationCatalogCursor.model_validate(training_state.data_state)
        if cursor.catalog_sha256 != expected_catalog_sha256:
            raise ValueError("training checkpoint catalog cursor digest mismatch")

    if training_state.recovery_state is None:
        ledger = RecoveryLedger.empty()
        current_learning_rate = training_state.learning_rate
    else:
        recovery = CatalogRecoveryResumeState.model_validate(training_state.recovery_state)
        if recovery.catalog_sha256 != expected_catalog_sha256:
            raise ValueError("training checkpoint recovery catalog digest mismatch")
        ledger = RecoveryLedger.from_state(recovery.ledger)
        current_learning_rate = recovery.current_learning_rate
        if abs(current_learning_rate - training_state.learning_rate) > max(1.0e-15, training_state.learning_rate * 1.0e-12):
            raise ValueError("training checkpoint learning-rate state mismatch")

    if training_state.trend_state is None:
        trend_tracker = TrainingTrendTracker(trend_thresholds)
    else:
        trend_state = TrainingTrendState.model_validate(training_state.trend_state)
        trend_tracker = TrainingTrendTracker.from_state(trend_thresholds, trend_state)

    return CatalogResumeBundle(
        cursor=cursor,
        ledger=ledger,
        trend_tracker=trend_tracker,
        current_learning_rate=current_learning_rate,
    )
