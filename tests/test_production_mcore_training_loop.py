from __future__ import annotations

import pytest

from koschei_sentinel.production_mcore_training_checkpoint import MCoreTrainingState
from koschei_sentinel.production_mcore_training_loop import (
    MCoreTrainingLoopConfig,
    MCoreTrainingLoopResult,
    MCoreTrainingStepResult,
    _learning_rate_for_step,
)


def _config(**overrides):
    payload = {
        "max_steps": 10,
        "microbatches_per_step": 4,
        "seq_length": 64,
        "learning_rate": 1.0e-5,
        "min_learning_rate": 1.0e-6,
        "warmup_steps": 2,
        "weight_decay": 0.1,
        "clip_grad": 1.0,
        "global_seed": 39735,
        "checkpoint_every_steps": 2,
    }
    payload.update(overrides)
    return MCoreTrainingLoopConfig.model_validate(payload)


def test_learning_rate_warmup_and_cosine_decay() -> None:
    config = _config()
    step0 = _learning_rate_for_step(config, 0)
    step1 = _learning_rate_for_step(config, 1)
    step2 = _learning_rate_for_step(config, 2)
    step9 = _learning_rate_for_step(config, 9)
    assert config.min_learning_rate <= step0 < step1 <= config.learning_rate
    assert step2 == pytest.approx(config.learning_rate)
    assert config.min_learning_rate <= step9 < step2


def test_invalid_training_schedule_fails_closed() -> None:
    with pytest.raises(ValueError, match="min_learning_rate"):
        _config(min_learning_rate=2.0e-5)
    with pytest.raises(ValueError, match="warmup_steps"):
        _config(warmup_steps=10)


def test_training_state_tracks_resume_metadata() -> None:
    state = MCoreTrainingState(
        global_step=7,
        consumed_microbatches=28,
        learning_rate=5.0e-6,
        global_seed=39735,
    )
    assert state.global_step == 7
    assert state.consumed_microbatches == 28


def test_training_loop_result_can_represent_resume() -> None:
    result = MCoreTrainingLoopResult(
        start_step=1,
        final_step=2,
        resumed_from_checkpoint=True,
        step_results=[
            MCoreTrainingStepResult(
                global_step=2,
                consumed_microbatches=8,
                learning_rate=5.0e-6,
                grad_norm=0.5,
            )
        ],
        final_checkpoint_dir="checkpoints/step-00000002",
    )
    assert result.resumed_from_checkpoint is True
    assert result.start_step == 1
    assert result.final_step == 2
    assert result.execution_authorized is False
