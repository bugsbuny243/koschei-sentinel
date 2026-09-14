import pytest
from pydantic import ValidationError

from koschei_sentinel.production_foundation_data import FoundationSamplerState
from koschei_sentinel.production_mcore_training_checkpoint import MCoreTrainingState


def test_training_state_carries_foundation_sampler_cursor() -> None:
    sampler = FoundationSamplerState(
        global_sequence_offset=64,
        corpus_sha256="a" * 64,
        tokenizer_ref="tokenizer/ref",
        tokenizer_revision="rev",
        data_parallel_size=16,
    )
    state = MCoreTrainingState(
        global_step=2,
        consumed_microbatches=4,
        learning_rate=1e-5,
        global_seed=39735,
        data_state=sampler.model_dump(mode="json"),
    )
    restored = FoundationSamplerState.model_validate(state.data_state)
    assert restored.global_sequence_offset == 64
    assert restored.data_parallel_size == 16


def test_sampler_state_rejects_invalid_digest() -> None:
    with pytest.raises(ValidationError):
        FoundationSamplerState(
            global_sequence_offset=1,
            corpus_sha256="bad",
            tokenizer_ref="tokenizer/ref",
            data_parallel_size=16,
        )
