from __future__ import annotations

import pytest
from pydantic import ValidationError

from koschei_sentinel.production_mcore_distributed_runtime import MCoreDistributedRuntime
from koschei_sentinel.production_mcore_optimizer_smoke import (
    MCoreOptimizerSmokeResult,
    build_mcore_distributed_optimizer,
)


D0 = "0" * 64
D1 = "1" * 64


def _result_payload() -> dict:
    return {
        "global_rank": 0,
        "world_size": 1024,
        "learning_rate": 1.0e-5,
        "weight_decay": 0.1,
        "clip_grad": 1.0,
        "grad_norm": 0.5,
        "parameter_name": "module.decoder.layers.0.weight",
        "before_digest": D0,
        "after_digest": D1,
    }


def test_optimizer_smoke_result_accepts_step_without_checkpoint() -> None:
    result = MCoreOptimizerSmokeResult.model_validate(_result_payload())
    assert result.optimizer_step_verified is True
    assert result.checkpoint_roundtrip_verified is False
    assert result.execution_authorized is False


def test_optimizer_smoke_result_requires_changed_digest() -> None:
    payload = _result_payload()
    payload["after_digest"] = payload["before_digest"]
    with pytest.raises(ValidationError, match="changed parameter digest"):
        MCoreOptimizerSmokeResult.model_validate(payload)


def test_verified_roundtrip_requires_matching_restored_digest() -> None:
    payload = _result_payload()
    payload.update(
        checkpoint_roundtrip_verified=True,
        checkpoint_dir="artifacts/smoke",
        restored_digest=D0,
    )
    with pytest.raises(ValidationError, match="restored digest"):
        MCoreOptimizerSmokeResult.model_validate(payload)


def test_verified_roundtrip_accepts_post_step_digest() -> None:
    payload = _result_payload()
    payload.update(
        checkpoint_roundtrip_verified=True,
        checkpoint_dir="artifacts/smoke",
        restored_digest=D1,
    )
    result = MCoreOptimizerSmokeResult.model_validate(payload)
    assert result.checkpoint_roundtrip_verified is True


def test_optimizer_argument_validation_happens_before_runtime_imports() -> None:
    runtime = MCoreDistributedRuntime(
        local_rank=0,
        global_rank=0,
        world_size=1024,
        data_parallel_size=16,
        model=object(),
    )
    with pytest.raises(ValueError, match="learning_rate"):
        build_mcore_distributed_optimizer(runtime, learning_rate=0.0)
    with pytest.raises(ValueError, match="weight_decay"):
        build_mcore_distributed_optimizer(runtime, weight_decay=-1.0)
    with pytest.raises(ValueError, match="clip_grad"):
        build_mcore_distributed_optimizer(runtime, clip_grad=0.0)
