from __future__ import annotations

from dataclasses import replace
from typing import Any

from koschei_sentinel.production_mcore_distributed_runtime import MCoreDistributedRuntime


def wrap_runtime_with_megatron_ddp(runtime: MCoreDistributedRuntime) -> MCoreDistributedRuntime:
    """Wrap the native MCore model in Megatron DDP for distributed-optimizer buffers.

    Distributed AdamW requires Megatron DDP's contiguous parameter/main-gradient buffers.
    The wrapper keeps communication synchronous for the one-step smoke path so correctness
    is verified before overlap features are introduced.
    """
    try:
        from megatron.core.distributed import DistributedDataParallel
        from megatron.core.distributed.distributed_data_parallel_config import (
            DistributedDataParallelConfig,
        )
    except ImportError as exc:  # pragma: no cover - production dependency path
        raise RuntimeError("Megatron-Core DDP is required for distributed optimizer smoke") from exc

    model = runtime.model
    config = getattr(model, "config", None)
    if config is None:
        raise RuntimeError("native MCore model does not expose TransformerConfig")

    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        use_distributed_optimizer=True,
    )
    ddp_model: Any = DistributedDataParallel(
        config=config,
        ddp_config=ddp_config,
        module=model,
        disable_bucketing=True,
    )
    ddp_model.broadcast_params()
    ddp_model.zero_grad_buffer()
    return replace(runtime, model=ddp_model)
