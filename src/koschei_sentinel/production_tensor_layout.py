from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_megatron_model_spec import ProductionMegatronModelSpec


class TensorLayoutEntry(StrictModel):
    name: str = Field(min_length=1)
    category: Literal[
        "embedding", "attention", "normalization", "shared_mlp", "router", "expert"
    ]
    global_shape: tuple[int, ...]
    shard_shape: tuple[int, ...]
    pipeline_stage: int = Field(ge=0)
    tensor_parallel_shards: int = Field(gt=0)
    expert_parallel_shards: int = Field(gt=0)
    replicated_across_data_parallel: Literal[True] = True
    meta_device_only: Literal[True] = True


class TensorLayoutPlan(StrictModel):
    schema_version: Literal["sentinel.production-tensor-layout.v1"] = (
        "sentinel.production-tensor-layout.v1"
    )
    status: Literal["dry_run_only"] = "dry_run_only"
    entries: list[TensorLayoutEntry] = Field(min_length=1)
    tensor_count: int = Field(gt=0)
    execution_authorized: Literal[False] = False


def _split_first(shape: tuple[int, ...], shards: int) -> tuple[int, ...]:
    if shape[0] % shards:
        raise ValueError(f"first dimension {shape[0]} not divisible by {shards}")
    return (shape[0] // shards, *shape[1:])


def _split_last(shape: tuple[int, ...], shards: int) -> tuple[int, ...]:
    if shape[-1] % shards:
        raise ValueError(f"last dimension {shape[-1]} not divisible by {shards}")
    return (*shape[:-1], shape[-1] // shards)


def build_tensor_layout(spec: ProductionMegatronModelSpec) -> TensorLayoutPlan:
    t = spec.transformer
    tp = spec.topology.tensor_model_parallel_size
    pp = spec.topology.pipeline_model_parallel_size
    ep = spec.topology.expert_model_parallel_size
    layers_per_stage = spec.layers_per_pipeline_stage

    if t.num_experts % ep:
        raise ValueError("num_experts must be divisible by expert parallel size")
    if t.hidden_size % tp:
        raise ValueError("hidden_size must be divisible by tensor parallel size")
    if (spec.num_query_groups * spec.head_dim) % tp:
        raise ValueError("KV projection width must be divisible by tensor parallel size")
    if t.expert_intermediate_size % tp:
        raise ValueError("expert intermediate size must be divisible by tensor parallel size")
    if t.shared_intermediate_size % tp:
        raise ValueError("shared intermediate size must be divisible by tensor parallel size")

    entries: list[TensorLayoutEntry] = []

    # Vocabulary-parallel embedding on the first pipeline stage.
    embedding_shape = (t.vocab_size, t.hidden_size)
    entries.append(
        TensorLayoutEntry(
            name="embedding.word_embeddings.weight",
            category="embedding",
            global_shape=embedding_shape,
            shard_shape=_split_first(embedding_shape, tp),
            pipeline_stage=0,
            tensor_parallel_shards=tp,
            expert_parallel_shards=1,
        )
    )

    q_width = spec.num_attention_heads * spec.head_dim
    kv_width = spec.num_query_groups * spec.head_dim
    experts_per_ep_rank = t.num_experts // ep

    for layer in range(t.num_layers):
        stage = layer // layers_per_stage
        prefix = f"decoder.layers.{layer}"

        entries.extend(
            [
                TensorLayoutEntry(
                    name=f"{prefix}.input_layernorm.weight",
                    category="normalization",
                    global_shape=(t.hidden_size,),
                    shard_shape=(t.hidden_size,),
                    pipeline_stage=stage,
                    tensor_parallel_shards=1,
                    expert_parallel_shards=1,
                ),
                TensorLayoutEntry(
                    name=f"{prefix}.self_attention.q_proj.weight",
                    category="attention",
                    global_shape=(q_width, t.hidden_size),
                    shard_shape=_split_first((q_width, t.hidden_size), tp),
                    pipeline_stage=stage,
                    tensor_parallel_shards=tp,
                    expert_parallel_shards=1,
                ),
                TensorLayoutEntry(
                    name=f"{prefix}.self_attention.k_proj.weight",
                    category="attention",
                    global_shape=(kv_width, t.hidden_size),
                    shard_shape=_split_first((kv_width, t.hidden_size), tp),
                    pipeline_stage=stage,
                    tensor_parallel_shards=tp,
                    expert_parallel_shards=1,
                ),
                TensorLayoutEntry(
                    name=f"{prefix}.self_attention.v_proj.weight",
                    category="attention",
                    global_shape=(kv_width, t.hidden_size),
                    shard_shape=_split_first((kv_width, t.hidden_size), tp),
                    pipeline_stage=stage,
                    tensor_parallel_shards=tp,
                    expert_parallel_shards=1,
                ),
                TensorLayoutEntry(
                    name=f"{prefix}.self_attention.o_proj.weight",
                    category="attention",
                    global_shape=(t.hidden_size, q_width),
                    shard_shape=_split_last((t.hidden_size, q_width), tp),
                    pipeline_stage=stage,
                    tensor_parallel_shards=tp,
                    expert_parallel_shards=1,
                ),
                TensorLayoutEntry(
                    name=f"{prefix}.post_attention_layernorm.weight",
                    category="normalization",
                    global_shape=(t.hidden_size,),
                    shard_shape=(t.hidden_size,),
                    pipeline_stage=stage,
                    tensor_parallel_shards=1,
                    expert_parallel_shards=1,
                ),
                TensorLayoutEntry(
                    name=f"{prefix}.shared_mlp.gate_up.weight",
                    category="shared_mlp",
                    global_shape=(2 * t.shared_intermediate_size, t.hidden_size),
                    shard_shape=_split_first((2 * t.shared_intermediate_size, t.hidden_size), tp),
                    pipeline_stage=stage,
                    tensor_parallel_shards=tp,
                    expert_parallel_shards=1,
                ),
                TensorLayoutEntry(
                    name=f"{prefix}.shared_mlp.down.weight",
                    category="shared_mlp",
                    global_shape=(t.hidden_size, t.shared_intermediate_size),
                    shard_shape=_split_last((t.hidden_size, t.shared_intermediate_size), tp),
                    pipeline_stage=stage,
                    tensor_parallel_shards=tp,
                    expert_parallel_shards=1,
                ),
                TensorLayoutEntry(
                    name=f"{prefix}.moe.router.weight",
                    category="router",
                    global_shape=(t.num_experts, t.hidden_size),
                    shard_shape=(t.num_experts, t.hidden_size),
                    pipeline_stage=stage,
                    tensor_parallel_shards=1,
                    expert_parallel_shards=1,
                ),
                TensorLayoutEntry(
                    name=f"{prefix}.moe.experts.gate_up.weight",
                    category="expert",
                    global_shape=(t.num_experts, 2 * t.expert_intermediate_size, t.hidden_size),
                    shard_shape=(experts_per_ep_rank, (2 * t.expert_intermediate_size) // tp, t.hidden_size),
                    pipeline_stage=stage,
                    tensor_parallel_shards=tp,
                    expert_parallel_shards=ep,
                ),
                TensorLayoutEntry(
                    name=f"{prefix}.moe.experts.down.weight",
                    category="expert",
                    global_shape=(t.num_experts, t.hidden_size, t.expert_intermediate_size),
                    shard_shape=(experts_per_ep_rank, t.hidden_size, t.expert_intermediate_size // tp),
                    pipeline_stage=stage,
                    tensor_parallel_shards=tp,
                    expert_parallel_shards=ep,
                ),
            ]
        )

    entries.append(
        TensorLayoutEntry(
            name="decoder.final_layernorm.weight",
            category="normalization",
            global_shape=(t.hidden_size,),
            shard_shape=(t.hidden_size,),
            pipeline_stage=pp - 1,
            tensor_parallel_shards=1,
            expert_parallel_shards=1,
        )
    )

    return TensorLayoutPlan(entries=entries, tensor_count=len(entries))
