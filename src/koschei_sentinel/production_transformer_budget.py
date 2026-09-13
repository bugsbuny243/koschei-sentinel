from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel

_TARGET_TOTAL_B = 397.0
_TARGET_ACTIVE_B = 35.0
_TOLERANCE_B = 0.5


class TransformerBudgetSpec(StrictModel):
    schema_version: Literal["sentinel.production-transformer-budget.v1"] = (
        "sentinel.production-transformer-budget.v1"
    )
    architecture_class: Literal["decoder-only-sparse-moe"] = "decoder-only-sparse-moe"
    num_layers: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    attention_kv_ratio: float = Field(gt=0.0, le=1.0)
    num_experts: int = Field(gt=1)
    experts_per_token: int = Field(gt=0)
    expert_intermediate_size: int = Field(gt=0)
    shared_intermediate_size: int = Field(ge=0)
    tied_embeddings: bool = True
    mlp_projection_count: Literal[3] = 3
    status: Literal["research_candidate"] = "research_candidate"

    @model_validator(mode="after")
    def validate_router(self) -> "TransformerBudgetSpec":
        if self.experts_per_token >= self.num_experts:
            raise ValueError("experts_per_token must be smaller than num_experts")
        return self


class TransformerBudgetResult(StrictModel):
    schema_version: Literal["sentinel.production-transformer-budget-result.v1"] = (
        "sentinel.production-transformer-budget-result.v1"
    )
    total_parameters_billion: float
    active_parameters_billion: float
    dense_shared_parameters_billion: float
    routed_expert_parameters_billion: float
    active_routed_expert_parameters_billion: float
    target_total_billion: Literal[397.0] = 397.0
    target_active_billion: Literal[35.0] = 35.0
    within_target_tolerance: bool
    blockers: list[str]
    status: Literal["research_candidate"] = "research_candidate"


def estimate_transformer_budget(spec: TransformerBudgetSpec) -> TransformerBudgetResult:
    """Estimate parameter budgets for a decoder-only sparse MoE transformer.

    The estimator intentionally models parameter arithmetic only. It is not evidence of
    numerical stability, throughput, convergence, checkpoint compatibility, or trainability.

    Attention approximation per layer:
      Q + O = 2 * H^2
      K + V = 2 * H^2 * attention_kv_ratio

    Routed/shared MLPs use SwiGLU-style three projections:
      gate + up + down = 3 * H * intermediate_size
    """
    h = spec.hidden_size
    layers = spec.num_layers

    embedding_parameters = spec.vocab_size * h
    if not spec.tied_embeddings:
        embedding_parameters *= 2

    attention_per_layer = (2.0 + 2.0 * spec.attention_kv_ratio) * h * h
    attention_parameters = layers * attention_per_layer

    shared_mlp_per_layer = spec.mlp_projection_count * h * spec.shared_intermediate_size
    shared_mlp_parameters = layers * shared_mlp_per_layer

    expert_per_layer = spec.mlp_projection_count * h * spec.expert_intermediate_size
    routed_expert_parameters = layers * spec.num_experts * expert_per_layer
    active_routed_expert_parameters = layers * spec.experts_per_token * expert_per_layer

    dense_shared_parameters = (
        embedding_parameters + attention_parameters + shared_mlp_parameters
    )
    total_parameters = dense_shared_parameters + routed_expert_parameters
    active_parameters = dense_shared_parameters + active_routed_expert_parameters

    total_b = total_parameters / 1e9
    active_b = active_parameters / 1e9
    dense_b = dense_shared_parameters / 1e9
    routed_b = routed_expert_parameters / 1e9
    active_routed_b = active_routed_expert_parameters / 1e9

    blockers: list[str] = []
    if abs(total_b - _TARGET_TOTAL_B) > _TOLERANCE_B:
        blockers.append("estimated total parameter count is outside 397B tolerance")
    if abs(active_b - _TARGET_ACTIVE_B) > _TOLERANCE_B:
        blockers.append("estimated active parameter count is outside 35B tolerance")

    return TransformerBudgetResult(
        total_parameters_billion=round(total_b, 6),
        active_parameters_billion=round(active_b, 6),
        dense_shared_parameters_billion=round(dense_b, 6),
        routed_expert_parameters_billion=round(routed_b, 6),
        active_routed_expert_parameters_billion=round(active_routed_b, 6),
        within_target_tolerance=not blockers,
        blockers=blockers,
    )
