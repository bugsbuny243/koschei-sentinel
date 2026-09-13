from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel

_DIGEST = r"^[a-f0-9]{64}$"


class ProductionTargetBinding(StrictModel):
    schema_version: Literal["sentinel.production-target-binding.v1"] = (
        "sentinel.production-target-binding.v1"
    )
    total_parameters: Literal["397B"] = "397B"
    active_parameters: Literal["35B"] = "35B"
    architecture_class: Literal["sparse-moe"] = "sparse-moe"
    model_ref: str = Field(min_length=1)
    model_revision: str = Field(pattern=_DIGEST)
    architecture_manifest_sha256: str = Field(pattern=_DIGEST)
    router_manifest_sha256: str = Field(pattern=_DIGEST)
    expert_topology_manifest_sha256: str = Field(pattern=_DIGEST)
    trainer_adapter: Literal["megatron-swift"] = "megatron-swift"
    trainer_adapter_version: str = Field(min_length=1)
    production_training_allowed: bool = False
    verification_status: Literal["unverified", "verified"] = "unverified"
    blockers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def fail_closed_until_verified(self) -> "ProductionTargetBinding":
        if self.production_training_allowed and self.verification_status != "verified":
            raise ValueError("production training requires a verified target binding")
        if self.verification_status == "verified" and self.blockers:
            raise ValueError("verified target binding cannot retain blockers")
        if not self.production_training_allowed and not self.blockers:
            raise ValueError("blocked target binding must explain why training is disabled")
        return self


def load_production_target_binding(path: str | Path) -> ProductionTargetBinding:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid production target binding: {source}") from exc
    return ProductionTargetBinding.model_validate(payload)


def require_verified_production_target_binding(
    binding: ProductionTargetBinding,
) -> ProductionTargetBinding:
    if binding.verification_status != "verified":
        raise ValueError("397B/35B production target binding is not verified")
    if not binding.production_training_allowed:
        raise ValueError("397B/35B production training is not authorized by the binding")
    if binding.blockers:
        raise ValueError("397B/35B production target binding still has blockers")
    return binding
