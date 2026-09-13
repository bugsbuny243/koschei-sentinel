from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel

_DIGEST = r"^[a-f0-9]{64}$"
_ZERO = "0" * 64


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ProductionModelSourceIntake(StrictModel):
    schema_version: Literal["sentinel.production-model-source-intake.v1"] = (
        "sentinel.production-model-source-intake.v1"
    )
    model_ref: str = Field(min_length=1, max_length=512)
    model_revision: str = Field(pattern=_DIGEST)
    source_kind: Literal["huggingface", "git", "vendor-artifact", "internal-artifact"]
    source_ref: str = Field(min_length=1, max_length=2048)
    source_revision: str = Field(pattern=_DIGEST)
    source_payload_sha256: str = Field(pattern=_DIGEST)
    license_ref: str = Field(min_length=1, max_length=2048)
    architecture_evidence_sha256: str = Field(pattern=_DIGEST)
    router_evidence_sha256: str = Field(pattern=_DIGEST)
    expert_topology_evidence_sha256: str = Field(pattern=_DIGEST)
    tokenizer_template_evidence_sha256: str = Field(pattern=_DIGEST)
    framework_compatibility_evidence_sha256: str = Field(pattern=_DIGEST)
    independently_reviewed: Literal[True] = True
    review_ref: str = Field(min_length=1, max_length=2048)
    intake_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def source_is_pinned_and_non_placeholder(self) -> "ProductionModelSourceIntake":
        if self.model_revision == _ZERO or self.source_revision == _ZERO:
            raise ValueError("production model source revisions cannot be zero placeholders")
        for field_name in (
            "source_payload_sha256",
            "architecture_evidence_sha256",
            "router_evidence_sha256",
            "expert_topology_evidence_sha256",
            "tokenizer_template_evidence_sha256",
            "framework_compatibility_evidence_sha256",
        ):
            if getattr(self, field_name) == _ZERO:
                raise ValueError(f"{field_name} cannot be a zero placeholder")
        if self.model_ref.upper().startswith("UNWIRED"):
            raise ValueError("production model source cannot use an UNWIRED placeholder")
        return self


def _unsigned_payload(intake: ProductionModelSourceIntake) -> dict[str, object]:
    payload = intake.model_dump(mode="json")
    payload.pop("intake_sha256", None)
    return payload


def load_production_model_source_intake(path: str | Path) -> ProductionModelSourceIntake:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid production model source intake: {source}") from exc
    intake = ProductionModelSourceIntake.model_validate(payload)
    if intake.intake_sha256 != _canonical_digest(_unsigned_payload(intake)):
        raise ValueError("production model source intake self-hash does not verify")
    return intake
