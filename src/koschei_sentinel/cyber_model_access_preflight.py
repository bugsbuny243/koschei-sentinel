from __future__ import annotations

from typing import Literal

from pydantic import Field

from koschei_sentinel.cyber_sft_training import CyberSFTConfig
from koschei_sentinel.models import StrictModel


class CyberModelAccessPreflight(StrictModel):
    schema_version: Literal["sentinel.cyber-model-access-preflight.v1"] = (
        "sentinel.cyber-model-access-preflight.v1"
    )
    base_model: str
    requested_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    resolved_revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")
    public_ungated: bool
    model_type: str | None
    causal_lm_class: str | None
    safetensors_files: int = Field(ge=0)
    tokenizer_files_present: bool
    ready: bool
    blockers: list[str]
    warnings: list[str]


def audit_model_access(config: CyberSFTConfig) -> CyberModelAccessPreflight:
    try:
        from huggingface_hub import model_info
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "model-access preflight requires huggingface_hub and transformers"
        ) from exc

    blockers: list[str] = []
    warnings: list[str] = []
    resolved_revision: str | None = None
    public_ungated = False
    model_type: str | None = None
    causal_lm_class: str | None = None
    safetensors_files = 0
    tokenizer_files_present = False

    try:
        info = model_info(config.base_model, revision=config.base_revision)
    except Exception as exc:  # Hub raises several HTTP/revision-specific exception classes.
        blockers.append(f"Hugging Face model/revision is not accessible: {exc}")
    else:
        resolved_revision = getattr(info, "sha", None)
        if resolved_revision != config.base_revision:
            blockers.append(
                "Hugging Face resolved revision differs from the pinned base revision"
            )
        gated = bool(getattr(info, "gated", False))
        private = bool(getattr(info, "private", False))
        public_ungated = not gated and not private
        if gated:
            blockers.append("base model is gated; anonymous Kaggle execution is not allowed")
        if private:
            blockers.append("base model is private; anonymous Kaggle execution is not allowed")

        sibling_names = {
            getattr(row, "rfilename", "")
            for row in (getattr(info, "siblings", None) or [])
        }
        safetensors_files = sum(name.endswith(".safetensors") for name in sibling_names)
        if safetensors_files == 0:
            blockers.append("base model revision exposes no safetensors weight files")
        tokenizer_files_present = any(
            name in sibling_names
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "tokenizer.model",
                "vocab.json",
            )
        )
        if not tokenizer_files_present:
            warnings.append(
                "model listing did not expose a common tokenizer filename; AutoTokenizer will be authoritative"
            )

    if not blockers:
        try:
            hf_config = AutoConfig.from_pretrained(
                config.base_model,
                revision=config.base_revision,
                trust_remote_code=False,
            )
            model_type = getattr(hf_config, "model_type", None)
            mapped_class = AutoModelForCausalLM._model_mapping[type(hf_config)]
            causal_lm_class = mapped_class.__name__
        except Exception as exc:
            blockers.append(f"Transformers causal-LM config mapping failed: {exc}")
        else:
            if model_type != "qwen3_5":
                blockers.append(
                    f"expected Qwen3.5 base config model_type=qwen3_5, got {model_type!r}"
                )
            if causal_lm_class != "Qwen3_5ForCausalLM":
                blockers.append(
                    "AutoModelForCausalLM does not resolve the pinned checkpoint to Qwen3_5ForCausalLM"
                )

    return CyberModelAccessPreflight(
        base_model=config.base_model,
        requested_revision=config.base_revision,
        resolved_revision=resolved_revision,
        public_ungated=public_ungated,
        model_type=model_type,
        causal_lm_class=causal_lm_class,
        safetensors_files=safetensors_files,
        tokenizer_files_present=tokenizer_files_present,
        ready=not blockers,
        blockers=blockers,
        warnings=warnings,
    )
