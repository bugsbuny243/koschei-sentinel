from __future__ import annotations

from pathlib import Path

from koschei_sentinel.gold_holdout_inference_runner import _load_inference_pack
from koschei_sentinel.gold_model_visible_context import (
    assert_gold_model_visible_context_is_answer_key_safe,
)


def preflight_gold_holdout_inference_pack(
    inference_pack_dir: str | Path,
) -> None:
    cases, _manifest, _manifest_raw = _load_inference_pack(inference_pack_dir)
    for case in cases:
        assert_gold_model_visible_context_is_answer_key_safe(case.input_context)
        if case.input_context.get("scenario_id") != case.scenario_id:
            raise ValueError(
                f"Gold HOLDOUT inference case scenario_id differs from visible context: {case.case_id}"
            )
