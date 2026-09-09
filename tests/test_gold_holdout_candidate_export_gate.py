import pytest

from koschei_sentinel.gold_holdout_inference_runner import (
    GoldHoldoutGenerationPolicy,
    build_gold_holdout_inference_plan,
)
from tests.test_cyber_sft_export_verify import _build_export
from tests.test_gold_holdout_inference_runner import _pack


def test_gold_holdout_plan_rejects_nonpromotion_candidate_export(
    tmp_path,
    monkeypatch,
) -> None:
    pack = _pack(tmp_path)
    candidate_export = _build_export(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="promotion-eligible"):
        build_gold_holdout_inference_plan(
            inference_pack_dir=pack,
            candidate_export_dir=candidate_export,
            model_ref="koschei-sentinel:test",
            generation_policy=GoldHoldoutGenerationPolicy(),
        )
