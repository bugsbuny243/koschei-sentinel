import hashlib
import json

import pytest

from koschei_sentinel.gold_holdout_pack_preflight import (
    preflight_gold_holdout_inference_pack,
)
from koschei_sentinel.training import canonical_json
from tests.test_gold_holdout_inference_runner import _pack


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_pack_preflight_accepts_exported_answer_key_isolated_pack(tmp_path) -> None:
    pack = _pack(tmp_path)

    preflight_gold_holdout_inference_pack(pack)


def test_pack_preflight_rejects_nested_answer_key_leak_after_all_inner_hashes_rewritten(
    tmp_path,
) -> None:
    pack = _pack(tmp_path)
    inputs_path = pack / "inputs.jsonl"
    manifest_path = pack / "manifest.json"

    rows = [
        json.loads(line)
        for line in inputs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0]["input_context"]["graph_snapshots"][0]["truth"] = "MALICIOUS"
    rows[0]["input_context_sha256"] = _sha256_text(
        canonical_json(rows[0]["input_context"])
    )
    input_payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    inputs_path.write_text(input_payload, encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs_sha256"] = _sha256_text(input_payload)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden answer-key/review fields"):
        preflight_gold_holdout_inference_pack(pack)
