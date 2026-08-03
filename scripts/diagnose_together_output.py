from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from pydantic import ValidationError

from koschei_sentinel.adapters import (
    _PROVIDER_USER_AGENT,
    _request_payload,
    load_candidate_registry,
)
from koschei_sentinel.anonymize import detect_sensitive_text
from koschei_sentinel.benchmark import load_benchmark_suite
from koschei_sentinel.models import SentinelOpinion

_SUITE = Path("fixtures/evals/suite.safe.jsonl")
_REGISTRY = Path("fixtures/models/candidates.together.low-cost.json")
_CANDIDATE = "sentinel-together-gpt-oss-20b"
_ENDPOINT = "https://api.together.ai/v1/chat/completions"
_OUTPUT = Path("build/diagnostic/provider-output.json")


def _safe_content(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    if detect_sensitive_text(value):
        return "[REDACTED_BY_SENTINEL]"
    return value[:4000]


def main() -> int:
    api_key = os.getenv("TOGETHER_API_KEY", "")
    if not api_key:
        raise SystemExit("TOGETHER_API_KEY is required")

    suite = load_benchmark_suite(_SUITE)
    registry = load_candidate_registry(_REGISTRY)
    spec = next(item for item in registry.candidates if item.candidate_id == _CANDIDATE)
    payload = _request_payload(spec, suite[0])
    request = Request(
        _ENDPOINT,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": _PROVIDER_USER_AGENT,
        },
        method="POST",
    )
    with urlopen(request, timeout=90) as response:
        envelope = json.loads(response.read(2 * 1024 * 1024))

    choice = envelope["choices"][0]
    message = choice["message"]
    content = message.get("content")
    result: dict[str, object] = {
        "http_status": 200,
        "model": envelope.get("model"),
        "finish_reason": choice.get("finish_reason"),
        "content": _safe_content(content),
        "content_length": len(content) if isinstance(content, str) else None,
        "reasoning_length": len(message.get("reasoning") or ""),
        "usage": {
            "prompt_tokens": envelope.get("usage", {}).get("prompt_tokens"),
            "completion_tokens": envelope.get("usage", {}).get("completion_tokens"),
            "reasoning_tokens": envelope.get("usage", {}).get("reasoning_tokens"),
            "total_tokens": envelope.get("usage", {}).get("total_tokens"),
        },
    }

    if isinstance(content, str):
        try:
            decoded = json.loads(content)
            result["json_type"] = type(decoded).__name__
        except json.JSONDecodeError as exc:
            result["json_error"] = {
                "line": exc.lineno,
                "column": exc.colno,
                "message": exc.msg,
            }
        try:
            SentinelOpinion.model_validate_json(content)
            result["sentinel_contract"] = "valid"
        except ValidationError as exc:
            result["sentinel_contract"] = "invalid"
            result["validation_errors"] = exc.errors(
                include_input=False,
                include_url=False,
            )[:32]

    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "finish_reason": result["finish_reason"],
                "content_length": result["content_length"],
                "reasoning_length": result["reasoning_length"],
                "sentinel_contract": result.get("sentinel_contract"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
