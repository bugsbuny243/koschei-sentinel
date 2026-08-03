from __future__ import annotations

import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from koschei_sentinel.adapters import _request_payload, load_candidate_registry
from koschei_sentinel.anonymize import detect_sensitive_text
from koschei_sentinel.benchmark import load_benchmark_suite

_SUITE = "fixtures/evals/suite.safe.jsonl"
_REGISTRY = "fixtures/models/candidates.together.low-cost.json"
_CANDIDATE = "sentinel-together-gpt-oss-20b"
_ENDPOINT = "https://api.together.ai/v1/chat/completions"


def _extract_message(raw: bytes, api_key: str) -> str:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "provider returned a non-JSON error body"

    value: object = payload
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            value = error.get("message") or error.get("type") or error
        else:
            value = payload.get("message") or payload

    message = " ".join(str(value).replace(api_key, "[REDACTED]").split())[:300]
    if not message:
        return "provider returned an empty error message"
    if detect_sensitive_text(message):
        return "provider error message was redacted by Sentinel"
    return message


def main() -> int:
    api_key = os.getenv("TOGETHER_API_KEY", "")
    if not api_key:
        print(json.dumps({"status": "missing_secret"}))
        return 2

    suite = load_benchmark_suite(_SUITE)
    registry = load_candidate_registry(_REGISTRY)
    spec = next(item for item in registry.candidates if item.candidate_id == _CANDIDATE)
    payload = _request_payload(spec, suite[0])
    payload["max_tokens"] = 16
    request = Request(
        _ENDPOINT,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            body = response.read(8192)
            print(json.dumps({"status": response.status, "response_bytes": len(body)}))
    except HTTPError as exc:
        body = exc.read(8192)
        print(
            json.dumps(
                {
                    "status": exc.code,
                    "provider_message": _extract_message(body, api_key),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
