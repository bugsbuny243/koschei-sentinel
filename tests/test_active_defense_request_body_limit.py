"""Deterministic adversarial checks for the V5 ASGI request framing boundary."""

from __future__ import annotations

import asyncio
import json

import pytest

from koschei_sentinel.request_body_limit import ActiveDefenseRequestBodyLimit


def _request(
    headers: list[tuple[bytes, bytes]],
    messages: list[dict],
    *,
    path: str = "/v1/active-defense/plan",
    method: str = "POST",
    supply_content_type: bool = True,
    per_frame_delay: float = 0.0,
) -> tuple[list[dict], list[dict]]:
    forwarded: list[dict] = []
    sent: list[dict] = []
    remaining = iter(messages)

    async def downstream(scope, receive, send) -> None:
        forwarded.append(await receive())
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive() -> dict:
        if per_frame_delay:
            await asyncio.sleep(per_frame_delay)
        try:
            return next(remaining)
        except StopIteration as exc:
            raise AssertionError("unexpected additional request-body read") from exc

    async def send(message: dict) -> None:
        sent.append(message)

    if (
        supply_content_type
        and method == "POST"
        and path.startswith("/v1/active-defense/")
        and not any(key.lower() == b"content-type" for key, _ in headers)
    ):
        headers = [*headers, (b"content-type", b"application/json")]
    scope = {"type": "http", "path": path, "method": method, "headers": headers}
    asyncio.run(ActiveDefenseRequestBodyLimit(downstream)(scope, receive, send))
    return sent, forwarded


def _body_chunks(*chunks: bytes) -> list[dict]:
    return [
        {"type": "http.request", "body": item, "more_body": index < len(chunks) - 1}
        for index, item in enumerate(chunks)
    ]


@pytest.mark.parametrize(
    "value",
    [b"-1", b"+2", b" 2", b"2 ", b"nope", b""],
)
def test_rejects_non_decimal_content_length(value: bytes) -> None:
    sent, forwarded = _request([(b"content-length", value)], _body_chunks(b"{}"))
    assert sent[0]["status"] == 400
    assert forwarded == []


def test_rejects_duplicate_content_length_even_when_equal() -> None:
    sent, forwarded = _request(
        [(b"content-length", b"2"), (b"Content-Length", b"2")],
        _body_chunks(b"{}"),
    )
    assert sent[0]["status"] == 400
    assert forwarded == []


def test_rejects_transfer_encoding_and_content_length_conflict() -> None:
    sent, forwarded = _request(
        [(b"content-length", b"2"), (b"transfer-encoding", b"chunked")],
        _body_chunks(b"{}"),
    )
    assert sent[0]["status"] == 400
    assert forwarded == []


@pytest.mark.parametrize("declared", [b"1", b"3"])
def test_rejects_short_or_long_declared_length(declared: bytes) -> None:
    sent, forwarded = _request(
        [(b"content-length", declared)],
        _body_chunks(b"{}"),
    )
    assert sent[0]["status"] == 400
    assert forwarded == []


def test_exact_length_replays_identical_bytes() -> None:
    sent, forwarded = _request(
        [(b"content-length", b"4")],
        _body_chunks(b"{", b'"', b'"', b"}"),
    )
    assert sent[0]["status"] == 204
    assert len(forwarded) == 1
    assert forwarded[0]["body"] == b'{""}'


def test_chunk_count_limits_zero_length_frame_flood(monkeypatch) -> None:
    monkeypatch.setattr(
        "koschei_sentinel.request_body_limit._MAX_ACTIVE_DEFENSE_BODY_CHUNKS", 2
    )
    sent, forwarded = _request([], _body_chunks(b"", b"", b"{}"))
    assert sent[0]["status"] == 413
    assert forwarded == []


def test_byte_limit_rejects_stream_without_declared_length(monkeypatch) -> None:
    monkeypatch.setattr(
        "koschei_sentinel.request_body_limit._MAX_ACTIVE_DEFENSE_BODY_BYTES", 3
    )
    sent, forwarded = _request([], _body_chunks(b"ab", b"cd"))
    assert sent[0]["status"] == 413
    assert forwarded == []


def test_declared_overlimit_does_not_read_body(monkeypatch) -> None:
    monkeypatch.setattr(
        "koschei_sentinel.request_body_limit._MAX_ACTIVE_DEFENSE_BODY_BYTES", 3
    )
    sent, forwarded = _request([(b"content-length", b"4")], [])
    assert sent[0]["status"] == 413
    assert forwarded == []


def test_client_disconnect_does_not_run_planner() -> None:
    sent, forwarded = _request([], [{"type": "http.disconnect"}])
    assert sent == []
    assert forwarded == []


def test_legacy_routes_do_not_use_v5_body_budget(monkeypatch) -> None:
    monkeypatch.setattr(
        "koschei_sentinel.request_body_limit._MAX_ACTIVE_DEFENSE_BODY_BYTES", 1
    )
    sent, forwarded = _request([], _body_chunks(b"{}"), path="/v1/opinions")
    assert sent[0]["status"] == 204
    assert forwarded[0]["body"] == b"{}"


def test_extremely_long_decimal_content_length_returns_413_not_500() -> None:
    sent, forwarded = _request(
        [(b"content-length", b"9" * 5000)],
        [],
    )
    assert sent[0]["status"] == 413
    assert forwarded == []


def test_large_number_of_leading_zeroes_does_not_overflow() -> None:
    sent, forwarded = _request(
        [(b"content-length", b"0" * 5000 + b"2")],
        _body_chunks(b"{}"),
    )
    assert sent[0]["status"] == 204
    assert forwarded[0]["body"] == b"{}"


def test_excessive_json_nesting_is_rejected_before_downstream() -> None:
    sent, forwarded = _request([], _body_chunks(b"[" * 65 + b"0" + b"]" * 65))
    assert sent[0]["status"] == 413
    assert forwarded == []


def test_nesting_limit_accepts_exact_boundary(monkeypatch) -> None:
    monkeypatch.setattr(
        "koschei_sentinel.request_body_limit._MAX_ACTIVE_DEFENSE_JSON_DEPTH", 3
    )
    payload = b'{"a":[{"b":0}]}'
    sent, forwarded = _request([], _body_chunks(payload[:5], payload[5:]))
    assert sent[0]["status"] == 204
    assert forwarded[0]["body"] == payload


def test_large_json_array_is_bounded_by_structure_tokens(monkeypatch) -> None:
    monkeypatch.setattr(
        "koschei_sentinel.request_body_limit._MAX_ACTIVE_DEFENSE_JSON_STRUCTURE_TOKENS",
        4,
    )
    sent, forwarded = _request([], _body_chunks(b"[0,1,", b"2,3,4]"))
    assert sent[0]["status"] == 413
    assert forwarded == []


def test_json_syntax_in_quoted_value_is_not_counted_across_frames(monkeypatch) -> None:
    monkeypatch.setattr(
        "koschei_sentinel.request_body_limit._MAX_ACTIVE_DEFENSE_JSON_DEPTH", 2
    )
    monkeypatch.setattr(
        "koschei_sentinel.request_body_limit._MAX_ACTIVE_DEFENSE_JSON_STRUCTURE_TOKENS",
        2,
    )
    payload = json.dumps(
        {"text": 'quoted " [{{{{{ ,,,,, ::::: ]}}}}} characters'},
        separators=(",", ":"),
    ).encode()
    split = payload.index(b'\\"') + 1  # Separate an escape from its quoted byte
    sent, forwarded = _request([], _body_chunks(payload[:split], payload[split:]))
    assert sent[0]["status"] == 204
    assert forwarded[0]["body"] == payload


def test_malformed_json_remains_the_downstream_parser_responsibility() -> None:
    sent, forwarded = _request([], _body_chunks(b'{"a":'))
    assert sent[0]["status"] == 204
    assert forwarded[0]["body"] == b'{"a":'


def test_post_without_json_content_type_is_denied_without_body_reads() -> None:
    sent, forwarded = _request([], [], supply_content_type=False)
    assert sent[0]["status"] == 415
    assert forwarded == []
    assert (b"x-sentinel-production-authorization", b"not-granted") in sent[0]["headers"]


@pytest.mark.parametrize(
    "value",
    [
        b"text/plain",
        b"application/x-www-form-urlencoded",
        b"application/json, text/plain",
        b"application/json; charset=latin-1",
        b"application/json; charset=utf-8; charset=utf-8",
        b"application/+json",
    ],
)
def test_unsupported_media_types_are_denied_without_parser(value: bytes) -> None:
    sent, forwarded = _request([(b"content-type", value)], [])
    assert sent[0]["status"] == 415
    assert forwarded == []


@pytest.mark.parametrize(
    "value",
    [b"application/json", b"application/json; charset=UTF-8", b"application/vnd.sentinel+json"],
)
def test_json_media_types_are_accepted(value: bytes) -> None:
    sent, forwarded = _request([(b"content-type", value)], _body_chunks(b"{}"))
    assert sent[0]["status"] == 204
    assert forwarded[0]["body"] == b"{}"


def test_duplicate_content_type_headers_are_rejected() -> None:
    sent, forwarded = _request(
        [
            (b"content-type", b"application/json"),
            (b"Content-Type", b"application/json"),
        ],
        [],
    )
    assert sent[0]["status"] == 400
    assert forwarded == []


def test_header_count_is_limited_before_reading_body(monkeypatch) -> None:
    monkeypatch.setattr(
        "koschei_sentinel.request_body_limit._MAX_ACTIVE_DEFENSE_HEADER_COUNT", 2
    )
    headers = [(b"x-a", b"1"), (b"x-b", b"2")]
    sent, forwarded = _request(headers, [])
    assert sent[0]["status"] == 431
    assert forwarded == []


def test_header_byte_size_is_limited_before_reading_body(monkeypatch) -> None:
    monkeypatch.setattr(
        "koschei_sentinel.request_body_limit._MAX_ACTIVE_DEFENSE_HEADER_BYTES", 40
    )
    sent, forwarded = _request([(b"x-long", b"x" * 50)], [])
    assert sent[0]["status"] == 431
    assert forwarded == []


def test_exact_header_budget_is_accepted(monkeypatch) -> None:
    headers = [(b"content-type", b"application/json"), (b"x-ok", b"yes")]
    exact = sum(len(key) + len(value) for key, value in headers)
    monkeypatch.setattr(
        "koschei_sentinel.request_body_limit._MAX_ACTIVE_DEFENSE_HEADER_COUNT", 2
    )
    monkeypatch.setattr(
        "koschei_sentinel.request_body_limit._MAX_ACTIVE_DEFENSE_HEADER_BYTES", exact
    )
    sent, forwarded = _request(headers, _body_chunks(b"{}"))
    assert sent[0]["status"] == 204
    assert forwarded[0]["body"] == b"{}"


def test_non_post_request_needs_no_json_content_type() -> None:
    sent, forwarded = _request(
        [], _body_chunks(b""), method="GET", supply_content_type=False
    )
    assert sent[0]["status"] == 204
    assert forwarded[0]["body"] == b""


@pytest.mark.parametrize(
    "payload",
    [
        b'{"a":"\\xff"}',
        b'{"a":"\\xc0\\xaf"}',
        b'{"a":"\\xe2"}',
        b'{"a":"\\xff\\xfe"}',
    ],
)
def test_rejects_invalid_or_incomplete_utf8_json(payload: bytes) -> None:
    sent, forwarded = _request([], _body_chunks(payload))
    assert sent[0]["status"] == 400
    assert forwarded == []
    assert (b"x-sentinel-production-authorization", b"not-granted") in sent[0]["headers"]


def test_rejects_utf8_byte_order_mark() -> None:
    sent, forwarded = _request([], _body_chunks(b"\\xef\\xbb\\xbf{}"))
    assert sent[0]["status"] == 400
    assert forwarded == []


def test_rejects_utf16_like_raw_nul_json() -> None:
    sent, forwarded = _request([], _body_chunks(b"{\\x00}\\x00"))
    assert sent[0]["status"] == 400
    assert forwarded == []


def test_accepts_multibyte_utf8_split_between_frames() -> None:
    payload = '{"name": "€"}'.encode("utf-8")
    euro = payload.index(b"\\xe2")
    sent, forwarded = _request(
        [], _body_chunks(payload[:euro + 1], payload[euro + 1:euro + 2], payload[euro + 2:])
    )
    assert sent[0]["status"] == 204
    assert forwarded[0]["body"] == payload


def test_escaped_json_nul_is_allowed_as_text() -> None:
    payload = b'{"name":"\\\\u0000"}'
    sent, forwarded = _request([], _body_chunks(payload))
    assert sent[0]["status"] == 204
    assert forwarded[0]["body"] == payload


def test_body_timeout_rejects_stalled_receive_before_planner(monkeypatch) -> None:
    monkeypatch.setattr(
        "koschei_sentinel.request_body_limit._MAX_ACTIVE_DEFENSE_BODY_READ_SECONDS",
        0.01,
    )
    sent, forwarded = _request([], _body_chunks(b"{}"), per_frame_delay=0.05)
    assert sent[0]["status"] == 408
    assert forwarded == []
    assert (b"x-sentinel-execution-mode", b"simulation-only") in sent[0]["headers"]


def test_total_body_timeout_rejects_slow_trickle(monkeypatch) -> None:
    monkeypatch.setattr(
        "koschei_sentinel.request_body_limit._MAX_ACTIVE_DEFENSE_BODY_READ_SECONDS",
        0.025,
    )
    sent, forwarded = _request(
        [], _body_chunks(b"{", b'"', b'"', b":" , b"0", b"}"), per_frame_delay=0.01
    )
    assert sent[0]["status"] == 408
    assert forwarded == []


def test_non_post_requests_remain_utf8_agnostic() -> None:
    sent, forwarded = _request(
        [], _body_chunks(b"\\xff"), method="GET", supply_content_type=False
    )
    assert sent[0]["status"] == 204
    assert forwarded[0]["body"] == b"\\xff"
