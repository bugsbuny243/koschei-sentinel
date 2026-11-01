"""Deterministic adversarial checks for the V5 ASGI request framing boundary."""

from __future__ import annotations

import asyncio

import pytest

from koschei_sentinel.request_body_limit import ActiveDefenseRequestBodyLimit


def _request(
    headers: list[tuple[bytes, bytes]],
    messages: list[dict],
    *,
    path: str = "/v1/active-defense/plan",
) -> tuple[list[dict], list[dict]]:
    forwarded: list[dict] = []
    sent: list[dict] = []
    remaining = iter(messages)

    async def downstream(scope, receive, send) -> None:
        forwarded.append(await receive())
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive() -> dict:
        try:
            return next(remaining)
        except StopIteration as exc:
            raise AssertionError("unexpected additional request-body read") from exc

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {"type": "http", "path": path, "headers": headers}
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
