"""Bound active-defense request bodies before FastAPI starts JSON parsing.

Both declared and streamed inputs are checked with a fixed byte and message
budget. These safeguards do not replace trusted reverse-proxy ingress limits.
"""

from __future__ import annotations

import asyncio
import codecs

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

_MAX_ACTIVE_DEFENSE_BODY_BYTES = 8 * 1024 * 1024
_MAX_ACTIVE_DEFENSE_BODY_CHUNKS = 4096
_MAX_ACTIVE_DEFENSE_JSON_DEPTH = 64
_MAX_ACTIVE_DEFENSE_JSON_STRUCTURE_TOKENS = 262144
_MAX_ACTIVE_DEFENSE_HEADER_COUNT = 128
_MAX_ACTIVE_DEFENSE_HEADER_BYTES = 16 * 1024
_MAX_ACTIVE_DEFENSE_BODY_READ_SECONDS = 15


def _supported_json_content_type(raw: bytes) -> bool:
    """Accept UTF-8 JSON media types, never ambiguous lists or charsets."""
    parts = [part.strip().lower() for part in raw.split(b";")]
    media_type = parts[0]
    valid_media = media_type == b"application/json" or (
        media_type.startswith(b"application/")
        and media_type.endswith(b"+json")
        and len(media_type) > len(b"application/+json")
    )
    if not valid_media:
        return False
    return len(parts) == 1 or (
        len(parts) == 2 and parts[1] in (b"charset=utf-8", b'charset="utf-8"')
    )




class _JsonStructureBudget:
    """Scan JSON punctuation before parsing; preserve string/escape state across chunks.

    This is a resource budget, not a JSON validator. Malformed JSON is still
    handled by FastAPI, while deeply nested or extremely wide inputs fail early.
    """

    def __init__(self) -> None:
        self.depth = 0
        self.tokens = 0
        self.in_string = False
        self.escaped = False

    def check_chunk(self, chunk: bytes) -> str | None:
        for char in chunk:
            if self.in_string:
                if self.escaped:
                    self.escaped = False
                elif char == 92:  # Backslash
                    self.escaped = True
                elif char == 34:  # Double quote
                    self.in_string = False
                continue

            if char == 34:
                self.in_string = True
            elif char in (91, 123):  # [ or {
                self.depth += 1
                self.tokens += 1
                if self.depth > _MAX_ACTIVE_DEFENSE_JSON_DEPTH:
                    return "active-defense JSON nesting limit exceeded"
            elif char in (93, 125):  # ] or }
                if self.depth:
                    self.depth -= 1
            elif char in (44, 58):  # Comma or colon
                self.tokens += 1

            if self.tokens > _MAX_ACTIVE_DEFENSE_JSON_STRUCTURE_TOKENS:
                return "active-defense JSON structure limit exceeded"
        return None



# Set on *all* handled V5 HTTP responses, including schema errors and denials.
_SIMULATION_HEADERS = (
    (b"cache-control", b"no-store"),
    (b"x-sentinel-execution-mode", b"simulation-only"),
    (b"x-sentinel-evidence-trust", b"caller-supplied"),
    (b"x-sentinel-production-authorization", b"not-granted"),
)
_SIMULATION_HEADER_NAMES = frozenset(key for key, _ in _SIMULATION_HEADERS)


def _error_response(status: int, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"detail": detail},
        headers={
            "Cache-Control": "no-store",
            "X-Sentinel-Execution-Mode": "simulation-only",
        },
    )


class ActiveDefenseRequestBodyLimit:
    """Read one bounded request before passing it to the JSON parser."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope.get("path", "").startswith(
            "/v1/active-defense/"
        ):
            await self.app(scope, receive, send)
            return

        async def simulation_send(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() not in _SIMULATION_HEADER_NAMES
                ]
                message = {
                    **message,
                    "headers": [*headers, *_SIMULATION_HEADERS],
                }
            await send(message)

        limit = _MAX_ACTIVE_DEFENSE_BODY_BYTES
        headers = scope.get("headers", [])
        if len(headers) > _MAX_ACTIVE_DEFENSE_HEADER_COUNT or (
            sum(len(key) + len(value) for key, value in headers)
            > _MAX_ACTIVE_DEFENSE_HEADER_BYTES
        ):
            await _error_response(431, "active-defense request header limit exceeded")(
                scope, receive, simulation_send
            )
            return

        if scope.get("method") == "POST":
            content_types = [
                value for key, value in headers if key.lower() == b"content-type"
            ]
            if len(content_types) > 1:
                await _error_response(400, "duplicate content-type headers")(
                    scope, receive, simulation_send
                )
                return
            if not content_types or not _supported_json_content_type(content_types[0]):
                await _error_response(415, "active-defense requires UTF-8 JSON content-type")(
                    scope, receive, simulation_send
                )
                return

        lengths = [value for key, value in headers if key.lower() == b"content-length"]
        if len(lengths) > 1:
            await _error_response(400, "duplicate content-length headers")(scope, receive, simulation_send)
            return

        declared_length: int | None = None
        if lengths:
            # RFC 9110 Content-Length consists of ASCII decimal digits only.
            if not lengths[0] or not lengths[0].isdigit():
                await _error_response(400, "invalid content-length header")(
                    scope, receive, simulation_send
                )
                return
            if any(key.lower() == b"transfer-encoding" for key, _ in headers):
                await _error_response(400, "conflicting request framing headers")(
                    scope, receive, simulation_send
                )
                return

            # Compare decimal bytes before int(): Python caps conversions of
            # very long decimal strings, which must not become an HTTP 500.
            normalized = lengths[0].lstrip(b"0") or b"0"
            maximum = str(limit).encode("ascii")
            if len(normalized) > len(maximum) or (
                len(normalized) == len(maximum) and normalized > maximum
            ):
                await _error_response(413, "active-defense request body limit exceeded")(
                    scope, receive, simulation_send
                )
                return
            declared_length = int(normalized)

        # Buffer as a single bytearray: a malicious stream with thousands of
        # zero-length frames must not allocate a list of unlimited ASGI events.
        body = bytearray()
        structure_budget = _JsonStructureBudget()
        utf8_decoder = (
            codecs.getincrementaldecoder("utf-8")(errors="strict")
            if scope.get("method") == "POST"
            else None
        )
        chunks = 0
        try:
            async with asyncio.timeout(_MAX_ACTIVE_DEFENSE_BODY_READ_SECONDS):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    if message["type"] != "http.request":
                        await _error_response(400, "invalid ASGI request body event")(
                            scope, receive, simulation_send
                        )
                        return

                    chunks += 1
                    if chunks > _MAX_ACTIVE_DEFENSE_BODY_CHUNKS:
                        await _error_response(413, "active-defense request chunk limit exceeded")(
                            scope, receive, simulation_send
                        )
                        return

                    fragment = message.get("body", b"")
                    if len(fragment) > limit - len(body):
                        await _error_response(413, "active-defense request body limit exceeded")(
                            scope, receive, simulation_send
                        )
                        return
                    rejection = structure_budget.check_chunk(fragment)
                    if rejection is not None:
                        await _error_response(413, rejection)(
                            scope, receive, simulation_send
                        )
                        return
                    if utf8_decoder is not None:
                        if b"\x00" in fragment:
                            await _error_response(
                                400, "active-defense JSON body contains a raw NUL byte"
                            )(scope, receive, simulation_send)
                            return
                        try:
                            utf8_decoder.decode(fragment)
                        except UnicodeDecodeError:
                            await _error_response(400, "active-defense JSON body is not UTF-8")(
                                scope, receive, simulation_send
                            )
                            return
                    body.extend(fragment)
                    if not message.get("more_body", False):
                        break

        except TimeoutError:
            await _error_response(408, "active-defense request body read timeout")(
                scope, receive, simulation_send
            )
            return

        if utf8_decoder is not None:
            try:
                utf8_decoder.decode(b"", final=True)
            except UnicodeDecodeError:
                await _error_response(400, "active-defense JSON body is not UTF-8")(
                    scope, receive, simulation_send
                )
                return
            if body.startswith(b"\xef\xbb\xbf"):
                await _error_response(400, "active-defense JSON UTF-8 BOM is not permitted")(
                    scope, receive, simulation_send
                )
                return

        if declared_length is not None and declared_length != len(body):
            await _error_response(400, "content-length does not match request body")(
                scope, receive, simulation_send
            )
            return

        # FastAPI must see exactly the bytes already inspected. A single replay
        # also avoids re-allocating per-chunk messages in downstream processing.
        payload = bytes(body)
        del body
        replayed = False

        async def replay_receive() -> dict:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": payload, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, simulation_send)
