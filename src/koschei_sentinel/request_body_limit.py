"""Bound active-defense request bodies before FastAPI starts JSON parsing.

Both declared and streamed inputs are checked; no more than the configured
byte ceiling is buffered. This does not replace reverse-proxy request limits.
"""

from __future__ import annotations

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

_MAX_ACTIVE_DEFENSE_BODY_BYTES = 8 * 1024 * 1024


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
    """Buffer at most one bounded request before passing it to the JSON parser."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope.get("path", "").startswith(
            "/v1/active-defense/"
        ):
            await self.app(scope, receive, send)
            return

        limit = _MAX_ACTIVE_DEFENSE_BODY_BYTES
        lengths = [
            value
            for key, value in scope.get("headers", [])
            if key.lower() == b"content-length"
        ]
        if len(lengths) > 1:
            await _error_response(400, "duplicate content-length headers")(scope, receive, send)
            return
        if lengths:
            try:
                declared_length = int(lengths[0])
            except ValueError:
                await _error_response(400, "invalid content-length header")(scope, receive, send)
                return
            if declared_length < 0:
                await _error_response(400, "invalid content-length header")(scope, receive, send)
                return
            if declared_length > limit:
                await _error_response(413, "active-defense request body limit exceeded")(
                    scope, receive, send
                )
                return

        # Read before handing control to FastAPI: its request parser converts
        # exceptions raised during receive() into a generic HTTP 400.
        buffered: list[dict] = []
        total_bytes = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                await _error_response(400, "invalid ASGI request body event")(
                    scope, receive, send
                )
                return
            total_bytes += len(message.get("body", b""))
            if total_bytes > limit:
                await _error_response(413, "active-defense request body limit exceeded")(
                    scope, receive, send
                )
                return
            buffered.append(message)
            if not message.get("more_body", False):
                break

        index = 0

        async def replay_receive() -> dict:
            nonlocal index
            if index < len(buffered):
                message = buffered[index]
                index += 1
                return message
            return await receive()

        await self.app(scope, replay_receive, send)
