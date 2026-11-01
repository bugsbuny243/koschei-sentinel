"""ASGI ingress body-size gate for unauthenticated simulation routes.

Check declared and actual streamed byte counts before Pydantic JSON parsing.
This is not a substitute for trusted reverse-proxy limits or rate limiting.
"""

from __future__ import annotations

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

_MAX_ACTIVE_DEFENSE_BODY_BYTES = 8 * 1024 * 1024


class _BodyLimitExceeded(Exception):
    """Internal control flow when the streamed request crosses its byte budget."""


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
    """Reject oversize active-defense requests before the JSON parser runs."""

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

        total_bytes = 0

        async def bounded_receive() -> dict:
            nonlocal total_bytes
            message = await receive()
            if message["type"] == "http.request":
                total_bytes += len(message.get("body", b""))
                if total_bytes > limit:
                    raise _BodyLimitExceeded
            return message

        try:
            await self.app(scope, bounded_receive, send)
        except _BodyLimitExceeded:
            await _error_response(413, "active-defense request body limit exceeded")(
                scope, receive, send
            )
