from __future__ import annotations

import hmac
import os

from fastapi import Depends, FastAPI, HTTPException, Request, status

from . import __version__
from .models import CheckedOpinion, SecurityCase
from .policy import baseline_opinion, validate_opinion

app = FastAPI(
    title="Koschei Sentinel",
    version=__version__,
    description="Evidence-grounded commentary boundary for Koschei security cases.",
)


def _production() -> bool:
    return os.getenv("APP_ENV", "").strip().lower() == "production"


def _authorize_opinion_request(request: Request) -> None:
    """Protect customer/Fabric opinion inference without hiding health probes.

    Production fails closed when SENTINEL_API_TOKEN is absent. Local/test
    environments keep the historical no-token behavior unless a token is set,
    so existing offline development remains usable.
    """

    expected = os.getenv("SENTINEL_API_TOKEN", "").strip()
    if not expected:
        if _production():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="sentinel_api_token_not_configured",
            )
        return

    authorization = request.headers.get("authorization", "").strip()
    scheme, separator, supplied = authorization.partition(" ")
    if (
        separator != " "
        or scheme.lower() != "bearer"
        or not supplied
        or not hmac.compare_digest(supplied.strip(), expected)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="sentinel_unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "koschei-sentinel", "version": __version__}


@app.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready", "engine": "sentinel-baseline-v0.1"}


@app.post(
    "/v1/opinions",
    response_model=CheckedOpinion,
    dependencies=[Depends(_authorize_opinion_request)],
)
def create_opinion(case: SecurityCase) -> CheckedOpinion:
    opinion = baseline_opinion(case)
    violations = validate_opinion(case, opinion)
    return CheckedOpinion(opinion=opinion, policy_violations=violations, accepted=not violations)
