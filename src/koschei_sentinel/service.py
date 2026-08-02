from __future__ import annotations

from fastapi import FastAPI

from . import __version__
from .models import CheckedOpinion, SecurityCase
from .policy import baseline_opinion, validate_opinion

app = FastAPI(
    title="Koschei Sentinel",
    version=__version__,
    description="Evidence-grounded commentary boundary for Koschei security cases.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "koschei-sentinel", "version": __version__}


@app.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready", "engine": "sentinel-baseline-v0.1"}


@app.post("/v1/opinions", response_model=CheckedOpinion)
def create_opinion(case: SecurityCase) -> CheckedOpinion:
    opinion = baseline_opinion(case)
    violations = validate_opinion(case, opinion)
    return CheckedOpinion(opinion=opinion, policy_violations=violations, accepted=not violations)
