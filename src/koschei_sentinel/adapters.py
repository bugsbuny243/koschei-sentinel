from __future__ import annotations

import ipaddress
import json
import math
import os
import socket
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Protocol
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic import Field, model_validator

from koschei_sentinel.anonymize import detect_sensitive_text
from koschei_sentinel.benchmark import (
    BenchmarkCase,
    PredictionRecord,
    baseline_predictions,
    load_predictions,
)
from koschei_sentinel.models import SentinelOpinion, StrictModel

AdapterKind = Literal["baseline", "replay", "openai-compatible", "together"]
JsonMode = Literal["json-object", "prompt-only"]
_CANDIDATE_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_ENV_NAME = r"^[A-Z][A-Z0-9_]{0,127}$"
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_PROVIDER_DIAGNOSTIC_CHARS = 240
_TOGETHER_BASE_URL = "https://api.together.ai/v1"
_PROVIDER_USER_AGENT = (
    "Koschei-Sentinel/0.5 (+https://github.com/bugsbuny243/koschei-sentinel)"
)
_SYSTEM_PROMPT = (
    "You are Koschei Sentinel. Return exactly one JSON object matching "
    "sentinel.opinion.v1. The signed deterministic verdict is final. "
    "Do not change the case ID, verdict signature, grade, triggered rules, or authority. "
    "Every factual claim must cite supplied evidence IDs. Never invent evidence. "
    "When evidence is missing, emit a limitation and abstain from unsupported claims. "
    "Do not include markdown, personal data, credentials, or text outside the JSON object."
)


class AdapterError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class CostPolicy(StrictModel):
    pricing_as_of: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    input_usd_per_million_tokens: float = Field(ge=0.0, le=1000.0)
    output_usd_per_million_tokens: float = Field(ge=0.0, le=1000.0)
    max_estimated_cost_usd: float = Field(gt=0.0, le=100.0)
    max_requests: int = Field(default=8, ge=1, le=128)


class CandidateCostPlan(StrictModel):
    schema_version: Literal["sentinel.candidate-cost-plan.v1"] = (
        "sentinel.candidate-cost-plan.v1"
    )
    candidate_id: str
    adapter: AdapterKind
    model: str | None = None
    request_count: int
    max_requests: int | None = None
    estimated_input_tokens: int | None = None
    max_output_tokens: int | None = None
    estimated_max_cost_usd: float | None = None
    max_estimated_cost_usd: float | None = None
    within_request_limit: bool
    within_budget: bool


class RegistryCostPlan(StrictModel):
    schema_version: Literal["sentinel.registry-cost-plan.v1"] = (
        "sentinel.registry-cost-plan.v1"
    )
    total_candidates: int
    network_candidates: int
    all_within_limits: bool
    candidates: list[CandidateCostPlan]


class CandidateSpec(StrictModel):
    schema_version: Literal["sentinel.candidate.v1"] = "sentinel.candidate.v1"
    candidate_id: str = Field(pattern=_CANDIDATE_ID)
    adapter: AdapterKind
    model: str | None = Field(default=None, min_length=1, max_length=256)
    base_url: str | None = Field(default=None, min_length=1, max_length=2048)
    api_key_env: str | None = Field(default=None, pattern=_ENV_NAME)
    predictions_path: str | None = Field(default=None, min_length=1, max_length=1024)
    timeout_seconds: float = Field(default=60.0, ge=1.0, le=300.0)
    max_output_tokens: int = Field(default=2048, ge=128, le=8192)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    json_mode: JsonMode = "json-object"
    cost_policy: CostPolicy | None = None

    @model_validator(mode="after")
    def adapter_fields_are_consistent(self) -> CandidateSpec:
        if self.adapter == "baseline":
            if any((self.model, self.base_url, self.api_key_env, self.predictions_path)):
                raise ValueError("baseline candidates cannot configure provider fields")
            if self.cost_policy is not None:
                raise ValueError("baseline candidates cannot configure a cost policy")
        elif self.adapter == "replay":
            if self.predictions_path is None:
                raise ValueError("replay candidates require predictions_path")
            if any((self.model, self.base_url, self.api_key_env)):
                raise ValueError("replay candidates cannot configure provider fields")
            if self.cost_policy is not None:
                raise ValueError("replay candidates cannot configure a cost policy")
        elif self.adapter == "together":
            if self.model is None:
                raise ValueError("Together candidates require model")
            if self.base_url is not None:
                raise ValueError("Together candidates cannot override the provider endpoint")
            if self.api_key_env != "TOGETHER_API_KEY":
                raise ValueError("Together candidates must use TOGETHER_API_KEY")
            if self.predictions_path is not None:
                raise ValueError("Together candidates cannot configure predictions_path")
            if self.cost_policy is None:
                raise ValueError("Together candidates require a cost policy")
        else:
            if self.model is None or self.base_url is None:
                raise ValueError("openai-compatible candidates require model and base_url")
            if self.predictions_path is not None:
                raise ValueError("openai-compatible candidates cannot configure predictions_path")
        configured_text = (self.model, self.base_url, self.predictions_path)
        secret_kinds = {"jwt", "bearer_credential", "api_key", "long_hex_secret"}
        if any(
            value and secret_kinds.intersection(detect_sensitive_text(value))
            for value in configured_text
        ):
            raise ValueError("candidate configuration contains secret-like text")
        return self


class CandidateRegistry(StrictModel):
    schema_version: Literal["sentinel.candidate-registry.v1"] = (
        "sentinel.candidate-registry.v1"
    )
    candidates: list[CandidateSpec] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def candidate_ids_are_unique(self) -> CandidateRegistry:
        identifiers = [item.candidate_id for item in self.candidates]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("candidate_id values must be unique")
        return self


class ModelAdapter(Protocol):
    spec: CandidateSpec

    def predict(self, suite: Iterable[BenchmarkCase]) -> list[PredictionRecord]: ...


class BaselineAdapter:
    def __init__(self, spec: CandidateSpec) -> None:
        self.spec = spec

    def predict(self, suite: Iterable[BenchmarkCase]) -> list[PredictionRecord]:
        return baseline_predictions(suite, candidate=self.spec.candidate_id)


class ReplayAdapter:
    def __init__(self, spec: CandidateSpec, *, base_dir: Path) -> None:
        self.spec = spec
        self.base_dir = base_dir

    def predict(self, suite: Iterable[BenchmarkCase]) -> list[PredictionRecord]:
        del suite
        assert self.spec.predictions_path is not None
        path = _resolve_relative_path(self.base_dir, self.spec.predictions_path)
        predictions = load_predictions(path)
        candidates = {item.candidate for item in predictions}
        if candidates != {self.spec.candidate_id}:
            raise AdapterError(
                "candidate_mismatch",
                "replay predictions do not match the configured candidate_id",
            )
        return predictions


class OpenAICompatibleAdapter:
    def __init__(
        self,
        spec: CandidateSpec,
        *,
        allow_network: bool,
        allow_local_network: bool,
    ) -> None:
        self.spec = spec
        self.allow_network = allow_network
        self.allow_local_network = allow_local_network

    def predict(self, suite: Iterable[BenchmarkCase]) -> list[PredictionRecord]:
        if not self.allow_network:
            raise AdapterError(
                "network_disabled",
                "network adapters require the explicit --allow-network flag",
            )
        benchmarks = list(suite)
        _enforce_cost_plan(plan_candidate_cost(self.spec, benchmarks))
        endpoint = _validate_endpoint(
            self.spec.base_url or "",
            allow_local_network=self.allow_local_network,
        )
        api_key = _load_api_key(self.spec.api_key_env)
        outputs: list[PredictionRecord] = []
        for benchmark in benchmarks:
            opinion = self._predict_one(benchmark, endpoint=endpoint, api_key=api_key)
            outputs.append(
                PredictionRecord(
                    test_id=benchmark.test_id,
                    candidate=self.spec.candidate_id,
                    opinion=opinion,
                )
            )
        return outputs

    def _predict_one(
        self,
        benchmark: BenchmarkCase,
        *,
        endpoint: str,
        api_key: str | None,
    ) -> SentinelOpinion:
        _assert_endpoint_resolution_is_safe(
            endpoint,
            allow_local_network=self.allow_local_network,
        )
        payload = _request_payload(self.spec, benchmark)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": _PROVIDER_USER_AGENT,
        }
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        request = Request(
            endpoint,
            data=json.dumps(payload, separators=(",", ":")).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.spec.timeout_seconds) as response:
                body = response.read(_MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            diagnostic = _safe_provider_diagnostic(exc.read(8192))
            message = f"provider returned HTTP status {exc.code}"
            if diagnostic is not None:
                message = f"{message}: {diagnostic}"
            raise AdapterError("provider_http_error", message) from None
        except (TimeoutError, OSError):
            raise AdapterError(
                "provider_transport_error",
                "provider request failed before a valid response was received",
            ) from None
        if len(body) > _MAX_RESPONSE_BYTES:
            raise AdapterError(
                "response_too_large",
                "provider response exceeded the configured safety limit",
            )
        return parse_chat_completion(body)


def build_adapter(
    spec: CandidateSpec,
    *,
    base_dir: Path,
    allow_network: bool = False,
    allow_local_network: bool = False,
) -> ModelAdapter:
    if allow_local_network and not allow_network:
        raise ValueError("allow_local_network requires allow_network")
    if spec.adapter == "baseline":
        return BaselineAdapter(spec)
    if spec.adapter == "replay":
        return ReplayAdapter(spec, base_dir=base_dir)
    active_spec = (
        spec.model_copy(update={"base_url": _TOGETHER_BASE_URL})
        if spec.adapter == "together"
        else spec
    )
    return OpenAICompatibleAdapter(
        active_spec,
        allow_network=allow_network,
        allow_local_network=allow_local_network,
    )


def select_candidates(
    registry: CandidateRegistry, candidate_ids: Iterable[str] | None
) -> CandidateRegistry:
    requested = list(dict.fromkeys(candidate_ids or []))
    if not requested:
        return registry
    available = {item.candidate_id: item for item in registry.candidates}
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError("unknown candidate_id values: " + ", ".join(missing))
    return CandidateRegistry(candidates=[available[item] for item in requested])


def plan_registry_costs(
    registry: CandidateRegistry, suite: Iterable[BenchmarkCase]
) -> RegistryCostPlan:
    benchmarks = list(suite)
    plans = [
        plan_candidate_cost(spec, benchmarks)
        for spec in sorted(registry.candidates, key=lambda item: item.candidate_id)
    ]
    return RegistryCostPlan(
        total_candidates=len(plans),
        network_candidates=sum(
            item.adapter in {"openai-compatible", "together"} for item in plans
        ),
        all_within_limits=all(
            item.within_request_limit and item.within_budget for item in plans
        ),
        candidates=plans,
    )


def plan_candidate_cost(
    spec: CandidateSpec, suite: Iterable[BenchmarkCase]
) -> CandidateCostPlan:
    benchmarks = list(suite)
    policy = spec.cost_policy
    if policy is None:
        return CandidateCostPlan(
            candidate_id=spec.candidate_id,
            adapter=spec.adapter,
            model=spec.model,
            request_count=len(benchmarks),
            within_request_limit=True,
            within_budget=True,
        )

    input_tokens = sum(
        len(
            json.dumps(
                _request_payload(spec, benchmark)["messages"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for benchmark in benchmarks
    )
    output_tokens = len(benchmarks) * spec.max_output_tokens
    estimated = (
        input_tokens * policy.input_usd_per_million_tokens
        + output_tokens * policy.output_usd_per_million_tokens
    ) / 1_000_000
    estimated = math.ceil(estimated * 100_000_000) / 100_000_000
    return CandidateCostPlan(
        candidate_id=spec.candidate_id,
        adapter=spec.adapter,
        model=spec.model,
        request_count=len(benchmarks),
        max_requests=policy.max_requests,
        estimated_input_tokens=input_tokens,
        max_output_tokens=output_tokens,
        estimated_max_cost_usd=estimated,
        max_estimated_cost_usd=policy.max_estimated_cost_usd,
        within_request_limit=len(benchmarks) <= policy.max_requests,
        within_budget=estimated <= policy.max_estimated_cost_usd,
    )


def _enforce_cost_plan(plan: CandidateCostPlan) -> None:
    if not plan.within_request_limit:
        raise AdapterError(
            "request_limit_exceeded",
            "candidate request count exceeds the configured safety limit",
        )
    if not plan.within_budget:
        raise AdapterError(
            "budget_exceeded",
            "candidate estimated maximum cost exceeds the configured USD budget",
        )


def _request_payload(spec: CandidateSpec, benchmark: BenchmarkCase) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": spec.model,
        "temperature": spec.temperature,
        "max_tokens": spec.max_output_tokens,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "schema_version": "sentinel.inference-request.v1",
                        "test_id": benchmark.test_id,
                        "case": benchmark.case.model_dump(mode="json"),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        ],
    }
    if spec.json_mode == "json-object":
        payload["response_format"] = {"type": "json_object"}
    return payload


def load_candidate_registry(path: str | Path) -> CandidateRegistry:
    source = Path(path)
    try:
        return CandidateRegistry.model_validate_json(source.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError("invalid candidate registry") from exc


def parse_chat_completion(body: bytes | str) -> SentinelOpinion:
    try:
        envelope = json.loads(body)
        choices = envelope["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise TypeError
        message = choices[0]["message"]
        content = message["content"]
        if isinstance(content, list):
            parts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            content = "".join(parts)
        if not isinstance(content, str) or not content.strip():
            raise TypeError
        if content.lstrip().startswith("```"):
            raise TypeError
        return SentinelOpinion.model_validate_json(content)
    except (KeyError, TypeError, ValueError):
        raise AdapterError(
            "malformed_provider_response",
            "provider response did not contain one valid Sentinel opinion",
        ) from None


def _safe_provider_diagnostic(body: bytes) -> str | None:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    value: object | None = None
    error = payload.get("error")
    if isinstance(error, dict):
        value = error.get("message") or error.get("detail") or error.get("type")
    if value is None:
        value = payload.get("message") or payload.get("detail") or payload.get("title")
    if not isinstance(value, str):
        return None

    message = " ".join(value.split())[:_MAX_PROVIDER_DIAGNOSTIC_CHARS]
    if not message:
        return None
    if detect_sensitive_text(message):
        return "provider diagnostic redacted by Sentinel"
    return message


def _load_api_key(name: str | None) -> str | None:
    if name is None:
        return None
    value = os.getenv(name, "")
    if not value:
        raise AdapterError(
            "missing_api_key",
            "the configured API key environment variable is empty",
        )
    return value


def _resolve_relative_path(base_dir: Path, value: str) -> Path:
    candidate = (base_dir / value).resolve()
    root = base_dir.resolve()
    if candidate != root and root not in candidate.parents:
        raise AdapterError(
            "unsafe_replay_path",
            "replay paths must remain inside the registry directory",
        )
    return candidate


def _validate_endpoint(base_url: str, *, allow_local_network: bool) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"https", "http"}:
        raise AdapterError("unsafe_endpoint", "provider URL must use HTTP or HTTPS")
    if parsed.scheme != "https" and not allow_local_network:
        raise AdapterError("unsafe_endpoint", "provider URL must use HTTPS")
    if not parsed.hostname:
        raise AdapterError("unsafe_endpoint", "provider URL must include a hostname")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise AdapterError(
            "unsafe_endpoint",
            "provider URL cannot contain credentials, query parameters, or fragments",
        )
    if parsed.hostname.casefold() in {"localhost", "localhost.localdomain"}:
        if not allow_local_network:
            raise AdapterError("unsafe_endpoint", "local provider endpoints are disabled")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_global and not allow_local_network:
        raise AdapterError("unsafe_endpoint", "private provider endpoints are disabled")
    return base_url.rstrip("/") + "/chat/completions"


def _assert_endpoint_resolution_is_safe(
    endpoint: str,
    *,
    allow_local_network: bool,
) -> None:
    if allow_local_network:
        return
    parsed = urlparse(endpoint)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        records = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    except OSError:
        raise AdapterError(
            "provider_dns_error",
            "provider hostname could not be resolved",
        ) from None
    if not records:
        raise AdapterError("provider_dns_error", "provider hostname returned no addresses")
    for record in records:
        address = ipaddress.ip_address(record[4][0])
        if not address.is_global:
            raise AdapterError(
                "unsafe_endpoint",
                "provider hostname resolved to a non-public address",
            )
