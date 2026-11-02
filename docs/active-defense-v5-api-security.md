# Sentinel V5 active-defense HTTP boundary

## Status

The V5 API routes are **offline, deterministic planning and simulation** endpoints.
They neither contact network routers, cloud controllers, HSMs, physical relays,
nor on-chain contracts. They **must not** be published as a production
authorization or command-execution service without an independently reviewed
trusted ingress and approval layer.

The existing `/health`, `/ready`, and `/v1/opinions` routes remain unchanged.

## Routes and trust

| Route | Inputs | Output | Effects |
| --- | --- | --- | --- |
| `POST /v1/active-defense/plan` | CyberStateGraph, optional critical/focus IDs | ActiveDefensePlan, InterceptionPlan, initial InterceptionExecution | Planning only |
| `POST /v1/active-defense/assured-plan` | CyberStateGraph, PerceptionGraphReceipt, PerceptionAssuranceSummary, PerceptionSourceRegistry, optional IDs/policy | AssuredActiveDefensePlan | Internal receipt/digest/domain consistency verification only |
| `POST /v1/active-defense/connector-envelope` | ProtectedScope, AssuredActiveDefensePlan, InterceptionPlan, step ID, precondition evidence, `dry_run` | AssuredDefenseConnectorEnvelope | `dry_run=true` only; `false` returns HTTP 403 |

A hash verifies that supplied content is internally consistent; it is **not**
proof of who submitted the data, which organization owns the protected scope,
or whether the telemetry came from an admitted sensor. The registry, graph,
receipts, and scope are provided by the caller and **not authenticated** at
this HTTP boundary. Even an apparently high-assurance plan must therefore be
treated as untrusted simulation output.

Every handled HTTP response under the V5 route prefix (including 400, 403,
404, 405, 413, 415, 422, 431, and successful responses) is marked with:
- `Cache-Control: no-store`
- `X-Sentinel-Execution-Mode: simulation-only`
- `X-Sentinel-Evidence-Trust: caller-supplied`
- `X-Sentinel-Production-Authorization: not-granted`

These headers do not establish provenance or authorization. The ingress
middleware applies them centrally, including to FastAPI validation responses.
Failures generated outside the ASGI application by an upstream proxy or
load balancer are outside this guarantee.

**Legacy schema compatibility and HTTP projection:** Internal
`AssuredDefenseConnectorEnvelope.production_authorized: Literal[True]`
remains unchanged for existing in-process connector contracts. The HTTP
connector-envelope endpoint returns a separate simulation-only projection:
`production_authorized: false`, and its `command.dry_run` must be `true`.
This HTTP response grants **no** execution permission, signed provenance, or
operator approval. The legacy internal field must never be treated as a
bearer token or server-issued production authorization.

## Request media types and headers

V5 `POST` endpoints require one unambiguous JSON `Content-Type` header:
`application/json` or a vendor media type ending in `+json`, optionally
with a single `charset=UTF-8` parameter. Missing, form-encoded, text,
comma-separated or non-UTF-8 media types return HTTP **415**. Duplicate
`Content-Type` headers return HTTP **400**.

The pre-parser ASGI boundary also rejects more than 128 request header
fields or more than 16 KiB in combined header-name/header-value bytes with
HTTP **431**, before reading the request body.
JSON request bodies are validated as strict UTF-8 across ASGI chunks, including
multibyte code points split between frames. Invalid or incomplete UTF-8,
UTF-8 byte-order marks, and raw NUL bytes (which could conceal UTF-16 JSON)
return HTTP **400**; escaped JSON text such as `\\u0000` is unaffected.
Non-V5 routes are not subject to this additional encoding gate. This is an application-level
backstop; the HTTP server/proxy must independently enforce limits on raw
header framing. The JSON media-type requirement applies only to V5 `POST`
requests; `GET` and legacy endpoints keep their existing handling.

## Resource controls

For these HTTP endpoints only:

| Control | Limit |
| --- | ---: |
| HTTP request body (declared or streamed) | 8 MiB |
| Total HTTP request body read time (V5 ingress only) | 15 seconds |
| HTTP body frame count (including empty frames) | 4,096 |
| HTTP request header count | 128 |
| HTTP request header name/value bytes combined | 16 KiB |
| JSON object/array nesting depth | 64 |
| JSON structural punctuation tokens (openings, commas and colons) | 262,144 |
| Graph entities | 4,096 |
| Graph relations | 16,384 |
| Requested critical IDs | 512 |
| Requested focus IDs | 512 |
| Registry enrollments | 4,096 |
| Interception steps supplied to the envelope endpoint | 128 |
| Preconditions supplied to the envelope endpoint | 256 |

The 8 MiB HTTP body-size gate runs in ASGI before the JSON/Pydantic parser.
It rejects oversized declared `Content-Length`, streamed request bytes, or
more than 4,096 ASGI body frames with HTTP 413; it returns HTTP 400 for
invalid or duplicate `Content-Length`, contradictory `Content-Length` and
`Transfer-Encoding`, and declared/actual length mismatches. The middleware
buffers at most 8 MiB into a single byte array and replays exactly the
validated bytes to the parser; a premature client disconnect does not run
the planner. Before JSON parsing it also scans each incoming chunk for excessive
nesting (over 64) or structural punctuation (over 262,144), returning HTTP
413. The scanner is aware of JSON strings and backslash-escaped quotes across
chunk boundaries; malformed JSON within those budgets remains FastAPI's
validation responsibility. The scanner is a **resource budget**, not a second
JSON parser, and cannot authenticate payload contents.
The V5 request-body receive loop also has a **15-second total wall-clock
read deadline**, independent of byte/frame limits. A stalled or slowly
trickling request is rejected with HTTP **408** before planning. Client
disconnects continue to terminate processing without running the planner.
This is not a comprehensive request/response deadline: reverse proxies and
servers still need connection, header, body and overall request timeouts.
All remaining table limits are Pydantic post-parse limits. The gate applies only
to `/v1/active-defense/` HTTP paths and does not modify legacy endpoints.
A production reverse proxy still needs its own body limits, authentication,
rate limiting, timeout limits, monitoring, and network isolation.

## Required production gates (not implemented)

1. An authenticated identity binding between the service caller and the
   tenant and protected scope (with server-side scope lookup, not a
   client-supplied grant).
2. Sensor source admission backed by trusted enrollment and cryptographic
   provenance with anti-replay, timestamp, and revocation checks.
3. Independently authorized policy decisions for high-impact containment,
   especially BGP route changes, protocol circuit breakers, HSM locks, and
   physical air-gap operations.
4. A separate execution plane with least-privilege connector credentials,
   scope-scoped idempotency, explicit human approval where required,
   bounded blast radius, and post-action verification.
5. Replay/audit trails and rollback or recovery procedures for each provider,
   including explicit handling of irreversible operations.
6. Deployment-level body-size gate, request rate limits, timeout ceilings,
   and network abuse/load tests. The HTTP service has an initial pre-parse
   8 MiB byte gate, but does not implement rate limits.

A connector command's action must be authorized **twice**: by the supplied
`ProtectedScope` and by the canonical `GUARD`/`COMBAT`/`SIEGE` ceiling
from `defense_authority.action_permitted_in_mode`. Caller-supplied
`DefenseDecision.permitted_actions` cannot elevate that ceiling. The
assured-envelope validator also checks the action/mode combination.

The V5 action enum and adapter profiles model **capabilities and evidence**,
not provider-specific execution implementations.

## Test status

`tests/test_active_defense_five_layers_and_api.py` covers entity mappings,
mode gates, deterministic adapter normalization, successful dry runs, rejection
of live HTTP connector commands, tampered graph receipts, trust headers, and
API request budgets, distinct simulation-only HTTP response projection, and
trust headers on successful and unsuccessful handled responses. The ASGI tests
also cover declared and streamed HTTP-body rejection.
The focused ASGI regression suite tests deep and wide JSON rejection, exact
boundary acceptance, quote/escape handling across streamed frames, duplicate
and missing JSON content types, invalid media types, excessive header counts
and bytes, and continued compatibility with legacy routes.
The V5 workflow also runs the standard-dependency test
suite, excluding three PyTorch-specific MCore/checkpoint modules. That suite
does **not** replace a full optional-dependencies training validation pass.

The main `ci` workflow runs a clean repository-wide `ruff check .`
and the standard-dependency suite. Its separate CPU-only PyTorch job runs the
three previously excluded checkpoint modules with NumPy installed. The
`Sentinel Production CPU Gates` workflow also exercises a real two-process
Gloo smoke in an isolated CI environment. These results do not constitute
live execution or production approval.
