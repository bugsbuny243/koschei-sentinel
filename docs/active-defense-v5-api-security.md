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

A successful response sets:
- `Cache-Control: no-store`
- `X-Sentinel-Execution-Mode: simulation-only`
- `X-Sentinel-Evidence-Trust: caller-supplied`

The connector-envelope route also sets
`X-Sentinel-Production-Authorization: not-granted`.

**Important schema caveat:** The existing
`AssuredDefenseConnectorEnvelope.production_authorized: Literal[True]`
is preserved for compatibility with the internal connector contract. When it
appears in a dry-run HTTP response, it does **not** represent server-granted
authority, signed provenance, an operator approval, or permission to execute
anything in production. Clients must inspect `command.dry_run` and the
HTTP trust boundary; do not interpret that legacy field as a bearer token.

## Resource controls

For these HTTP endpoints only:

| Control | Limit |
| --- | ---: |
| HTTP request body (declared or streamed) | 8 MiB |
| Graph entities | 4,096 |
| Graph relations | 16,384 |
| Requested critical IDs | 512 |
| Requested focus IDs | 512 |
| Registry enrollments | 4,096 |
| Interception steps supplied to the envelope endpoint | 128 |
| Preconditions supplied to the envelope endpoint | 256 |

The 8 MiB HTTP body-size gate runs in ASGI before the JSON/Pydantic parser.
It rejects oversized declared `Content-Length` or streamed request bytes with
HTTP 413, and rejects malformed or duplicate `Content-Length` with HTTP 400.
All other table limits are Pydantic post-parse limits. The gate applies only
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
API request budgets, and both declared and streamed HTTP-body rejection.
The V5 workflow also runs the standard-dependency test
suite, excluding three PyTorch-specific MCore/checkpoint modules. That suite
does **not** replace a full optional-dependencies training validation pass.

The baseline repository-wide Ruff gate includes unrelated pre-existing lint
issues. The V5 workflow validates changed Python files separately; these
checks must not be represented as a clean repository-wide lint result.
