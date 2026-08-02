# Secure model adapters and comparison matrix

Koschei Sentinel evaluates every model through the same versioned prediction and benchmark contracts. Provider-specific code is isolated behind adapters and cannot bypass the deterministic promotion gate.

## Candidate registry

A registry uses `sentinel.candidate-registry.v1` and contains one or more `sentinel.candidate.v1` entries. Candidate IDs must be unique and safe for artifact paths.

Supported adapters:

- `baseline`: runs the deterministic, non-generative Sentinel oracle.
- `replay`: loads an existing prediction JSONL file for reproducible offline evaluation.
- `openai-compatible`: calls a chat-completions endpoint and parses one strict `sentinel.opinion.v1` JSON object per benchmark case.

Literal API keys are forbidden. Network candidates reference an environment variable name through `api_key_env`; the secret value is read only at execution time and is never written to reports.

## Network safety

Network access is disabled by default. It requires `--allow-network`.

Without the separate `--allow-local-network` override, Sentinel rejects:

- HTTP endpoints that do not use TLS;
- loopback, private, link-local, reserved, multicast, or unspecified IP addresses;
- localhost endpoints;
- hostnames that resolve to a non-public address;
- URLs containing usernames, passwords, query parameters, or fragments.

Provider response bodies are bounded to 2 MiB. HTTP failures expose only the status code. Transport failures and malformed responses use controlled error codes and do not echo credentials or response bodies.

## Safe offline comparison

```bash
sentinel-compare \
  --suite fixtures/evals/suite.safe.jsonl \
  --registry fixtures/models/candidates.safe.json \
  --output-dir build/comparisons/safe \
  --require-all
```

The fixture compares the built-in baseline with a replay candidate. It needs no network access or secret.

## Real provider candidate

A provider entry contains no secret value:

```json
{
  "schema_version": "sentinel.candidate.v1",
  "candidate_id": "candidate-qwen-local",
  "adapter": "openai-compatible",
  "model": "organization/model-name",
  "base_url": "https://inference.example/v1",
  "api_key_env": "SENTINEL_MODEL_API_KEY",
  "json_mode": "json-object"
}
```

Run it only after setting the named environment variable and explicitly allowing network access.

## Artifacts

A comparison release is written atomically and existing directories are never overwritten. It contains:

```text
comparison-matrix.json
<candidate-id>/predictions.jsonl
<candidate-id>/benchmark-report.json
```

Prediction timestamps are omitted from stored JSONL so repeated offline runs are byte-identical. Ranking is based only on promotion status and benchmark quality scores; wall-clock timing cannot change the order.

Exit code `0` means at least one candidate passed, or every candidate passed when `--require-all` is used. Exit code `2` means the comparison input was rejected. Exit code `3` means no candidate passed the requested gate.
