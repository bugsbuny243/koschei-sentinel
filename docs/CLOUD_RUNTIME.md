# Sentinel Ephemeral Cloud Runtime

Koschei Sentinel does not require a Koschei-owned physical server.

The default runtime contract is provider-neutral and assumes ephemeral rented GPU compute. The provider is intentionally not part of Sentinel's identity or persistent state.

## Persistent state boundary

Ephemeral GPU nodes are disposable. They must not be the only copy of any checkpoint, dataset artifact, manifest, evaluation result, or security telemetry required for reproducibility.

Persistent state is split into:

- object storage for checkpoints, model artifacts, manifests, dataset artifacts, and immutable hashes;
- ClickHouse for security analytics, event/telemetry records, model/runtime observations, and large analytical queries;
- Git for code, configuration, schema, and run-definition provenance.

Hugging Face or another external registry may be used as an optional bootstrap source. It is not required to operate Sentinel and it must not define Sentinel model identity.

## Compute lifecycle

1. Select a GPU provider outside the Sentinel identity boundary.
2. Attach or expose the required persistent-store bindings to the runtime through secret environment injection.
3. Run the Sentinel cloud-runtime preflight.
4. Keep paid compute denied unless an explicit launch approval and launch session are present.
5. Start the existing distributed launcher using the provider's private/internal rendezvous network.
6. During execution, write reproducibility metadata and checkpoint manifests.
7. Before releasing ephemeral nodes, synchronize required checkpoints/artifacts to object storage and verify their SHA256-bound manifests.
8. Terminate rented GPU capacity when it is no longer needed.

## Required environment bindings

The profile refers only to environment-variable names; values must be injected by the runtime secret mechanism and must not be committed to Git.

- `KOSCHEI_OBJECT_STORE_URI` — persistent object storage binding.
- `KOSCHEI_CLICKHOUSE_DSN` — ClickHouse binding.
- `KOSCHEI_CLOUD_RUNTIME_APPROVED=YES` — explicit paid-compute authorization.
- `KOSCHEI_CLOUD_RUNTIME_SESSION` — non-empty launch-session identity.

The existing distributed Megatron launcher still requires its normal node/rendezvous variables such as `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` for multi-node execution.

## Security defaults

- public inbound access: DENY by default;
- cluster rendezvous: private/provider-internal only;
- local persistent state: forbidden as the sole source of truth;
- paid compute: DENY by default;
- checkpoint release: blocked until remote persistence and hash manifest requirements are satisfied;
- provider binding: UNBOUND, so RunPod, Lambda, another GPU cloud, or future infrastructure can be swapped without changing Sentinel identity.

## Preflight

```bash
sentinel-cloud-runtime-preflight --profile configs/runtime/ephemeral-cloud-gpu.v1.json
```

A missing ClickHouse/object-store binding or missing paid-run authorization returns a non-zero exit code. The preflight never prints secret values.

This runtime contract does not authorize a 397B training run. The Sentinel-native ~397B total / ~35B active architecture remains the target, and large paid runs remain separately gated by training readiness, dataset/evaluation requirements, and explicit launch approval.
