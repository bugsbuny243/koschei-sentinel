# Sentinel Ephemeral Cloud Runtime

Koschei Sentinel does not require a Koschei-owned physical server.

The recommended runtime contract is provider-neutral, cost-aware, and assumes ephemeral rented GPU compute. The provider is not part of Sentinel identity or persistent state.

## Cost-aware storage boundary

Sentinel must not treat ClickHouse as universal storage.

The recommended v2 profile splits persistence into three tiers:

- **Google Drive cold archive** — required for low-cost persistent storage of dataset releases, manifests, SHA256 records, provenance, evaluation artifacts, selected checkpoints/adapters, and recovery material.
- **ClickHouse hot analytics** — optional. It is only for data that benefits from fast analytical queries, such as recent security events, compact telemetry, run metrics, and aggregated threat intelligence. Default retention is short.
- **Object storage** — optional during normal development and required only when an explicitly authorized large distributed run needs durable high-throughput checkpoint staging.

Git remains the source of truth for code, configuration, schema, and run-definition provenance.

Hugging Face or another external registry may be used as an optional bootstrap source. It is not required to operate Sentinel and must not define Sentinel model identity.

## Why Google Drive is not the live 397B checkpoint filesystem

Google Drive is a cold archive and recovery layer, not the synchronous checkpoint backend for a multi-node 397B run. Large distributed runs should write active checkpoint shards to local NVMe and temporary object storage, verify their manifests, and only then archive the release material that belongs in Drive.

This avoids turning Drive latency or mount behavior into a training failure mode while keeping normal development storage costs low.

## Normal development lifecycle

1. Rent ephemeral GPU compute only when needed.
2. Bind the existing Sentinel Drive archive using `KOSCHEI_DRIVE_ARCHIVE_FOLDER_ID`.
3. Leave ClickHouse unbound unless hot analytical queries are actually needed.
4. Leave object storage unbound unless a large distributed run is requested.
5. Run the v2 cloud-runtime preflight.
6. Keep paid compute denied unless explicit approval and a launch-session identity are present.
7. Write local batch telemetry and reproducibility metadata during execution.
8. Archive required artifacts, manifests, hashes, and recovery state to Drive before releasing ephemeral compute.

## Large-run lifecycle

A large run is a separate fail-closed mode.

In addition to the normal runtime approval, it requires:

- `KOSCHEI_LARGE_RUN_APPROVED=YES`;
- `KOSCHEI_OBJECT_STORE_URI` for checkpoint staging;
- the existing training-readiness, dataset, evaluation, checkpoint, and launch gates.

ClickHouse remains optional even for a large run; training must not depend on an analytics database being online.

## Environment bindings

Values must be injected by the runtime secret mechanism and must not be committed to Git.

- `KOSCHEI_DRIVE_ARCHIVE_FOLDER_ID` — persistent Sentinel Google Drive archive folder.
- `KOSCHEI_CLICKHOUSE_DSN` — optional hot-analytics binding.
- `KOSCHEI_OBJECT_STORE_URI` — optional normally; required for explicit large-run checkpoint staging.
- `KOSCHEI_CLOUD_RUNTIME_APPROVED=YES` — explicit paid-compute authorization.
- `KOSCHEI_CLOUD_RUNTIME_SESSION` — non-empty launch-session identity.
- `KOSCHEI_LARGE_RUN_APPROVED=YES` — additional authorization for a large distributed run.

The existing distributed Megatron launcher still requires normal rendezvous variables such as `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` for multi-node execution.

## Security and cost defaults

- public inbound access: DENY;
- provider binding: UNBOUND;
- paid compute: DENY;
- large run: DENY;
- ClickHouse: optional hot analytics only;
- ClickHouse retention: short by default;
- object storage: large-run-only by default;
- Google Drive: cold persistent archive;
- local NVMe/cache: disposable working state, never the only persistent copy;
- manifests and promoted artifacts: SHA256-bound before release.

## Preflight

Normal cost-aware runtime:

```bash
sentinel-cloud-runtime-preflight \
  --profile configs/runtime/ephemeral-cloud-gpu-cost-aware.v2.json
```

Explicit large-run gate:

```bash
sentinel-cloud-runtime-preflight \
  --profile configs/runtime/ephemeral-cloud-gpu-cost-aware.v2.json \
  --large-run
```

The v1 object-storage-plus-ClickHouse profile remains supported for compatibility, but v2 is the recommended low-cost operating model.

This runtime contract does not authorize a 397B training run. The Sentinel-native ~397B total / ~35B active architecture remains the target, and large paid runs remain separately gated.
