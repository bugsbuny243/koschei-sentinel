# Sentinel Drive Artifact Sync v1

Sentinel uses Google Drive as a low-cost cold archive. Drive is not treated as a live distributed-training filesystem and it is not trusted merely because an upload command returned success.

## Integrity model

The archive flow is:

1. enumerate the local artifact tree;
2. reject symbolic links and path escapes;
3. record every relative path, byte count, and SHA256;
4. compute a deterministic artifact digest;
5. upload the artifact into a content-addressed Drive path using immutable-copy semantics;
6. upload the archive manifest beside the data;
7. make rclone read the remote bytes and recompute SHA256 for every archived file and the manifest;
8. emit a transport receipt only after the remote checksum command succeeds.

Restore reverses the flow:

1. download into an empty local directory;
2. enumerate the restored tree;
3. recompute every SHA256 and the artifact digest;
4. fail closed if paths, sizes, hashes, source-manifest binding, or artifact digest differ.

## Remote layout

The Drive folder bound by `KOSCHEI_DRIVE_ARCHIVE_FOLDER_ID` is treated as the rclone root. Artifact data is content-addressed below it:

```text
sentinel-archive/
  <artifact-id>/
    <artifact-digest>/
      archive-manifest.json
      data/
        ...artifact files...
```

`rclone copy --immutable` is used instead of destructive synchronization. Existing mismatched archive files therefore fail rather than being silently replaced.

## Required runtime bindings

- `KOSCHEI_DRIVE_ARCHIVE_FOLDER_ID` — the Sentinel Drive archive root. The raw ID is never written into a transport receipt; the receipt contains only its SHA256 binding.
- `KOSCHEI_RCLONE_REMOTE` — the configured rclone remote name.
- `KOSCHEI_RCLONE_CONFIG` — optional path to an injected rclone config. Credentials or tokens must not be committed to Git.

The rclone Google Drive backend may also obtain its authentication from its normal secure runtime configuration. Sentinel does not own or persist those secrets.

## Commands

The CLI module supports four operations:

```bash
python -m koschei_sentinel.drive_archive_cli plan \
  --artifact-root build/checkpoint \
  --artifact-id run-001 \
  --kind CHECKPOINT \
  --source-manifest checkpoint-manifest.json \
  --output build/drive-archive-manifest.json

python -m koschei_sentinel.drive_archive_cli verify \
  --manifest build/drive-archive-manifest.json \
  --artifact-root build/checkpoint

python -m koschei_sentinel.drive_archive_cli push \
  --manifest build/drive-archive-manifest.json \
  --artifact-root build/checkpoint \
  --receipt-output build/drive-archive-receipt.json

python -m koschei_sentinel.drive_archive_cli pull \
  --manifest build/drive-archive-manifest.json \
  --destination build/restored-checkpoint
```

The `push` command requires rclone and the Drive runtime bindings. It never prints the Drive folder ID or credential material. A transport receipt is produced only after remote SHA256 read-back succeeds.

## Checkpoint integration

Existing Sentinel checkpoint lineage remains authoritative. A `checkpoint-manifest.json` produced and verified by the existing checkpoint pipeline should be included inside the artifact root and supplied as `--source-manifest`. Drive Artifact Sync then binds the cold archive to the SHA256 of that manifest without redefining checkpoint identity.

For a large 397B/35B distributed run, active checkpoint shards still belong on local NVMe and the separately gated object-storage staging layer. Google Drive receives verified release/recovery material after the high-throughput checkpoint phase; it is not used for synchronous expert/checkpoint traffic.

## Cost boundary

ClickHouse is not involved in archive transfer. Drive Artifact Sync therefore does not create ClickHouse storage, ingestion, or query cost. ClickHouse remains an optional short-retention hot analytics layer under the cost-aware runtime v2 policy.
