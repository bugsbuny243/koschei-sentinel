# Dataset release quality gate

A training release is created only from validated `sentinel.dataset.v1` rows.

## Leakage control

Every example carries a deterministic `group_ref`. The splitter hashes the versioned seed together with this group reference, so all examples from the same wallet cluster or incident lineage are assigned to exactly one of `train`, `validation`, or `test`.

The default allocation is 80/10/10 in basis points. The manifest records the seed and allocation. Changing either creates a different release and must be treated as a new benchmark lineage.

## Quality gates

Before any output directory is created, the release process rejects:

- duplicate `example_id` values;
- duplicate private `source_digest` values;
- identifiers that do not match Sentinel pseudonym formats;
- raw Solana-style addresses in text;
- email addresses, phone numbers, JWTs, bearer credentials, API keys, and long hexadecimal secrets;
- malformed dataset rows.

## Atomic release directory

The command writes all split files and the quality manifest into a staging directory, syncs them, and renames the complete directory into place. Existing release directories are never overwritten.

A release contains:

```text
train.jsonl
validation.jsonl
test.jsonl
quality-manifest.json
```

Tiny datasets are accepted for pipeline testing, but the manifest warns when a split is empty. Empty datasets are rejected.

## Command

```bash
sentinel-dataset-split \
  --input fixtures/dataset.safe.jsonl \
  --output-dir build/releases/sentinel-v0.2
```

Dry-run validation writes nothing:

```bash
sentinel-dataset-split \
  --input fixtures/dataset.safe.jsonl \
  --dry-run
```
