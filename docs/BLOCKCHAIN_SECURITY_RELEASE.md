# Blockchain Security Release v1

Status: offline training-data release gate  
Training authority: none  
Production authority: none

This layer turns one verified multi-source blockchain-security corpus into immutable `train`, `validation` and `test` splits without allowing one source snapshot or one incident/family lineage to cross split boundaries.

## Why a second split gate exists

Document-level random splitting is unsafe for security training. One protocol repository can contain hundreds of related files, and one incident family can appear in multiple independently ingested reports. If sibling files or the same family land in both training and evaluation, held-out scores become inflated.

The release builder therefore treats provenance as the split unit:

1. every document is mapped back to its immutable source snapshot;
2. source snapshots sharing any `family_ref` are joined into one connected component;
3. the complete component is assigned to exactly one split;
4. each split must independently cover every mandatory chain family and threat domain with the configured minimum number of independent sources;
5. the training split must retain the configured minimum source share.

The repository policy uses a fixed split seed. Changing the seed after inspecting benchmark results is a policy rotation, not an ordinary training parameter change.

## Inputs are re-proven

Release construction does not trust a previous green JSON file by itself. It reloads and recomputes:

- the source catalog and source-catalog policy;
- every immutable ingest manifest/corpus referenced by that catalog;
- the composed blockchain-security corpus;
- the held-out content/family fingerprints;
- the blockchain corpus policy and corpus audit;
- the release policy.

The stored source-catalog manifest and corpus audit must exactly equal the recomputed results and both must be ready. Corpus bytes, catalog lineage, holdout digest, benchmark suite digest and all policy digests are then bound into the release manifest.

## Command

```bash
sentinel-blockchain-release build \
  --catalog build/catalogs/blockchain-sources.json \
  --catalog-policy configs/pretraining/blockchain-source-catalog.v1.json \
  --source-catalog-manifest build/corpora/blockchain-security.source-manifest.json \
  --corpus build/corpora/blockchain-security.composed.jsonl \
  --holdout build/evals/blockchain-security-holdout.json \
  --blockchain-policy configs/pretraining/blockchain-security.v1.json \
  --corpus-audit build/audits/blockchain-security.json \
  --release-policy configs/pretraining/blockchain-release.v1.json \
  --output build/releases/blockchain-security-v1 \
  --root .
```

Verification is offline:

```bash
sentinel-blockchain-release verify build/releases/blockchain-security-v1
```

The materialized release contains only:

```text
train.jsonl
validation.jsonl
test.jsonl
blockchain-security-release-manifest.json
```

Publication uses atomic no-replace directory rename on supported POSIX filesystems. Existing releases are never overwritten.

## Boundary

Passing this gate means the release is reproducible, provenance-bound, source-isolated, family-isolated and linked to a passing held-out contamination audit. It does not mean a model trained on the release is production-safe. The next stage is an immutable training plan and offline executor, followed by independent held-out security benchmarks and promotion gates.
