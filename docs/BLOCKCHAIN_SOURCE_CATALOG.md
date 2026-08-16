# Blockchain Security Source Catalog v1

Status: offline data-governance contract  
Training authority: none  
Network access: none

## Why this layer exists

A large blockchain-security corpus can still be weak if one repository, one incident family or one synthetic generator dominates it. The snapshot-ingest layer proves what bytes entered one source. This catalog layer proves which independent sources were combined, how much each source contributes and whether chain/threat coverage comes from genuinely diverse snapshots.

The catalog therefore sits between immutable source ingest and the final 50k-document blockchain corpus audit.

```text
reviewed source snapshot
        -> immutable ingest manifest + corpus
        -> source catalog diversity gate
        -> composed canonical corpus
        -> blockchain-security 50k/2k-family corpus audit
        -> immutable Stage 2 training plan
```

## Trust tiers

Every catalog entry assigns one reviewed trust tier:

- `PRIMARY_PROTOCOL`: canonical protocol/client/runtime source or official technical documentation;
- `OFFICIAL_INCIDENT`: first-party incident report or postmortem;
- `INDEPENDENT_AUDIT`: independent professional security review;
- `FORMAL_SPECIFICATION`: formal or normative protocol/security specification;
- `PUBLIC_RESEARCH`: public research material that is useful but not treated as primary authority;
- `KOSCHEI_CURATED`: Koschei-curated material that still requires rights, privacy and provenance gates.

Trust tier is a curation assertion, not proof that prose is correct. Model training never turns a source label into production authority.

## Repository policy

`configs/pretraining/blockchain-source-catalog.v1.json` currently requires at least:

- 200 independent source snapshots;
- 5 source classes;
- 10 independent sources for every mandatory chain family;
- 10 independent sources for every mandatory threat domain;
- 60% high-trust source share;
- no more than 15% synthetic-source share;
- no single source contributing more than 5% of composed documents;
- unique snapshot digests;
- zero duplicate document content across different snapshots.

These source-level gates are intentionally separate from the later document/family-level policy in `blockchain-security.v1.json`. Both must pass.

## Catalog artifact

A `sentinel.blockchain-source-catalog.v1` file references only repository-local immutable ingest artifacts:

```json
{
  "schema_version": "sentinel.blockchain-source-catalog.v1",
  "catalog_id": "multichain-security-2026-08",
  "sources": [
    {
      "source_id": "example.protocol",
      "trust_tier": "PRIMARY_PROTOCOL",
      "manifest_path": "build/sources/example/manifest.json",
      "corpus_path": "build/sources/example/corpus.jsonl",
      "expected_snapshot_digest": "<64 hex>"
    }
  ]
}
```

Each source is revalidated against its immutable ingest manifest. The catalog verifies the snapshot digest pin, exact corpus file digest, document count, source class, rights basis, chain labels, threat labels, family lineage and canonical row order before aggregation.

## Compose command

Audit only:

```bash
sentinel-blockchain-compose \
  --catalog build/catalogs/blockchain-sources.json \
  --policy configs/pretraining/blockchain-source-catalog.v1.json \
  --root .
```

Audit and materialize a composed corpus only when every source-level gate passes:

```bash
sentinel-blockchain-compose \
  --catalog build/catalogs/blockchain-sources.json \
  --policy configs/pretraining/blockchain-source-catalog.v1.json \
  --root . \
  --corpus-output build/corpora/blockchain-security.composed.jsonl \
  --manifest-output build/corpora/blockchain-security.source-manifest.json
```

The command performs no network fetch and starts no model training. Output publication is no-replace. A failed catalog cannot materialize a training corpus.

## Security boundary

This layer proves byte lineage, source diversity and declared metadata consistency. It does not prove that every security claim is true, that a declared license is legally sufficient, or that a model trained on the corpus is safe. Those remain separate review, benchmark and promotion gates.
