# Blockchain Security Source Catalog v1

Status: offline data-governance contract  
Training authority: none  
Network access: none

## Why this layer exists

A blockchain-security corpus can still be weak if one repository dominates it, if source labels are copied blindly to every file, or if benchmark/evaluation material leaks into training data. The catalog layer therefore sits after immutable acquisition and semantic document review.

The current pipeline is:

```text
reviewed source snapshot
        -> immutable acquisition + lineage
        -> source-level semantic review
        -> document-level retain/exclude + label review
        -> reviewed source catalog + approved document corpus
        -> source-composition gate
        -> canonical BlockchainSecurityDocument corpus
        -> later blockchain-security corpus / holdout audit
        -> immutable Stage 2 training plan
```

The catalog gate is deliberately not a corpus-scale gate. Large-corpus size, family diversity, holdout isolation and final training-readiness constraints belong to the later `blockchain-security.v1.json` audit.

## Trust tiers

Every catalog source assigns one reviewed trust tier:

- `PRIMARY_PROTOCOL`: canonical protocol/client/runtime source or official technical documentation;
- `OFFICIAL_INCIDENT`: first-party incident report or postmortem;
- `INDEPENDENT_AUDIT`: independent professional security review;
- `FORMAL_SPECIFICATION`: formal or normative protocol/security specification;
- `PUBLIC_RESEARCH`: useful public research not treated as primary authority;
- `KOSCHEI_CURATED`: Koschei-curated material that still requires rights, privacy and provenance gates.

Trust tier is a curation assertion, not proof that prose is correct. Model training never turns a source label into production authority.

## Repository policy

`configs/pretraining/blockchain-source-catalog.v1.json` is the reviewed-pilot composition policy. It requires at least:

- 20 independently pinned reviewed sources;
- 5 real source classes;
- every mandatory chain family represented by at least one reviewed source;
- every mandatory threat domain represented by at least one reviewed source;
- 60% high-trust source share;
- no more than 15% synthetic-source share;
- no single source contributing more than 25% of approved documents;
- unique source snapshot digests;
- zero duplicate approved document content across different source snapshots.

These thresholds answer a narrow question: is the reviewed pilot genuinely multi-source, multi-chain and multi-threat without one source swallowing the whole corpus? They do not claim production scale. The later blockchain-security corpus audit remains fail-closed for document volume, family concentration, holdout isolation and other Stage 2 requirements.

## Supported catalog modes

`sentinel-blockchain-compose` accepts two catalog schemas.

### Immutable-ingest catalog

`sentinel.blockchain-source-catalog.v1` references repository-local source manifests and source corpora produced by the immutable ingest layer:

```json
{
  "schema_version": "sentinel.blockchain-source-catalog.v1",
  "catalog_id": "multichain-security-example",
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

This mode requires document labels to match the immutable source manifest exactly.

### Reviewed-document catalog

`sentinel.reviewed-source-catalog.v1` is used after semantic document review. It points to one canonical approved-document JSONL and records, per real source, the reviewed document-class counts, reviewed chain coverage, reviewed threat coverage, trust tier, license evidence and retained-document-set digest.

This mode exists because a real source repository can legitimately contain mixed document classes. For example, one protocol repository may contain both protocol implementation material and formal specification material. The reviewed catalog verifies the approved document set instead of forcing every retained file to inherit one source-wide class.

Each approved row must be `sentinel.approved-blockchain-document.v1` and is fail-closed on:

- exact source ID, repository and commit binding;
- exact source snapshot digest;
- exact license binding;
- text SHA-256;
- non-empty reviewed chain/threat labels;
- canonical source/path ordering;
- per-source retained document count;
- per-source reviewed class counts;
- per-source reviewed chain/threat unions;
- retained-document-set digest.

Approved rows are converted deterministically into canonical `sentinel.blockchain-security-document.v1` records. Rights basis is derived only from the allowlisted SPDX evidence already present in the reviewed catalog. Document-level class overrides are preserved.

## Compose command

Audit only:

```bash
sentinel-blockchain-compose \
  --catalog build/catalogs/blockchain-sources.json \
  --policy configs/pretraining/blockchain-source-catalog.v1.json \
  --root .
```

The CLI auto-detects immutable-ingest versus reviewed-document catalog schema.

Audit and materialize the canonical corpus only when the selected catalog policy passes:

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

This layer proves reviewed byte lineage, source diversity, approved-document binding and declared metadata consistency. It does not prove that every security claim is true, that a declared license is legally sufficient in every jurisdiction, that the final large corpus passes its holdout/family audit, or that a trained model is safe. Those remain separate gates.
