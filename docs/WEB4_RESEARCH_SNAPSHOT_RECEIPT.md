# Web4 Research Snapshot Receipt

`sentinel-web4-research-snapshot` creates byte-integrity receipts for operator-supplied local Web4 research files.

This artifact is intentionally non-authorizing. It does **not** prove that the supplied file is authentic source content, does not perform network fetching, does not resolve licensing, does not materialize a training corpus, and does not grant evaluation, training, promotion, or production authority.

A v1 receipt binds:

- Web4 source registry identity and registry SHA256;
- canonical locator and observed revision-status metadata;
- operator-supplied local snapshot basename, byte size, and SHA256;
- explicit capture timestamp;
- `source_match_verified=false`;
- `provenance_review_status=REVIEW_REQUIRED`;
- `registry_license_status=REVIEW_REQUIRED`;
- closed training/evaluation/promotion/production authority;
- a self-hash over the complete unsigned receipt.

The `verify` command rebuilds the receipt from the current registry and supplied local bytes and rejects source-registry drift, receipt tampering, or byte changes.

Production source authenticity and rights decisions require separate reviewed artifacts. A research snapshot receipt must never be treated as proof that source content is authentic, licensed for training, or safe to admit into a model dataset.
