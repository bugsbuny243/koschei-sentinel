# Blockchain Security Source Ingest v1

Status: offline corpus-construction contract  
Network access: none  
Production authority: none

The multi-chain corpus policy is useful only if incoming material has reproducible provenance. `sentinel-blockchain-ingest` therefore consumes local, already acquired source snapshots and never downloads content by itself.

## Trust ceremony

1. Acquire a source snapshot outside the training process under a documented rights basis.
2. Place the snapshot under the repository/workspace root.
3. Run `prepare` to enumerate eligible UTF-8 source/text files and seal their path, byte length and SHA-256 into one snapshot digest.
4. Review the generated spec, especially source class, rights basis, chain families and threat domains.
5. Run `build`. If any eligible source byte changed after preparation, ingest fails closed.
6. Concatenate only reviewed canonical snapshot corpora into the candidate blockchain-security corpus, then run `sentinel-blockchain-corpus-audit`.

Example:

```bash
sentinel-blockchain-ingest prepare \
  --source-id protocol-example-v1 \
  --source-class PROTOCOL_SOURCE \
  --rights-basis APACHE_2_0 \
  --snapshot build/snapshots/protocol-example \
  --chain EVM \
  --threat SMART_CONTRACT \
  --threat PRIVILEGED_ACCESS \
  --output build/source-specs/protocol-example.json

sentinel-blockchain-ingest build \
  --spec build/source-specs/protocol-example.json \
  --corpus-output build/source-corpora/protocol-example.jsonl \
  --manifest-output build/source-corpora/protocol-example.manifest.json
```

## Security properties

- Source snapshot paths are confined to the configured root.
- Symbolic links are rejected.
- Hidden metadata trees such as `.git` are ignored.
- Only an explicit set of source/text extensions is eligible.
- Files are individually SHA-256 bound and bounded in size.
- The complete snapshot is bound by an ordered content-addressed digest.
- Snapshot drift between prepare and build is rejected.
- Duplicate file content inside one snapshot is rejected so a source cannot inflate corpus weight through copies.
- Existing output corpus/manifest files are never overwritten.
- Every emitted row still passes `BlockchainSecurityDocument`, including rights, pseudonym, privacy, chain and threat-domain validation.

The command intentionally does not decide whether a source is legally usable and does not scrape the web. The declared `rights_basis` remains a reviewed provenance assertion, and the later corpus audit remains mandatory before training.
