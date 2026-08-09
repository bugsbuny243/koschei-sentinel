# Koschei Language Foundation Stage

Koschei Sentinel's first new specialization is the Koschei programming language itself. This stage is intentionally separated from Web3 evidence training and from later teacher/instruction tuning.

## Input trust boundary

The only accepted v1 source is a verified `koschei.language-foundation-corpus.v1` artifact exported by `bugsbuny243/koschei-lang` from an exact Git commit.

The importer verifies:

- source repository identity;
- exact 40-character Koschei language commit;
- exporter schema and generator version;
- every document source SHA-256;
- every content-derived document ID;
- the family derived from the repository path;
- the complete canonical corpus SHA-256.

Self-declared hashes are not treated as authentication. Release creation additionally requires two trusted values obtained from the authoritative language export ceremony:

- `--expect-source-commit`;
- `--expect-source-corpus-sha256`.

A corpus whose commit or canonical digest differs from either trusted pin is rejected even if an altered artifact is internally re-hashed consistently.

## Leakage-safe release

Build a release with:

```bash
sentinel-language-foundation build \
  --corpus build/koschei-language-foundation.json \
  --output build/language-foundation-v1 \
  --expect-source-commit <exact-koschei-lang-commit> \
  --expect-source-corpus-sha256 <trusted-corpus-sha256>
```

The builder emits:

```text
train.jsonl
validation.jsonl
test.jsonl
language-foundation-manifest.json
```

Splitting is deterministic from the configured seed and happens at the `family` level, never per file. Multi-file examples under one example directory stay together, all top-level `.ks` examples stay together, and translated reference documents share one family. `README.md`, `README.tr.md`, and `README.en.md` normalize to the same README family. This prevents sibling modules or translated near-duplicates from crossing train/evaluation boundaries.

Each split is SHA-256 bound in the manifest. `sentinel-language-foundation verify <release>` re-hashes every split, rejects ambiguous duplicate JSON members, re-validates every document, reconstructs the original canonical source-corpus digest, replays the seed-derived family assignment, and fails if a family appears in more than one split.

Release publication never exposes a partially populated destination. All split files and the manifest are written, verified and fsynced in a hidden staging directory first. Publication then uses one atomic no-replace directory rename; if the platform or filesystem cannot provide that primitive, the operation fails closed rather than weakening the boundary. Any pre-existing destination causes publication to fail without replacing it.

## What this does not do yet

This change prepares authoritative, leakage-safe language-domain data. It does **not** start GPU training, does not change the production model, does not enable automatic training, and does not give Sentinel live Web3 or deployment authority.

The next stage is a dedicated continued-pretraining/QLoRA path that consumes this release as Koschei language material. Teacher behavior (explaining code, helping a developer inside Koschei Laboratory, debugging, and guiding learning) remains a later instruction-tuning stage built on top of the language foundation rather than mixed into the first corpus.
