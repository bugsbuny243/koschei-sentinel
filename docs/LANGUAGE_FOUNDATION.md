# Koschei Language Foundation Stage

Koschei Sentinel's first new specialization is the Koschei programming language itself. This stage is intentionally separated from Web3 evidence training and from later teacher/instruction tuning.

## Input trust boundary

The only accepted v1 source is a verified `koschei.language-foundation-corpus.v1` artifact exported by `bugsbuny243/koschei-lang`.

The importer verifies:

- source repository identity;
- exact 40-character Koschei language commit;
- exporter schema and generator version;
- every document source SHA-256;
- every content-derived document ID;
- the family derived from the repository path;
- the complete canonical corpus SHA-256.

A caller may additionally require `--expect-source-commit`; a corpus from any other commit is rejected.

## Leakage-safe release

Build a release with:

```bash
sentinel-language-foundation build \
  --corpus build/koschei-language-foundation.json \
  --output build/language-foundation-v1 \
  --expect-source-commit <exact-koschei-lang-commit>
```

The builder emits:

```text
train.jsonl
validation.jsonl
test.jsonl
language-foundation-manifest.json
```

Splitting is deterministic from the configured seed and happens at the `family` level, never per file. A multi-file Koschei program therefore cannot be divided across train and test.

Each split is SHA-256 bound in the manifest. `sentinel-language-foundation verify <release>` re-hashes every split, re-validates every document and fails if a family appears in more than one split.

The output directory is no-replace.

## What this does not do yet

This change prepares authoritative, leakage-safe language-domain data. It does **not** start GPU training, does not change the production model, does not enable automatic training, and does not give Sentinel live Web3 or deployment authority.

The next stage is a dedicated continued-pretraining/QLoRA path that consumes this release as Koschei language material. Teacher behavior (explaining code, helping a developer inside Koschei Laboratory, debugging, and guiding learning) remains a later instruction-tuning stage built on top of the language foundation rather than mixed into the first corpus.
