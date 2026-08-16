# Blockchain Continued Pretraining v1

Status: offline research training path  
Production authority: none  
Signing/deployment authority: none

This stage consumes only a verified `sentinel.blockchain-security-release.v1` artifact and produces a lineage-bound LoRA/QLoRA research adapter. The release must already have passed source diversity, rights/privacy, held-out contamination, source isolation and family isolation gates.

## Training boundary

`sentinel-blockchain-train` has two explicit modes:

1. plan-only: validate the complete release lineage and write an immutable training plan;
2. `--execute`: require that exact existing plan and start the offline GPU run.

Planning binds:

- immutable base-model revision;
- release manifest bytes and semantic digest;
- complete source-corpus digest;
- held-out benchmark-suite digest;
- release-policy and holdout lineage;
- train, validation and test split digests;
- split document and independent-source counts;
- exact training-config digest.

Any output directory that already exists blocks planning/execution. Training artifacts are published no-replace.

## Held-out test isolation

The GPU executor loads only `train.jsonl` and `validation.jsonl`.

During execution-lineage verification it checks the held-out test digest and source/document counts from the already verified release manifest, but intentionally does not open, hash, parse or tokenize `test.jsonl`. The adapter manifest records:

```text
held_out_test_consumed: false
```

The test split is reserved for a separate evaluation process after training. This prevents a convenient training helper from accidentally turning the held-out set into validation or prompt material.

## Training representation

Continued-pretraining rows contain the security material plus bounded domain metadata:

- chain families;
- threat domains;
- source class;
- source text.

Training text deliberately omits provenance identifiers such as document refs, source snapshot digests and incident/family pseudonyms. Those identifiers remain governance metadata rather than facts the model should memorize.

Long documents are tokenized deterministically into fixed maximum-length causal-LM chunks. Train and validation use the same tokenizer/base revision. The trainer uses PEFT LoRA with 4/8-bit quantization, gradient checkpointing and explicit reproducibility seeds.

## Example

Create the immutable plan:

```bash
sentinel-blockchain-train \
  --config build/configs/blockchain-training.json \
  --plan build/plans/blockchain-training.json \
  --root .
```

Only after reviewing that plan, explicitly execute:

```bash
sentinel-blockchain-train \
  --config build/configs/blockchain-training.json \
  --plan build/plans/blockchain-training.json \
  --root . \
  --execute
```

A successful run emits an adapter directory plus `blockchain-adapter-manifest.json` with exact model/release/config lineage, held-out test binding, adapter-file list and adapter digest.

## What this stage does not prove

Low train/eval loss is not a security promotion signal. The resulting adapter remains `offline_blockchain_research_only` until it passes separate held-out security benchmarks, authority/privacy/grounding gates, adversarial evaluation and the owner-controlled promotion path.

This executor also does not fetch training sources, modify chain state, sign transactions, deploy contracts, move funds or automatically promote a model.
