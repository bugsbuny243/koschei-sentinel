# Koschei Sentinel Model Target

## Product target

Koschei Sentinel's long-term product architecture target is one Sentinel-native sparse/MoE model at
approximately **397B total parameters / 35B active parameters**.

This is an architecture target, not a claim that the full model has already been trained.
Development must prove the architecture, tokenizer/data model, dataset pipeline, trainer, routing,
checkpoint system, security evaluation, and distributed execution before a large paid run is
considered.

The machine-readable contract is:

`configs/model/sentinel-moe-397b-a35b.target.json`

## External bootstrap boundary

`Qwen/Qwen3.5-397B-A17B` at revision
`8472618112abcbd45acbcdc58436aff4233c23f7` remains a pinned **external bootstrap and
trainer-compatibility proof target**.

It is not Koschei Sentinel, and its 17B activated-parameter topology does not replace the native
approximately 35B-active target.

Existing Qwen Megatron-SWIFT runbooks and fail-closed paid-compute gates remain useful for proving
large-MoE dataset rendering, topology validation, checkpoint/resume identity, and distributed
training plumbing. They must not be treated as the product identity contract.

Where older reactivation/runbook wording calls Qwen the "only long-term target", this document and
the machine-readable target contract supersede that wording for **model identity and architecture
vision**. Existing Qwen execution safety gates remain in force until deliberately replaced by a
verified native-target trainer contract.

## Non-negotiable promotion boundary

A Sentinel candidate must preserve:

- immutable checkpoint identity and SHA256 provenance;
- dataset version and dataset SHA256;
- tokenizer identity/version;
- Git commit, configuration, seed, environment, dependency and GPU topology records;
- training metrics and security-specific evaluation evidence;
- Gold/HOLDOUT isolation and regression gates;
- explicit evidence, confidence, uncertainty, source, and assumptions where applicable; and
- fail-closed handling of unknown or untrusted training material.

External teachers, evaluators, synthetic-data helpers, and bootstrap models may assist development,
but they never become the identity of Koschei Sentinel merely by being used in the pipeline.
