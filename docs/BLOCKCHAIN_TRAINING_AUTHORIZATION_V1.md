# Blockchain Training Authorization v1

Stage-2 blockchain continued pretraining must not begin merely because a model fits on a GPU. The official execution path now requires a separate owner-signed authorization lineage that binds the exact model, preflight evidence, candidate registry, verified blockchain release, and training configuration.

## Required lineage

A proposal is constructed only after all of the following re-verify successfully:

- the runtime seal is `sentinel.blockchain-runtime-preflight-seal.v1`, digest-valid, `PASSED`, and still marked `training_authorized=false`;
- every file named by the seal still matches its SHA-256 digest;
- hardware inventory, preflight plan, probe receipt, audit, and summary agree on the candidate and passing state;
- the current candidate registry has the same registry digest, candidate digest, model ID, and immutable revision used by the preflight;
- the blockchain release passes `verify_blockchain_security_release`, including train/validation/test byte digests and source/family isolation;
- the training config pins the same base model/revision, source-corpus digest, benchmark-suite digest, and release path.

The proposal also binds the GitHub base commit, normalized source tree, source-patch SHA-256, preflight policy digest, hardware inventory digest, plan digest, receipt digest, release manifest byte/semantic digests, holdout digest, all split digests, and the training config digest.

## Owner approval

The proposal state is `awaiting_owner_signature` and cannot authorize training. An Ed25519 owner key signs a domain-separated proposal digest. The resulting approval has `training_authorized=true` but retains:

- `training_started=false` at approval creation;
- `production_authority=false`;
- `web3_runtime_authority=false`;
- `automatic_training_allowed=false`.

Changing any bound model, registry, preflight artifact, release byte, training config, owner key, or authorization policy invalidates verification.

## Official execution gate

`sentinel-blockchain-train --execute` requires all authorization inputs:

- `--authorization-proposal`
- `--authorization-approval`
- `--authorization-policy`
- `--owner-public-key`
- `--preflight-seal`
- `--preflight-run`
- `--registry`

The signed bundle is fully re-verified before the existing training executor is called. Planning remains non-executing and leaves `training_authorized=false`.

The underlying Python training routine remains an internal low-level primitive; the supported command execution boundary is the authorization-gated CLI. No authorization artifact grants deployment, wallet, signing, RPC, or Web3 runtime authority.

## Current policy

`configs/training/blockchain-training-authorization.v1.json` currently admits only `qwen2.5-coder-7b-base` / `Qwen/Qwen2.5-Coder-7B` at exact revision `0396a76181e127dfc13e5c5ec48a8cee09938b02`. This policy does not assert that a qualifying blockchain corpus already exists and does not start training.
