# Owner-Key Rotation Checkpoint v1

`sentinel-owner-rotation` creates cryptographic governance evidence for handing Sentinel owner authority from one Ed25519 key to another without silently activating the new key.

The current shadow baseline lineage is part of the signed handoff. A proposal binds:

- the exact `lineage_digest`;
- the current lineage head candidate;
- the sealed replay SHA-256;
- the current owner key fingerprint;
- the proposed next owner key fingerprint.

The current owner signs the proposal together with an explicit `approver_id`. The proposed next owner then counter-signs the proposal, the current-owner approval digest, and its own `accepter_id`. The final checkpoint re-verifies the existing baseline lineage with the current owner key and verifies both handoff signatures.

## Flow

```text
verified shadow baseline lineage
→ rotation proposal
→ current owner Ed25519 approval
→ next owner Ed25519 counter-signature
→ dual-signed rotation checkpoint
→ canonical one-successor rotation claim
```

Example:

```bash
sentinel-owner-rotation propose \
  --lineage build/shadow/baseline-lineage.json \
  --current-owner-public-key owner-current.pub.pem \
  --next-owner-public-key owner-next.pub.pem \
  --output build/governance/rotation.proposal.json

sentinel-owner-rotation approve-current \
  --proposal build/governance/rotation.proposal.json \
  --current-owner-private-key owner-current.private.pem \
  --approver-id owner@koschei \
  --output build/governance/rotation.current-approval.json

sentinel-owner-rotation accept-next \
  --proposal build/governance/rotation.proposal.json \
  --current-approval build/governance/rotation.current-approval.json \
  --current-owner-public-key owner-current.pub.pem \
  --next-owner-private-key owner-next.private.pem \
  --accepter-id successor@koschei \
  --output build/governance/rotation.next-acceptance.json

sentinel-owner-rotation checkpoint \
  --proposal build/governance/rotation.proposal.json \
  --current-approval build/governance/rotation.current-approval.json \
  --next-acceptance build/governance/rotation.next-acceptance.json \
  --lineage build/shadow/baseline-lineage.json \
  --current-owner-public-key owner-current.pub.pem \
  --next-owner-public-key owner-next.pub.pem \
  --claim-dir build/governance/rotation-claims \
  --output build/governance/rotation.checkpoint.json
```

Writers for the same governance lineage must share the same canonical `--claim-dir`. The claim key is derived from the current owner fingerprint plus the predecessor lineage digest. Retrying the exact same checkpoint is idempotent; trying to consume that checkpoint with a different next owner fails closed.

## Handoff is not activation

The checkpoint deliberately contains:

```text
state = dual_signed_not_activated
baseline_rekey_required = true
automatic_key_activation_allowed = false
automatic_promotion_allowed = false
automatic_deployment_allowed = false
production_deployment_allowed = false
web3_runtime_integration_allowed = false
verdict_mutation_allowed = false
```

A valid checkpoint therefore proves that both owner keys agreed to a specific handoff at a specific baseline checkpoint. It does not rewrite historical lineage signatures, replace the active owner key, promote a model, deploy anything, or authorize live Web3 integration.

A later, separate rekey operation must define how the historical single-owner lineage is preserved while authority moves to the next owner. That operation must consume this checkpoint rather than treating the new public key as self-authorizing.
