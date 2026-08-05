# Signed Shadow Promotion v1

Koschei Sentinel promotion remains an offline research workflow. Owner approval can move a finalized incubation candidate only into the `shadow_research_candidate` stage; it cannot deploy or connect the model to Koschei Web3 Hub.

## 1. Create an owner-signature proposal

```bash
sentinel-promotion propose \
  --finalization build/finalizations/sentinel-v1/candidate-finalization.json \
  --owner-public-key keys/owner-ed25519-public.pem \
  --output build/promotions/sentinel-v1.proposal.json
```

The proposal binds:

- finalized incubation digest;
- adapter digest;
- benchmark suite and report digests;
- finalized registry digest;
- SHA-256 fingerprint of the owner's Ed25519 public key.

The proposal is permanently limited to:

```text
requested_stage = shadow_research_candidate
authority = explanation_only
automatic_promotion_allowed = false
automatic_deployment_allowed = false
production_deployment_allowed = false
web3_runtime_integration_allowed = false
```

## 2. Sign explicitly with the owner key

```bash
sentinel-promotion approve \
  --proposal build/promotions/sentinel-v1.proposal.json \
  --owner-private-key /secure/offline/owner-ed25519-private.pem \
  --approver-id owner@koschei \
  --output build/promotions/sentinel-v1.approval.json
```

The private key is read only for this explicit command. It is never embedded in the proposal, approval, repository, job envelope, adapter, or registry.

## 3. Verify independently

```bash
sentinel-promotion verify \
  --proposal build/promotions/sentinel-v1.proposal.json \
  --approval build/promotions/sentinel-v1.approval.json \
  --owner-public-key keys/owner-ed25519-public.pem
```

Verification checks both artifact digests, the candidate/proposal binding, the owner-key fingerprint, and the Ed25519 signature over the domain-separated proposal digest.

## Authority boundary

An approval proves that the named owner key explicitly approved offline shadow research for the exact finalized candidate. It does not:

- grant ARVIS verdict authority;
- serve customer traffic;
- replace a registry automatically;
- deploy an adapter;
- enable Web3 runtime integration;
- authorize production promotion.

A later production design must use a separate schema, separate review gates, explicit owner action, rollback evidence, and bounded canary controls. Passing this stage can never silently authorize that later stage.
