# Sealed Shadow Replay Plan v1

`sentinel-shadow-plan` converts one owner-approved shadow-research promotion into a manually dispatched, offline-only replay plan.

```bash
sentinel-shadow-plan \
  --proposal build/promotions/candidate.proposal.json \
  --approval build/promotions/candidate.approval.json \
  --owner-public-key keys/owner-ed25519-public.pem \
  --replay fixtures/shadow/historical-cases.jsonl \
  --output-dir build/shadow/candidate \
  --output build/shadow/candidate.plan.json \
  --root .
```

Before producing `sentinel.shadow-replay-plan.v1`, the command:

- verifies the proposal and approval digests;
- verifies the Ed25519 owner signature against the supplied public key;
- requires the exact `shadow_research_candidate` stage and `explanation_only` authority;
- confines replay and output paths to the repository root;
- hashes the exact JSONL replay bytes;
- requires every replay row to be a JSON object with a unique `case_id` or `test_id`;
- rejects empty, oversized, duplicate, malformed, or non-UTF-8 replay datasets.

The plan permanently records:

```text
manual_dispatch_required = true
network_access_allowed = false
live_chain_reads_allowed = false
live_customer_traffic_allowed = false
verdict_mutation_allowed = false
automatic_deployment_allowed = false
production_deployment_allowed = false
web3_runtime_integration_allowed = false
```

This command plans research; it does not execute inference, read live chain state, serve customers, modify ARVIS verdicts, deploy an adapter, or connect Sentinel to Koschei Web3 Hub.
