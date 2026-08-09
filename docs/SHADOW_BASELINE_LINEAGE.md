# Owner-Signed Shadow Baseline Lineage v1

`sentinel-shadow-baseline` records which reviewed shadow candidate is the current comparison baseline without allowing the model, CI, or a regression score to choose that baseline automatically.

A candidate can advance the lineage only when all of the following are true:

1. the candidate and current baseline have passing human-review scorecards;
2. the submitted regression report is recomputed from the exact supplied scorecards, receipts and tolerance, and the complete report must match that deterministic result;
3. the regression report passed on the same sealed replay dataset and binds the exact baseline/candidate scorecards and replay receipts;
4. when a lineage already exists, the report baseline is exactly the current lineage head;
5. the candidate has never appeared earlier in the lineage;
6. an owner Ed25519 key signs the exact baseline-advance proposal **and the approver identity**;
7. application re-verifies the proposal, signature, regression evidence, scorecards, receipts, replay identity and previous lineage digest;
8. every historical `advance` carries its proposal digest, approver identity, Ed25519 signature and approval digest, and the complete signature chain is re-verified before another candidate can be proposed or applied;
9. before a successor lineage is published, a shared canonical successor-claim directory atomically consumes the predecessor lineage digest. A second different successor from the same predecessor is rejected.

The first approved advance creates a two-entry lineage: the passing baseline is recorded as the unsigned `seed`, and the approved candidate becomes the first signed `advance`. Later approvals append one signed candidate at a time. The seed cannot claim a regression or owner approval that never occurred.

Each later proposal binds the digest of the complete previous lineage snapshot. During verification Koschei reconstructs every historical proposal from the parent/candidate scorecard and receipt digests, replay identity, regression digest and prior snapshot digest, then verifies the stored owner signature. Recomputing a forged unkeyed `lineage_digest` is therefore insufficient to forge history.

## Canonical successor claim

A signature proves that an owner authorized a candidate, but two separately signed candidates could otherwise fork from the same historical head. `sentinel-shadow-baseline apply` therefore requires `--claim-dir`. The claim filename is keyed by:

```text
owner key fingerprint
+ replay SHA-256
+ previous lineage digest
```

The file is installed with an atomic no-replace operation. The exact same claim is idempotent and can resume after an interrupted publication. A different successor attempting to consume the same predecessor digest fails closed. The claim is written **before** the successor lineage output is published.

The claim directory must be a shared canonical location for that baseline lineage. Using separate claim directories would intentionally create separate governance namespaces rather than one singular baseline history.

```bash
sentinel-shadow-baseline propose \
  --regression build/shadow/candidate.regression.json \
  --baseline-scorecard build/shadow/baseline.scorecard.json \
  --baseline-receipt build/shadow/baseline.receipt.json \
  --candidate-scorecard build/shadow/candidate.scorecard.json \
  --candidate-receipt build/shadow/candidate.receipt.json \
  --owner-public-key owner-public.pem \
  --output build/shadow/candidate.baseline-proposal.json

sentinel-shadow-baseline approve \
  --proposal build/shadow/candidate.baseline-proposal.json \
  --owner-private-key owner-private.pem \
  --approver-id owner@koschei \
  --output build/shadow/candidate.baseline-approval.json

sentinel-shadow-baseline apply \
  --proposal build/shadow/candidate.baseline-proposal.json \
  --approval build/shadow/candidate.baseline-approval.json \
  --regression build/shadow/candidate.regression.json \
  --baseline-scorecard build/shadow/baseline.scorecard.json \
  --baseline-receipt build/shadow/baseline.receipt.json \
  --candidate-scorecard build/shadow/candidate.scorecard.json \
  --candidate-receipt build/shadow/candidate.receipt.json \
  --owner-public-key owner-public.pem \
  --claim-dir build/shadow/baseline-claims \
  --output build/shadow/baseline-lineage.json

sentinel-shadow-baseline verify \
  --lineage build/shadow/baseline-lineage.json \
  --owner-public-key owner-public.pem
```

For later advances, pass the current immutable lineage to both `propose` and `apply` with `--lineage`, reuse the same canonical `--claim-dir`, and write the next lineage snapshot to a new path. Lineage artifacts and successor claims use atomic no-replace publication.

The lineage is research governance evidence only. Every artifact keeps these boundaries closed:

```text
historical_signatures_verified = true
automatic_baseline_selection_allowed = false
automatic_promotion_allowed = false
automatic_deployment_allowed = false
production_deployment_allowed = false
web3_runtime_integration_allowed = false
verdict_mutation_allowed = false
```

A passing regression therefore never silently becomes the new baseline. The owner must explicitly sign each advance, and each predecessor can be consumed by only one successor inside the canonical claim namespace.
