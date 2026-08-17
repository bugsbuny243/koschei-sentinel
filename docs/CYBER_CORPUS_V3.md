# Koschei Sentinel Cyber Corpus v3

Cyber Corpus v3 is the data foundation for the large-model Sentinel architecture. The existing blockchain corpus remains a validated seed, not the final scope.

## Objective

Build a sealed, provenance-first cybersecurity corpus large and diverse enough to specialize a strong open-weight foundation model without teaching it to overclaim, memorize benchmarks, or confuse untrusted text with verified evidence.

The first release gate targets at least 50,000 approved documents from at least 120 distinct reviewed sources. The preferred target is 150,000 documents. Size is subordinate to provenance, licensing, security relevance, and diversity.

## Security domains

The collector must cover evidence grounding and incident response; endpoint/EDR and malware analysis; IAM, cloud, privileged access, keys and wallets; signing integrity; software supply chain; application, network and container security; cryptography and post-quantum security; blockchain protocol and bridge security; threat intelligence and vulnerability management.

No single source may dominate the release. No single domain should become a shortcut that makes Sentinel merely a Web3 model or merely an endpoint model.

## Provenance tiers

- `T0_PRIMARY`: normative specifications, primary source code/release material, first-party protocol or security artifacts.
- `T1_AUTHORITATIVE`: official security guidance, official advisories and authoritative public vulnerability material.
- `T2_REVIEWED`: high-quality incident postmortems or security research admitted after source and license review.
- `T3_CONTEXT_ONLY`: useful context that is not authorized for model training.

Mutable branches are never sufficient provenance. Every approved artifact must be pinned to a revision or release and hashed.

## License gate

Unknown licensing is not permission. Every source begins at `REVIEW_REQUIRED`. Training requires an explicit `ALLOW_TRAINING` or `ALLOW_WITH_ATTRIBUTION` decision and a recorded license reference. `EVAL_ONLY` and `BLOCKED` material are never mixed into training.

## Leakage gate

Benchmark prompts, answers, held-out incident cases, behavior-eval prompts, and their derived answer variants are excluded from training. Content and group hashes are checked across splits before a release can be sealed.

## Document contract

Every approved document must carry at least: source ID, pinned provenance, content SHA-256, source snapshot SHA-256, domain labels, source class, license decision, lineage, and review decision. Rejected material is retained in a rejection ledger so later builds cannot silently re-admit it.

## Collector phases

1. `catalog`: register candidate sources without downloading training material.
2. `license-review`: establish whether training is authorized.
3. `snapshot`: acquire only approved versioned/pinned artifacts.
4. `extract`: convert artifacts into normalized document records while preserving origin metadata.
5. `classify`: assign domain and security relevance labels.
6. `dedupe`: exact and near-duplicate filtering, including evaluation hashes.
7. `review`: reject noise, secrets, irrelevant code and ungrounded material.
8. `seal`: emit manifest, source catalog, rejection ledger and SHA-256 tree.

Collection and training are deliberately separate. A corpus may be collected and audited without granting training authorization.

## Relationship to older Sentinel artifacts

The 1,099-document blockchain corpus and its 7B adapter remain historical evidence and regression baselines. They are not discarded. Their held-out and behavior tests become part of the promotion suite for later large-model candidates.
