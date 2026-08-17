# Cyber Corpus v3 collection batches

Cyber Corpus v3 collection batches combine already pinned and extracted source releases into one audited training-corpus candidate. Collection remains separate from training authorization.

## Flow

1. Acquire a source only at its approved immutable revision.
2. Run the source-specific extractor to emit an artifact manifest and a training-corpus JSONL.
3. Build a collection-batch spec referencing the approved catalog and each extractor output.
4. Run `sentinel-cyber-batch-seal --spec <spec.json> --output-dir <dir>`.
5. Treat `seal.json.ready_for_training_pipeline=true` as permission to continue to later corpus composition/audit stages, not as permission to start model training.

The batch sealer revalidates source IDs and pinned revisions, verifies every corpus text SHA-256 against its manifest entry, rejects unauthorized corpus rows, detects duplicate artifact IDs, invokes the cross-source Cyber artifact audit, and detects duplicate content across sources. Output is deterministic: artifacts and training rows are sorted by artifact ID before hashing.

## Fail-closed behavior

A batch is not ready when a source is absent from the approved catalog, a source is not training-authorized, an artifact revision differs from the approved pin, an authorized artifact is missing from the corpus, an unauthorized artifact appears in the corpus, manifest and corpus hashes disagree, or duplicate content exists across sources.

## Batch 0001 extraction scope

The example Batch 0001 now contains four sources whose dedicated extraction boundaries are implemented:

- RustSec: advisory records with per-artifact CC0/CC-BY license resolution.
- MITRE ATT&CK: non-revoked, non-deprecated STIX security knowledge objects only.
- Kubernetes: English security/authentication/authorization/admission/policy documentation only; localization and unrelated website material are excluded.
- YARA: first-party documentation and `libyara` engine source only; vendor and third-party trees are excluded.

NIST CSF 2.0 and SP 800-61 Rev. 3 remain approved source families but are intentionally absent from Batch 0001 until the NIST artifact-level rights/text-export resolver is implemented. Their `PER_ARTIFACT` license scope must not be bypassed by a generic PDF or text extractor.
